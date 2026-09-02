import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout
import os
import pickle
import keras

def load_model_weights(model_path, input_shape): #input_shape=input_shape, 

    model = Sequential([
        keras.Input(shape = input_shape),
        LSTM(48, activation='relu', 
             recurrent_activation='hard_sigmoid', return_sequences=False),
        Dense(16),
        Dropout(0.1),
        Dense(1)
    ])

    model.load_weights(model_path)
    return model

def get_metrics(df_merged, model_cols, col = 'SURGE'):

    """
        The comparison metrics used are correlation coefficient (R), RMSE, mean error (ME), 
        standard deviation of error (SDE), maximum error (maxE).
    
    """

    ### Drop columns that have NaN for any of the models to ensure
    ### we are evaluating the same points 
    df_merged = df_merged.dropna(subset = model_cols)
    df_merged['DATETIME'] = pd.to_datetime(df_merged['DATETIME'])
    df_merged = df_merged.set_index('DATETIME')


    results = pd.DataFrame(columns=['Model', 'Sample Size', 'Correlation', 'RMSE', 'Mean Error', 'SD',
                                        'Max Error', 'Max Error Date'])
   
    for model in model_cols:

        correlation = np.corrcoef(df_merged[col], df_merged[model])[0, 1]
        mean_error = np.mean(df_merged[model] - df_merged[col])
        sd_error = np.std(df_merged[model] - df_merged[col])
        rmse = np.sqrt(((df_merged[model] - df_merged[col]) ** 2).mean())
        error = df_merged[model] - df_merged[col]
        max_error = error.loc[np.abs(error).idxmax()] ## keeping the sign
        max_error_datetime = (abs(df_merged[model] - df_merged[col])).idxmax()

        # Add results to the table -full
        results = pd.concat([results, pd.DataFrame({
        'Model': model,
        'Column': col,
        'Sample Size': [len(df_merged[col])],
        'Correlation': [correlation],
        'Mean Error': [mean_error],
        'SD': [sd_error],
        'RMSE': [rmse],
        'Max Error': [max_error],
        'Max Error Date': [max_error_datetime]
        })], ignore_index=True)

    return results


def split_sequence_consecutive(df, input_cols, target_col, window_size):
    df = df.sort_index()
    time_diff = df.index.to_series().diff().dt.total_seconds() / 3600.0
    breaks = np.where(time_diff != 1)[0]
    segments = np.split(df, breaks)

    X_list, y_list, ts_list = [], [], []
    n_features = len(input_cols)

    for seg in segments:
        if len(seg) < window_size:
            continue

        data = seg[input_cols].to_numpy()
        targets = seg[target_col].to_numpy()

        num_samples = len(data) - window_size + 1  # +1 to include last timestep
        if num_samples <= 0:
            continue

        # Create sliding windows
        shape = (num_samples, window_size, n_features)
        strides = (data.strides[0], data.strides[0], data.strides[1])
        X_seg = np.lib.stride_tricks.as_strided(data, shape=shape, strides=strides)

        # Align y with the last timestep in the sequence
        y_seg = targets[window_size - 1 : window_size - 1 + num_samples]
        ts_seg = seg.index[window_size - 1 : window_size - 1 + num_samples]

        X_list.append(X_seg)
        y_list.append(y_seg)
        ts_list.append(ts_seg)

    X = np.concatenate(X_list, axis=0)
    y = np.concatenate(y_list, axis=0)
    timestamps = np.concatenate(ts_list, axis=0)

    return X, y, timestamps


def model_surge(model_folder, df_input, input_cols, num_models = 20):

    df_input['DATETIME'] = pd.to_datetime(df_input['DATETIME'])
    df_input = df_input.set_index('DATETIME')
    df_input = df_input.dropna(subset = input_cols)
    df_input = df_input.sort_index()
    X_test, y_test, ts_test = split_sequence_consecutive(df_input, input_cols,  'SURGE', 25)
    
    # Load saved scaler
    with open(model_folder + "scaler_X.pkl", "rb") as f:
        scaler_X = pickle.load(f)
    with open(model_folder + "scaler_y.pkl", "rb") as f:
        scaler_y = pickle.load(f)

    # Flatten test data and scale
    n_features = X_test.shape[2]
    X_test_flat = X_test.reshape(-1, n_features)
    X_test_scaled = (
        scaler_X.transform(X_test_flat)
        .reshape(X_test.shape)
    )

    ensemble_preds = []

    for n in range(num_models):
        print(f"\n Loading model {n+1}/{num_models}")
        # Build and train model
        model_path = model_folder + 'Weights/model_' + str(n+1) + '.weights.h5'
        model = load_model_weights(model_path, input_shape=(X_test_scaled.shape[1], X_test_scaled.shape[2]) )
        # Predict on test set 
        y_pred_scaled = model.predict(X_test_scaled, verbose=0).flatten()
        # Back-transform to original scale
        y_pred = scaler_y.inverse_transform(y_pred_scaled.reshape(-1, 1)).flatten()
        ensemble_preds.append(y_pred)

    ensemble_preds = np.array(ensemble_preds)
    y_pred_mean = np.mean(ensemble_preds, axis=0)
    y_pred_std = np.std(ensemble_preds, axis=0)
    y_pred_max = np.max(ensemble_preds, axis = 0)
    y_pred_min = np.min(ensemble_preds, axis = 0)

    ## Evaluate the performance (save to dataframe )
    df_validation = df_input.copy()
    df_validation['LSTM_mean'] = np.nan
    df_validation['LSTM_std'] = np.nan
    df_validation.loc[ts_test, 'LSTM_mean'] = y_pred_mean
    df_validation.loc[ts_test, 'LSTM_std'] = y_pred_std
    df_validation.loc[ts_test, 'LSTM_min'] = y_pred_min
    df_validation.loc[ts_test, 'LSTM_max'] = y_pred_max

    df_validation['DATETIME'] = df_validation.index

    return df_validation

def get_holdout_results(station, num_models = 20, folder_model = 'examples/Data/Model_Data/', met_data = 'era5'):

    ## Load the models for the station and predict for the holdout year (2024 or 2025)
    if met_data == 'hres':
        input_folder = folder_model + station + '/hres_validation_data.csv'
    else:
        input_folder = folder_model+station+'/validation_data.csv'
    output_folder = 'examples/Forecast_Results/'+station+'/'+met_data+'/'


    df_input = pd.read_csv(input_folder)
    df_input['DATETIME'] = pd.to_datetime(df_input['DATETIME'])
    surge_columns = ['LSTM_mean', 'LSTM_max', 'LSTM_min']
    twl_columns = ['TWL_LSTM_mean', 'TWL_LSTM_min', 'TWL_LSTM_max']

    file_path_surge = output_folder + '/surge_results.csv'
    file_path_twl = output_folder+'/twl_results.csv'
    
    
    print('Loading and evaluating the models...')
    model_folder = folder_model+station+'/Trained_Models/'
    model_columns = [ 'msl', 'u_s', 'v_s','TIDE']
    df_surge = model_surge(model_folder, df_input, input_cols = model_columns, num_models=num_models)
    os.makedirs(output_folder, exist_ok=True)
    df_surge = df_surge.reset_index(drop=True)

    df_training = pd.read_csv(folder_model+station+'/training_data.csv')
    df_training['DATETIME'] = pd.to_datetime(df_training['DATETIME'])
    
    #get the last months baseline value
    baseline = df_training['Water_Level_trend'].dropna().iloc[-1]
    df_surge['TWL_LSTM_mean'] = df_surge['TIDE']+df_surge['LSTM_mean'] + baseline
    df_surge['TWL_LSTM_max'] = df_surge['TIDE']+df_surge['LSTM_min'] + baseline
    df_surge['TWL_LSTM_min'] = df_surge['TIDE']+df_surge['LSTM_max'] + baseline

    full_metrics_surge = get_metrics(df_surge, surge_columns, col = 'SURGE') 
    full_metrics_surge.to_csv(file_path_surge, index = False)
    full_metrics_twl = get_metrics(df_surge, twl_columns, col = 'TWL_OD') 
    full_metrics_twl.to_csv(file_path_twl, index = False)

    df_surge.to_csv(output_folder+'/lstm_surge.csv', index = False)

    return 0

def table_results(
    stations,
    metric,
    output_file,
    data_file='era5',
    model='LSTM_mean',  
):
    metrics = [
        'Correlation',
        'RMSE',
        'Mean Error',
        'Max Error',
        'Max Error Date'
    ]

    file = f'{metric}_results.csv'
    rows = []

    for station in stations:

        file_path = f'examples/Forecast_Results/{station}/{data_file}/{file}'

        try:
            df = pd.read_csv(file_path)
        except FileNotFoundError:
            print(f'File not found for {station}: {file_path}')
            continue

        model_rows = df[df['Model'] == model]

        if model_rows.empty:
            print(f'{model} not found for {station}')
            continue

        model_row = model_rows.iloc[0]

        row_data = {
            'Station': station,
            'Sample Size': df['Sample Size'].iloc[0]
        }

        for metric_name in metrics:
            row_data[metric_name] = model_row[metric_name]

        rows.append(row_data)

    if not rows:
        return pd.DataFrame()

    results_df = pd.DataFrame(rows)

    # Convert Max Error Date
    if 'Max Error Date' in results_df.columns:
        results_df['Max Error Date'] = (
            pd.to_datetime(
                results_df['Max Error Date'],
                errors='coerce'
            )
            .dt.strftime('%Y-%m-%d')
        )

    # Round numeric metrics
    numeric_metrics = [
        'Correlation',
        'RMSE',
        'Mean Error',
        'Max Error',
        'Sample Size'
    ]

    for metric_name in numeric_metrics:
        if metric_name in results_df.columns:
            results_df[metric_name] = pd.to_numeric(
                results_df[metric_name],
                errors='coerce'
            ).round(2)

    # Sort by station alphabetically
    results_df = (
        results_df
        .sort_values('Station')
        .reset_index(drop=True)
    )

    results_df.to_csv(output_file)

    return 0




def main(stations = None):


    if stations == None:
        ## if subset of stations not provided, run for all stations
        stations = ['Aranmore', 'Ballycotton', 'Ballyglass', 'Castletownbere', 'Dunmore', 'Carrigaholt',  'Dublin Port',  'Galway Port', 'Howth', 'Inishmore', 
                'Killybegs Port', 'Malin Head',  'Skerries Harbour', 'Sligo', 'Wexford', 'Fenit','Ferry Bridge Maigue', 'Foynes', 'Moneycashen', 'Port Bridge Swilly', 'Port Oriel', 'Ringaskiddy NMCI','Rossaveel Pier']


    """ To get the ERA5 results """
    for station in stations:
        get_holdout_results(station, num_models = 20, folder_model = 'examples/Data/Model_Data/', met_data = 'era5')

    """ To get the HRES results """
    for station in stations:
        get_holdout_results(station, num_models = 20, folder_model = 'examples/Data/Model_Data/', met_data='hres')


    table_results(stations = stations, metric='surge', data_file='era5', model='LSTM_mean', output_file = 'Data/Forecast_Results/era5_surge.csv')
    table_results(stations = stations, metric='twl', data_file='era5', model='TWL_LSTM_mean', output_file = 'Data/Forecast_Results/era5_TWL.csv')

    table_results(stations = stations, metric='surge', data_file='hres', model='LSTM_mean', output_file = 'Data/Forecast_Results/hres_surge.csv')
    table_results(stations = stations, metric='twl', data_file='hres', model='TWL_LSTM_mean', output_file = 'Data/Forecast_Results/hres_TWL.csv')


if __name__ == '__main__':
    main()