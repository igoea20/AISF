import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout
from keras import regularizers, metrics, optimizers
import keras_tuner as kt
from tensorflow.keras.optimizers import Adam
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
from pathlib import Path
import pickle


"""
    This python file trains an ensemble of models.
"""

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



def build_lstm_model(input_shape, lstm_hyperparameters):

    """
        Sets up the LSTM model with the provided architecture and hyperparameters.
    
    """

    model = Sequential([
        LSTM(lstm_hyperparameters['LSTM_layers'], activation=lstm_hyperparameters['LSTM_activation'], 
             recurrent_activation=lstm_hyperparameters['LSTM_recurrent_activation'], input_shape=input_shape, return_sequences=False),
        Dense(lstm_hyperparameters['DENSE_layers']),
        Dropout(lstm_hyperparameters['DROPOUT_rate']),
        Dense(1)
    ])

    optimizer = Adam(learning_rate = lstm_hyperparameters['lr'])

    model.compile(optimizer=optimizer, loss='mse', metrics=[

        metrics.MeanSquaredError(),
       metrics.R2Score(
    class_aggregation="uniform_average", num_regressors=0, name="r2_score", dtype=None
    )])
    return model



def train_models(train_file, input_cols, lstm_hyperparameters, model_folder, num_models = 20):

    """
    This trains 20 models, each on 50% of data (shuffles index).
    """

    df = pd.read_csv(train_file)
    df['DATETIME'] = pd.to_datetime(df['DATETIME'])
    df = df.set_index('DATETIME')
    df = df.dropna(subset=input_cols + ['SURGE'])
    X_trainval, y_trainval, _ = split_sequence_consecutive(df, input_cols, 'SURGE', 25)


    scaler_X = StandardScaler()
    n_features = X_trainval.shape[2]

    # Fit scaler on the training+validation data and save for future
    X_trainval_flat = X_trainval.reshape(-1, n_features)
    scaler_X.fit(X_trainval_flat)
    X_trainval_scaled = scaler_X.transform(X_trainval_flat).reshape(X_trainval.shape)
    with open(model_folder + "scaler_X.pkl", "wb") as f:
        pickle.dump(scaler_X, f)    

    # Target normalization 
    scaler_y = StandardScaler()
    y_trainval_scaled = scaler_y.fit_transform(y_trainval.reshape(-1, 1)).flatten()
    with open(model_folder + "scaler_y.pkl", "wb") as f:
        pickle.dump(scaler_y, f)   

    # Target normalization
    scaler_y = StandardScaler()
    y_trainval_scaled = scaler_y.fit_transform(y_trainval.reshape(-1, 1)).flatten()


    for n in range(num_models):
        print(f"\n Training model {n+1}/{num_models}")

        # Randomly select 50% of train+val data
        total_samples = len(X_trainval_scaled)
        subset_idx = np.random.choice(total_samples, size=int(0.5 * total_samples), replace=False)

        X_subset = X_trainval_scaled[subset_idx]
        y_subset = y_trainval_scaled[subset_idx]

        # Split 70/30 into train and val
        n_train = int(0.7 * len(X_subset))
        idx = np.arange(len(X_subset))
        np.random.shuffle(idx)


        X_train = X_subset[idx[:n_train]]
        y_train = y_subset[idx[:n_train]]
        X_val = X_subset[idx[n_train:]]
        y_val = y_subset[idx[n_train:]]
        
        # Build and train model
        model = build_lstm_model((X_trainval_scaled.shape[1], X_trainval_scaled.shape[2]), lstm_hyperparameters)

        early_stop = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss',
            patience=3,          # stop if no improvement after 3 epochs
            restore_best_weights=True
        )

        history = model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=50,
            batch_size=240,
            verbose=1,
            shuffle=True,  # shuffle sequences (batch-wise)
            callbacks=[early_stop]
        )

        ## save model weights
        model.save_weights(model_folder+'Weights/' + 'model_' + str(n+1) + '.weights.h5')

    return 0


def main(stations = None):


    if stations == None:
        ## if subset of stations not provided, run for all stations
        stations = ['Aranmore', 'Ballycotton', 'Ballyglass', 'Castletownbere', 'Dunmore', 'Carrigaholt',  'Dublin Port',  'Galway Port', 'Howth', 'Inishmore', 
                'Killybegs Port', 'Malin Head',  'Skerries Harbour', 'Sligo', 'Wexford', 'Fenit','Ferry Bridge Maigue', 'Foynes', 'Moneycashen', 'Port Bridge Swilly', 'Port Oriel', 'Ringaskiddy NMCI','Rossaveel Pier']

    ### Train the 20 models and evaluate
    input_cols = [ 'msl', 'u_s', 'v_s', 'TIDE'] 


    lstm_hyperparameters = {'LSTM_layers': 48, 'LSTM_activation': 'relu', 
                            'LSTM_recurrent_activation': 'hard_sigmoid', 
                            'DENSE_layers': 16, 'DROPOUT_rate': 0.1, 'lr': 0.001}

    for station in stations:
        print(station)

        train_file = '../Data/Model_Data/' + station + '/training_data.csv'
        model_folder = '../Data/Model_Data/' + station + '/Trained_Models/'

        output_dir = model_folder + 'Weights'
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        ### Train each of the 20 models and save them to a file
        train_models(train_file, input_cols, lstm_hyperparameters, model_folder, num_models=20)

if __name__ == '__main__':
    main()



