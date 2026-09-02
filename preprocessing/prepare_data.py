from datetime import datetime, timedelta
import numpy as np
import pandas as pd

"""  
    This file prepares the data for input to the NN models. Assumes that the data is already cleaned and in suitable folders.

"""

def downsample(df, option = 'mean'):

    """
        This function downsamples the data to the hourly mean.
    """

    df_downsample = pd.DataFrame()

    df = df.set_index('DATETIME')
    df = df.sort_index()
    df_downsample = df.copy()
    if option == 'mean':
        df_downsample = df.resample('1h').mean()
    if option == 'max':
        df_downsample = df.resample('1h').max()

    return df_downsample


def join_data(df_tide_gauge, df_weather):


    #### First downsample to hourly (weather is already hourly) #### -> NB: this is for the mean surge
    df_tide_gauge = df_tide_gauge[['DATETIME', 'SURGE', 'TIDE', 'Water_Level_OD_Malin','Water_Level_trend']]
    df_tide_gauge = downsample(df_tide_gauge, option = 'max')
    df_tide_gauge = df_tide_gauge.rename(columns = {'Water_Level_OD_Malin': 'TWL_OD'})
  
    df_weather = df_weather[['DATETIME', 'msl', 'u', 'v']]
    df_weather = df_weather.set_index('DATETIME')
    df_weather = df_weather.sort_index()

    ## add the wind stress
    df_weather['abs_w'] = np.sqrt(df_weather['u']**2 + df_weather['v']**2)
    ## drag coefficient
    df_weather['c'] = np.where(
        df_weather['abs_w'] < 7.5,
        0.001,
        0.00061 + 0.000063 * df_weather['abs_w']
    )
    p_air = 1.225 #density air kg/m
    df_weather['u_s'] = p_air*df_weather['c']*df_weather['abs_w']*df_weather['u']
    df_weather['v_s'] = p_air*df_weather['c']*df_weather['abs_w']*df_weather['v']

    #### Match the data by the hour ####
    df_merged = (
    df_tide_gauge
    .merge(df_weather, how='left', left_index=True, right_index=True)
    )

    #### Add back in DATETIME as a column ####
    df_merged = df_merged.reset_index()
    #### Detrend the data ####
    temp = df_merged.copy()
    return temp


def main(stations = None):


    if stations == None:
        ## if subset of stations not provided, run for all stations
        stations = ['Aranmore', 'Ballycotton', 'Ballyglass', 'Castletownbere', 'Dunmore', 'Carrigaholt',  'Dublin Port',  'Galway Port', 'Howth', 'Inishmore', 
                'Killybegs Port', 'Malin Head',  'Skerries Harbour', 'Sligo', 'Wexford', 'Fenit','Ferry Bridge Maigue', 'Foynes', 'Moneycashen', 'Port Bridge Swilly', 'Port Oriel', 'Ringaskiddy NMCI','Rossaveel Pier']


    folder_base = '../Data/PreProcessed/'
    folder_models = '../Data/Model_data/'

    print('Preparing data for input to the model: ', stations)

    for station in stations:
        print('.....', station, '....')
        ### load all the data ###
        df_tide_gauge = pd.read_csv(folder_base + station +'/tide_gauge.csv')
        df_weather = pd.read_csv(folder_base + station +'/ecmwf.csv')
        ### make sure they have the same DATETIME format
        df_tide_gauge['DATETIME'] = pd.to_datetime(df_tide_gauge['DATETIME']).dt.tz_localize(None)
        df_weather['DATETIME'] = pd.to_datetime(df_weather['DATETIME']).dt.tz_localize(None)




        
        df_merged = join_data(df_tide_gauge=df_tide_gauge, df_weather=df_weather)
        df_merged['DATETIME'] = pd.to_datetime(df_merged['DATETIME'])
        
        #### seperate the final year and a half for final validation ####
        if station in ['Ballyglass', 'Castletownbere'] :
            df_val = df_merged[df_merged['DATETIME'].dt.year == 2024]
            df_training = df_merged[df_merged['DATETIME'].dt.year <=2023]
        else:
            df_val = df_merged[df_merged['DATETIME'].dt.year == 2025]
            df_training = df_merged[df_merged['DATETIME'].dt.year <=2024]

        df_val = df_val[['DATETIME', 'SURGE','TIDE','TWL_OD','msl', 'u', 'v', 'abs_w', 'u_s', 'v_s', 'Water_Level_trend']]
        df_val.to_csv(folder_models + station +'/validation_data.csv', index = False)

        # just using tide msl and wind as training parameters 
        df_training = df_training[['DATETIME', 'SURGE','TIDE', 'TWL_OD', 'msl', 'u', 'v', 'abs_w', 'u_s', 'v_s','Water_Level_trend']]
        df_training.to_csv(folder_models + station +'/training_data.csv', index = False)

        print('-----------------------------------------------------')


if __name__ == '__main__':
    main()







