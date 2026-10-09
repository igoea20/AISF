from pathlib import Path
import sys, os
sys.path.append("../../")
import ecmwfapi
import numpy as np
import pandas as pd
import xarray as xr
from dotenv import load_dotenv
import os
from datetime import datetime, timedelta
import hatyan
import pickle
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from evaluation.evaluate import model_surge
from preprocessing.process_data import apply_empirical_qm


DEFAULT_STEP = 72
DEFAULT_AREA = "55/349/51/355" #to cover Ireland
##MARS_KEY = ####
##MARS_EMAIL = #### 

PARAMETERS = {
    "u": "165.128",      # 10 m u-component of wind
    "v": "166.128",      # 10 m v-component of wind
    "msl": "151.128",    # mean sea-level pressure
}





def download_weather_data(
    forecast_date,
    output_file,
    step=DEFAULT_STEP,
    area=DEFAULT_AREA,
    forecast_hour = '00z'
):

    """
    Download HRES forecast data for a given step count. 

    Data are downloaded from MARS in one request for the given area

    Parameters
    ----------
    forecast date : str
        Start date, e.g. "2025-01-01".
    output_file : Path
        Destination GRIB file.
    step : int
        Maximum forecast lead time in hours.
    area : str
        ECMWF area string.

    Notes
    -----
    This function assumes the Mars Key and Email details are save as env variables.
    """

    
    output_file = Path(output_file)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    param_string = "/".join(PARAMETERS.values())
    step_range = f"0/to/{step}/by/1"



    server = ecmwfapi.ECMWFService(
        service="mars",
        key=MARS_KEY, #os.environ["MARS_KEY"],
        url="https://api.ecmwf.int/v1",
        email=MARS_EMAIL #os.environ["MARS_EMAIL"],
    )

    if forecast_hour == '00z':
        time_param = "00:00:00"
    elif forecast_hour == '06z':
        time_param = "06:00:00"
    elif forecast_hour == '12z':
        time_param = "12:00:00"
    elif forecast_hour == '18z':
        time_param = "18:00:00"
    else:
        print("Incorrect formatting of forecast hour: ", forecast_hour, ". Running of 00z.")
        time_param = "00:00:00"

    server.execute(
        {
            "class": "od",
            "stream": "oper",
            "expver": "1",
            "levtype": "sfc",
            "param": param_string,
            "step": step_range,
            "time": time_param,
            "type": "fc",
            "target": "output",
            "grid": "0.25/0.25",
            "area": area,
            "date": forecast_date,
            "expect": "any",
        },
        str(output_file),
    )

    return output_file

def select_station_points(ds, station_locs):
    """
    Select the HRES grid point nearest to each station.

    Parameters
    ----------
    ds : xarray.Dataset
        HRES GRIB dataset.
    station_locs : pandas.DataFrame
        Must contain:
            Station
            era5_latitude
            era5_longitude

    Returns
    -------
    xarray.Dataset
        Dataset containing only the selected station grid points.
        Dimensions are station and forecast time.
    """

    stations = station_locs[
        ["Station", "era5_latitude", "era5_longitude"]
    ].copy()

    # ---------------------------------------------------------
    # HRES grid coordinates
    # ---------------------------------------------------------

    grid_latitude = ds["latitude"].values
    grid_longitude = ds["longitude"].values

    # HRES regular grid should have 1-D latitude/longitude.
    if grid_latitude.ndim != 1 or grid_longitude.ndim != 1:
        raise ValueError(
            "Expected 1-D latitude/longitude coordinates in the GRIB. "
            f"Got latitude.ndim={grid_latitude.ndim}, "
            f"longitude.ndim={grid_longitude.ndim}."
        )

    # ---------------------------------------------------------
    # Convert station longitude to the GRIB convention
    # ---------------------------------------------------------

    station_latitude = stations["era5_latitude"].to_numpy()
    station_longitude = stations["era5_longitude"].to_numpy()

    grib_uses_360 = grid_longitude.max() > 180

    if grib_uses_360:
        station_longitude = station_longitude % 360
    else:
        station_longitude = (
            (station_longitude + 180) % 360
        ) - 180

    # ---------------------------------------------------------
    # Find nearest grid point
    # ---------------------------------------------------------

    latitude_indices = np.abs(
        grid_latitude[:, None]
        - station_latitude[None, :]
    ).argmin(axis=0)

    longitude_indices = np.abs(
        grid_longitude[:, None]
        - station_longitude[None, :]
    ).argmin(axis=0)

    nearest_latitude = grid_latitude[latitude_indices]
    nearest_longitude = grid_longitude[longitude_indices]

    # ---------------------------------------------------------
    # Select the required grid points
    # ---------------------------------------------------------

    selected = ds.isel(
        latitude=xr.DataArray(
            latitude_indices,
            dims="station",
        ),
        longitude=xr.DataArray(
            longitude_indices,
            dims="station",
        ),
    )

    #station metadata
    selected = selected.assign_coords(
        Station=(
            "station",
            stations["Station"].to_numpy(),
        ),
        era5_latitude=(
            "station",
            stations["era5_latitude"].to_numpy(),
        ),
        era5_longitude=(
            "station",
            stations["era5_longitude"].to_numpy(),
        ),
        hres_latitude=(
            "station",
            nearest_latitude,
        ),
        hres_longitude=(
            "station",
            nearest_longitude,
        ),
    )

    return selected

def calculate_wind_stress(df):
    """
    Calculate wind stress components from 10 m wind.
    """

    df = df.copy()
    df["abs_w"] = np.hypot(df["u"], df["v"])
    df["c"] = np.where(
        df["abs_w"] < 7.5,
        0.001,
        0.00061 + 0.000063 * df["abs_w"],
    )
    air_density = 1.225  # kg/m^3
    df["u_s"] = (air_density* df["c"]* df["abs_w"]* df["u"])
    df["v_s"] = (air_density* df["c"]* df["abs_w"]* df["v"])

    return df


def convert_weather(grib_file, weather_file,station_locs_file):
    """
    Convert one HRES GRIB file into a station-level DataFrame.
    Only the nearest HRES grid point to each station is retained.
    """

    grib_file = Path(grib_file)
    station_locs = pd.read_csv(station_locs_file)
    print(f"Converting {grib_file} ...")

    required_variables = ["u10","v10","msl",]
    ds = xr.open_dataset(
        grib_file,
        engine="cfgrib",
        backend_kwargs={
            "indexpath": "",
            "filter_by_keys": {
                "typeOfLevel": "surface",
            },
        },
    )
    print("Dataset coordinates:", list(ds.coords))
    print("Dataset dimensions:", list(ds.dims))

    try:
        # only station grid points
        selected = select_station_points(ds, station_locs)
        # Keep only the variables we actually need.
        selected = selected[required_variables]
        weather = (selected.to_dataframe().reset_index())

    finally:
        ds.close()

    weather = weather.rename(
        columns={
            "u10": "u",
            "v10": "v",
            "station_latitude": "era5_Latitude",
            "station_longitude": "ECMWF Longitude",
            "step": "Step"
        }
    )
    weather["DATETIME"] = pd.to_datetime(weather["valid_time"])

    # Convert input variables
    weather["msl"] = weather["msl"] / 100.0
    weather = calculate_wind_stress(weather)

    weather = weather[
        [
            "DATETIME",
            "Step",
            "Station",
            "u_s",
            "v_s",
            "msl",
        ]
    ]
    weather = weather.dropna()

    weather = (
        weather
        .drop_duplicates(
            subset=["Station","DATETIME",],
            keep="last",
        )
        .sort_values(["Station","DATETIME",])
        .reset_index(drop=True)
    )

    weather.to_csv(weather_file, index = False)

    ## Convert hres to eqm hres 
    output_file = weather_file.with_name(
        f"{weather_file.stem}_eqm{weather_file.suffix}"
    )

    df_weather = fit_eqm(
        weather_file=weather_file,
        output_file=output_file,
        qm_file = '../../Data/EQM/qm_parameters.pkl'
    )

    return 0


def fit_eqm(weather_file, output_file,qm_file="qm_parameters.pkl"):
    """
    Loads previously fitted EQM parameters and applies them
    to all stations in the HRES weather file.
    """

    with open(qm_file, "rb") as f:
        qm_params = pickle.load(f)

    df = pd.read_csv(weather_file)

    vars_to_correct = ["u_s", "v_s", "msl"]

    corrected = []

    for station, station_df in df.groupby("Station", sort=False):

        
        if station not in qm_params:
                print(f"No EQM parameters found for station: {station}")
                corrected.append(station_df)
        else:
            station_df = station_df.copy()

            for var in vars_to_correct:

                if var not in station_df.columns:
                    raise ValueError(
                        f"{var} not found in HRES data for {station}"
                    )

                if var not in qm_params[station]:
                    raise ValueError(
                        f"No EQM parameters found for "
                        f"{station} / {var}"
                    )

                params = qm_params[station][var]

                station_df[var] = apply_empirical_qm(
                    station_df[var],
                    params["hres_sorted"],
                    params["era5_sorted"],
                    params["probabilities"],
                )

            corrected.append(station_df)

    df_corrected = pd.concat(
        corrected,
        ignore_index=True,
    )

    df_corrected.to_csv(weather_file, index=False)

    return 0


def get_tide(tide_file, date, harmonic_analysis_dir):

    """ 
        Calculate the tide forecast for the given month, for all stations. 
        Using the harmonic analysis of the stations.
    """

    # get all the files in the harmonic folder
    date = pd.Timestamp(date)

    start_date = date.normalize()
    start_date = date.replace(
        day=1,
        hour=0,
        minute=0,
        second=0,
        microsecond=0,
    )
    end_date = start_date + pd.offsets.MonthEnd(1) + pd.Timedelta(hours=23)

    future_times = pd.date_range(
        start=start_date,
        end=end_date,
        freq="5min",
    )

    # Hatyan expects timezone-naive timestamps in your workflow
    if future_times.tz is not None:
        future_times = (
            future_times
            .tz_convert("UTC")
            .tz_localize(None)
        )

    analysis_files = sorted(
        harmonic_analysis_dir.glob("*_harmonic_analysis.pkl")
    )

    if not analysis_files:
        raise FileNotFoundError(
            f"No harmonic analysis files found in {harmonic_analysis_dir}"
        )
    all_forecasts = []

    for analysis_file in analysis_files:
        #calculate for each station

        station_name = analysis_file.stem.replace(
            "_harmonic_analysis", "",)

        print(f"Forecasting tide for {station_name}...")

        try:
            with open(analysis_file, "rb") as f:
                analysis = pickle.load(f)

            comp = analysis["comp"]
            mean_offset = analysis.get("mean_offset",0.0,)

            tide = hatyan.prediction(comp=comp,times=future_times,)

            # Apply stored mean offset
            tide["values"] += mean_offset

            station_forecast = pd.DataFrame(
                {
                    "DATETIME": future_times,
                    "Station": station_name,
                    "Tide": tide["values"].to_numpy(),
                }
            )
            all_forecasts.append(station_forecast)


        except Exception as e:
            print(
                f"  ERROR processing {station_name}: {e}"
            )
    if not all_forecasts:
        raise RuntimeError(
            "No station forecasts were successfully generated."
        )

    tide_forecast = pd.concat(
        all_forecasts,
        ignore_index=True,
    )

    tide_forecast = (
        tide_forecast
        .sort_values(
            ["Station", "DATETIME"]
        )
        .reset_index(drop=True)
    )
    print(tide_forecast)


    ## sub sample to hourly maximum 
    hourly_tide_forecast = tide_forecast.copy()
    hourly_tide_forecast['HOUR'] = tide_forecast['DATETIME'].dt.floor("h")
    hourly_tide_forecast = (hourly_tide_forecast.groupby(
        ['Station', 'HOUR'], as_index = False)
        ["Tide"].max() #hourly maximum 
        .rename(columns = {'HOUR': 'DATETIME'})
    )
    print(hourly_tide_forecast)

    hourly_tide_forecast.to_csv(
        tide_file,
        index=False,
    )

    return 0


def get_input_data(date, date_str, output_dir, input_file,forecast_steps = 72, forecast_hour = '00z'):

    """
        For the given date this function creates the necessary data file
        for the model and saves it to folder.
    """
    weather_dir =  output_dir / "Weather"
    station_locs_file = 'gauge_info.csv'
    weather_dir.mkdir(parents=True, exist_ok=True)

    grib_name = (f"hres_{date_str}.grib")
    grib_file = weather_dir / grib_name
    weather_file = weather_dir / f"hres_{date_str}.csv"
    tide_dir = Path('Outputs/Data/Tide/') / str(date.year) 
    tide_dir.mkdir(parents=True, exist_ok=True)
    tide_file = tide_dir / (f"{str(date.month)}.csv")
    harmonic_analysis_dir = Path('../../Data/Harmonic_Analysis/')

    if not weather_file.exists():
        if not grib_file.exists():
            """ Do not redownload data that's already there. """
            download_weather_data(
                forecast_date=str(date),
                output_file=grib_file,
                step=forecast_steps,
                forecast_hour=forecast_hour
            )
        else:
            print(
                f"Already downloaded: {grib_file}"
            )
        ## Convert the GRIB file and save it
        convert_weather(grib_file, weather_file, station_locs_file=station_locs_file)
    else:
        print(
            f"Already downloaded and converted: {weather_file}"
        )

    # Download and convert the previous day's weather for model context.
    previous_date = pd.Timestamp(date) - pd.Timedelta(days=1)
    previous_date_str = str(previous_date.date()).replace("-", "_")
    previous_folder = Path("Outputs/Data")/ previous_date_str 
    previous_grib_file =  previous_folder / "Weather" / f"hres_{previous_date_str}.grib"
    previous_weather_file =previous_folder / "Weather" / f"hres_{previous_date_str}.csv"

    if not previous_weather_file.exists():
        if not previous_grib_file.exists():
            download_weather_data(
                forecast_date=str(previous_date.date()),
                output_file=previous_grib_file,
                step=24,
                forecast_hour=forecast_hour,
            )

        convert_weather(
            previous_grib_file,
            previous_weather_file,
            station_locs_file=station_locs_file,
        )

    # Using the tide harmonics to predict tide, could also use tide forecasts.
    if not tide_file.exists():
        get_tide(tide_file, date, harmonic_analysis_dir)


    # Load the previous day's weather and retain only
    # timestamps before the current forecast date.
    df_previous = pd.read_csv(previous_weather_file)
    df_previous["DATETIME"] = pd.to_datetime(df_previous["DATETIME"])

    current_start = pd.Timestamp(date).normalize()
    df_previous = df_previous[
        df_previous["DATETIME"] < current_start
    ]

    # Load current-day weather.
    df_weather = pd.read_csv(weather_file)
    df_weather["DATETIME"] = pd.to_datetime(df_weather["DATETIME"])

    # Combine historical context and current forecast.
    df_weather = pd.concat(
        [df_previous, df_weather],
        ignore_index=True,
    )

    df_weather = (
        df_weather
        .drop_duplicates(
            subset=["Station", "DATETIME"],
            keep="last",
        )
        .sort_values(["Station", "DATETIME"])
        .reset_index(drop=True)
    )

    df_tide = pd.read_csv(tide_file)
    df_tide["DATETIME"] = pd.to_datetime(df_tide["DATETIME"])

    df_input = df_weather.merge(
        df_tide,
        on=["DATETIME", "Station"],
        how="inner",
    )

    df_input["SURGE"] = -99
    df_input.to_csv(input_file, index=False)

    return 0

def get_figures(stations, date_str, hour_str):

    """ 
        Plots the surge and TWL for the given stations, starting on the given date. 
        Saves to file as png.
    """
    
    output_path = Path('Outputs/Figures/'+date_str+ '/' + hour_str )
    output_path.mkdir(parents=True, exist_ok=True)

    for station in stations:
        input_file = 'Outputs/Data/'+date_str+ '/' + hour_str + '/Forecast/'+station+'.csv'
        output_surge_file = output_path/f'{station}_surge.png'
        output_twl_file = output_path/f'{station}_TWL.png'
    
        df = pd.read_csv(input_file)
        df['DATETIME'] = pd.to_datetime(df['DATETIME'])

        ### Plot the surge
        fig, ax = plt.subplots(figsize=(10, 4))

        ax.plot(df['DATETIME'], df['LSTM_mean'], 'k', label='Ensemble mean')

        ax.fill_between(
            df['DATETIME'],
            df['LSTM_min'],
            df['LSTM_max'],
            color='blue',
            alpha=0.6,
            label='Ensemble Range'
        )

        #ax.xaxis.set_major_locator(mdates.HourLocator(interval=12))
        #ax.xaxis.set_major_formatter(mdates.DateFormatter('%d %b\n%H:%M'))
        #ax.xaxis.set_minor_locator(mdates.HourLocator(interval=1))
        ax.set_ylabel("Surge (m)", fontsize=18)
        ax.tick_params(axis='x', labelsize=12)
        ax.tick_params(axis='y', labelsize=14)
        ax.legend(fontsize=14)
        #ax.set_xlim(pd.to_datetime(start_date), )
        ax.grid(True, which='both', alpha=0.3)
        #fig.autofmt_xdate() 
        plt.title(station, fontsize = 18)
        #ax.set_xticklabels([])
        ax.xaxis.set_major_locator(mdates.HourLocator(byhour=range(0, 24, 12)))
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Hh\n%d/%m'))
        plt.savefig(output_surge_file)
        plt.show() ## comment out


        df['TWL_mean'] = df['Tide'] + df['LSTM_mean']
        df['TWL_min'] = df['Tide'] + df['LSTM_min']
        df['TWL_max'] = df['Tide'] + df['LSTM_max']


        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(df['DATETIME'], df['TWL_mean'], 'k', label='Ensemble mean')

        ax.fill_between(
            df['DATETIME'],
            df['TWL_min'],
            df['TWL_max'],
            color='blue',
            alpha=0.6,
            label='Ensemble range'
        )
        
        ax.set_ylabel("TWL (m)", fontsize=18)
        ax.tick_params(axis='x', labelsize=12)
        ax.tick_params(axis='y', labelsize=14)
        ax.legend(fontsize=14)
        ax.grid(True, which='both', alpha=0.3)
        ax.xaxis.set_major_locator(mdates.HourLocator(byhour=range(0, 24, 12)))
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Hh\n%d/%m'))
        #fig.autofmt_xdate() 
        #ax.set_xticklabels([])
        plt.title(station, fontsize = 18)
        plt.savefig(output_twl_file)
        plt.show() ##comment out

    return 0
    


def main(forecast_hour = '00z', date = '', forecast_steps = 72):

    """
        This function is called by the bash script. It takes in the forecast hour, and the date. If no date 
        is provided the current date is used. Currently data is downloaded from ECMWF Mars catalog.
    """

    if date == '':
        date = datetime.now().date()
    else:
        date = pd.to_datetime(date).date()
    
    date_str = str(date).replace('-', '_') #to save as a file

    output_dir = Path('Outputs/Data/') / date_str / forecast_hour
    input_file = output_dir / 'input_data.csv'
    model_dir ='../../Data/Model_Data/'
    forecast_dir = output_dir/ 'Forecast'
    forecast_dir.mkdir(parents=True, exist_ok=True)

    print('Downloading HRES data for: ', date)

    stations = ['Aranmore', 'Ballycotton', 'Ballyglass', 'Castletownbere', 'Dunmore', 'Carrigaholt',  'Dublin Port',  'Galway Port', 'Howth', 'Inishmore', 
                'Killybegs Port', 'Malin Head',  'Skerries Harbour', 'Sligo', 'Wexford', 'Fenit','Ferry Bridge Maigue', 'Foynes', 'Moneycashen', 'Port Bridge Swilly', 'Port Oriel', 'Ringaskiddy NMCI','Rossaveel Pier']

    ## Make sure all the data exists.
    if not input_file.exists():
        get_input_data(date, date_str, output_dir, input_file, forecast_steps=forecast_steps, forecast_hour = forecast_hour)

    #Forecast Surge and TWL
    df_input = pd.read_csv(input_file)
    
    for station in stations:
        print('Forecasting for ', station)
        model_folder = model_dir + station + '/Trained_Models/'

        df_station = df_input[df_input['Station'] == station]
        df_station = model_surge(model_folder, df_station, [ 'msl', 'u_s', 'v_s','Tide'], num_models = 20)
        df_station.to_csv(forecast_dir/f'{station}.csv')
    
    get_figures(stations, date_str, forecast_hour)



if __name__ == "__main__":
    main()