from datetime import datetime, timedelta
import numpy as np
import pandas as pd
import os
import hatyan
from scipy import signal
from pathlib import Path

"""
    Load and pre-process all observational hydro and meteo data, and TSFF hydro.
    Does some cleaning of the data but assumes errors have been flagged by QC.

"""

def process_tide_gauge(folder):

    """ This takes the QC file of the gauge and calculates the tide and surge residual, and saves
        the water level trend to file. """
    
    file_path = folder + 'cleaned_tide_gauge.csv'

    df = pd.read_csv(file_path)
    df['DATETIME'] = pd.to_datetime(df['DATETIME'])
    df = df.drop_duplicates() 
    combined_df = df.copy()
    ## Do tide surge analysis 
    combined_df=combined_df.set_index('DATETIME')
    combined_df = combined_df.sort_index()
    combined_df = combined_df.dropna()
    
    combined_df['water_level'] = combined_df['Water_Level_OD_Malin'].astype(float) # forcing type here
    

    ## detrending to remove sea level rise from 'water_level'
    trend = combined_df['water_level'].rolling('365D').mean()
    combined_df['water_level'] = combined_df['water_level'] - trend
    combined_df['Water_Level_trend'] = trend

    results, comp = perform_tide_surge_analysis(combined_df, combined_df['Station_Name'].unique()[0])
    surge_name = 'SURGE'
    print(results.head(5))
    print(results.columns)
    combined_df[surge_name] = results['surge_filtered']
    combined_df['TIDE'] = results['tide']
    station_report = generate_station_report(results, comp, combined_df['Station_Name'].unique()[0])
    combined_df = combined_df.drop(columns = ['water_level'])
    combined_df = combined_df.reset_index()

    return combined_df, station_report




def process_tssf(folder, gauge):
    """ keeping the smallest T+ (shortest forecast)"""

    
    filenames = os.listdir(folder)  
    gauge_files = [f for f in filenames if gauge in f]

    surge_data = []
    tide_data = []
    twl_data = []

    for file in gauge_files:
        
        file_path = os.path.join(folder, file)
        df = pd.read_csv(file_path)
        df['date_time'] = pd.to_datetime(df['date_time'])
        df['T+'] = (df['date_time'] - df['date_time'].iloc[0]).dt.total_seconds() / 3600
        if 'surge' in file:
            surge_data.append(df)
        elif 'tide' in file:
            tide_data.append(df)
        elif 'twl' in file:
            twl_data.append(df)
        else:
            print('Unknown file...', file)
    if gauge_files == []:
        print('No files with ', gauge, ' as a name.')
        return 0
            
    surge_df = pd.concat(surge_data, ignore_index=True)
    tide_df = pd.concat(tide_data, ignore_index=True)
    twl_df = pd.concat(twl_data, ignore_index=True)
    surge_df = surge_df.rename(columns = {'Surge Residual (m)': 'SURGE', 'date_time': 'DATETIME'})[['DATETIME', 'T+', 'SURGE']]
    tide_df = tide_df.rename(columns ={'Tidal Level (m Malin)': 'TIDE', 'date_time': 'DATETIME'})[['DATETIME','T+',  'TIDE']]
    twl_df = twl_df.rename(columns ={'Total Water Level (m Malin)': 'TWL', 'date_time': 'DATETIME'})[['DATETIME','T+',  'TWL']]

    df_all = surge_df.merge(
        tide_df,
        on=['DATETIME', 'T+'],
        how='outer'
    )
    df_all = df_all.merge(
        twl_df,
        on=['DATETIME', 'T+'],
        how='outer'
    )
    df_all = df_all.sort_values('DATETIME').reset_index(drop=True)

    df_unique = (
            df_all.sort_values(["DATETIME", "T+"])  # smallest T+ first
            .drop_duplicates(subset=["DATETIME"], keep="first")
    )

    print(len(df_unique['DATETIME'].unique()), len(df_unique))
    print(min(df_unique['DATETIME']), max(df_unique['DATETIME']))

    return df_unique


"""
    The Following functions are from tide_extraction.py by Ryan McGeady

"""

def perform_tide_surge_analysis(df_clean, station_name):
    """ 
        Perform tide and surge separation analysis. 
    
        Based on code from tide_extraction.py by Ryan McGeady 
    
    """
    try:
        # Prepare the time series for hatyan
        df_clean.index = (
            df_clean.index
            .tz_convert("UTC")          # ensure UTC (no-op if already)
            .tz_localize(None)          # temporarily remove tz
            .astype("datetime64[ns]")   # enforce ns resolution
        )

        ts_df = pd.DataFrame(
            {"values": df_clean["water_level"].to_numpy(dtype=float)},
            index=df_clean.index
        )
        ts_obs = pd.Series(
            df_clean["water_level"].to_numpy(dtype=float),
            index=df_clean.index,
            name="values"
        )

        # Convert to DataFrame format expected by hatyan
        ts_df = ts_obs.to_frame(name="values")
        ts_df = ts_df[~ts_df.index.duplicated(keep="first")]
        
        print(ts_df.dtypes)
        print(ts_df.index.dtype)      
        # Comprehensive constituent selection for Irish waters
        constituents = [
            # Principal semi-diurnal
            'M2', 'S2', 'N2', 'K2', 'NU2', 'MU2', 'L2', 'T2',
            # Principal diurnal
            'K1', 'O1', 'P1', 'Q1', 'J1', 'OO1',
            # Long period
            'MM', 'MF', 'MSF', 'SA', 'SSA',
            # Higher harmonics
            'M4', 'MS4', 'M6'
        ]
        
        # Perform harmonic analysis with error handling
        print(f"Performing harmonic analysis for {station_name}...")
        
        try:
            # First attempt with full constituent list
            comp = hatyan.analysis(ts=ts_obs, 
                                  const_list=constituents,
                                  source='schureman')
            print(f"Successfully analyzed {len(comp)} constituents")
            
        except Exception as e:
            print(f"Full analysis failed: {e}")
            # Fallback to essential constituents only
            print("Trying with essential constituents...")
            essential_constituents = ['M2', 'S2', 'N2', 'K1', 'O1', 'P1', 'Q1', 'K2']
            #['M2', 'S2', 'N2', 'K1', 'O1', 'P1', 'Q1', 'K2']
            comp = hatyan.analysis(ts=ts_df, 
                                  const_list=essential_constituents,
                                  source='schureman')
            print(f"Successfully analyzed {len(comp)} essential constituents")
        
        # Generate tidal prediction
        ts_prediction = hatyan.prediction(comp=comp, times=ts_df.index)
        
        mean_offset = ts_df['values'].mean() - ts_prediction['values'].mean()
        ts_prediction['values'] += mean_offset
        
        # Calculate surge component
        surge = ts_df['values'] - ts_prediction['values']
        
        # Apply low-pass filter to remove high-frequency noise from surge
        fs = 1/(15*60)  # Sampling frequency in Hz (15-minute data)
        cutoff_hours = 6
        cutoff_freq = 1/(cutoff_hours*3600)  # Convert to Hz
        nyquist = fs/2
        normalized_cutoff = cutoff_freq/nyquist
        
        if normalized_cutoff < 1:
            b, a = signal.butter(3, normalized_cutoff, btype='low')
            surge_filtered = signal.filtfilt(b, a, surge.values)
        else:
            surge_filtered = surge.values
            print("Warning: Cutoff frequency too high for filtering")
        
        # Create results DataFrame
        results = pd.DataFrame({
            'observed': ts_df['values'],
            'tide': ts_prediction['values'],
            'surge_raw': surge,
            'surge_filtered': surge_filtered,
            'total_predicted': ts_prediction['values'] + surge_filtered
        })
        
        # Remove any NaN values
        results = results.dropna()
        
        return results, comp
        
    except Exception as e:
        print(f"ERROR in analysis for {station_name}: {str(e)}")
        return None, None
    

def generate_station_report(results, comp, station_name):
    """
    
        Generate comprehensive report for a single station
        
        Based on code from tide_extraction.py by Ryan McGeady 
    
    """

    major_constituents = ['M2', 'S2', 'N2', 'K1', 'O1', 'P1', 'Q1', 'K2']
    for const in major_constituents:
        if const in comp.index:
            amp = comp.loc[const, 'A']
            phase = comp.loc[const, 'phi_deg']
            print(f"  {const}: Amplitude = {amp:.3f} m, Phase = {phase:.1f}°")
    

    # Quality metrics
    correlation_tide = np.corrcoef(results['observed'], results['tide'])[0, 1]
    correlation_total = np.corrcoef(results['observed'], results['total_predicted'])[0, 1]
    rmse_tide = np.sqrt(np.mean((results['observed'] - results['tide'])**2))
    rmse_total = np.sqrt(np.mean((results['observed'] - results['total_predicted'])**2))

    return pd.DataFrame([{
        'station': station_name,
        'data_points': len(results),
        'days': (results.index[-1] - results.index[0]).days,
        'tide_range': results['tide'].max() - results['tide'].min(),
        'surge_mean': results['surge_filtered'].mean(),
        'surge_std': results['surge_filtered'].std(),
        'surge_min': results['surge_filtered'].min(),
        'surge_max': results['surge_filtered'].max(),
        'surge_95th': results['surge_filtered'].quantile(0.95),
        'surge_5th': results['surge_filtered'].quantile(0.05),
        'corr_tide': correlation_tide,
        'corr_total': correlation_total,
        'rmse_tide': rmse_tide,
        'rmse_total': rmse_total,
        'var_explained_tide': correlation_tide**2*100,
        'var_explained_total': correlation_total**2*100
    }])


def process_ecmwf(folder):

    """   
        Unpack the csv files and join.
    
    """
    files = os.listdir(folder)
    combined_df = None

    for file in files: 
        print('Processing.... ', file)

        df = pd.read_csv(folder + file)
        df['DATETIME'] = pd.to_datetime(df['valid_time'])
        df = df.drop(columns=["valid_time", "latitude", "longitude"])
        df = df.set_index("DATETIME")

        if combined_df is None:
            combined_df = df
        else:
            # join based on the datetime index
            combined_df = combined_df.join(df, how="outer")

    combined_df = combined_df.rename(columns={'v10': 'v', 'u10': 'u'})
    ## get msl in correct unit
    combined_df['msl'] = combined_df['msl']/100
    combined_df = combined_df[['u','v','msl']]

    ## Add datetime back as a column
    combined_df = combined_df.reset_index()

    # Only up to 2026
    combined_df = combined_df[combined_df['DATETIME'].dt.year <2026]

    return combined_df



def fit_empirical_qm(hres_train, era5_train):
    h = pd.to_numeric(hres_train, errors="coerce").to_numpy()
    e = pd.to_numeric(era5_train, errors="coerce").to_numpy()

    mask = np.isfinite(h) & np.isfinite(e)
    h = h[mask]
    e = e[mask]

    if len(h) < 10:
        raise ValueError("Not enough valid paired HRES/ERA5 values for quantile mapping.")

    h_sorted = np.sort(h)
    e_sorted = np.sort(e)

    # Empirical plotting positions
    p = (np.arange(1, len(h_sorted) + 1) - 0.5) / len(h_sorted)

    return h_sorted, e_sorted, p


def apply_empirical_qm(x, h_sorted, e_sorted, p):
    x = pd.to_numeric(x, errors="coerce").to_numpy()

    # Step 1: find cumulative probability of HRES value
    px = np.interp(
        x,
        h_sorted,
        p,
        left=p[0],
        right=p[-1]
    )

    # Step 2: map that probability to ERA5 value
    y = np.interp(
        px,
        p,
        e_sorted,
        left=e_sorted[0],
        right=e_sorted[-1]
    )

    return y

def empirical_quantile_mapping_2025(
    stations,
    vars_to_correct,
    hres_dir=".",
    model_data_dir="Model_Data",
    excluded_stations=None,
):
    """
    Fit empirical quantile mapping using 2024 HRES/ERA5 data and apply
    the fitted mapping to 2025 HRES data.

    Corrected 2025 data are saved to:

        Model_Data/<station>/hres_validation_data.csv

    """

    if excluded_stations is None:
        excluded_stations = []

    excluded_stations = set(excluded_stations)

    qm_params = {}

    for station in stations:

        if station in excluded_stations:
            print(f"Skipping {station}")
            continue

        print(f"\nEQM: {station}")

        # ------------------------------------------------------------------
        # File paths
        # ------------------------------------------------------------------

        era5_file = (model_data_dir +station+"/training_data.csv")

        hres_2024_file = (hres_dir + f"2024/stations/{station}.csv")
        hres_2025_file = (hres_dir + f"2025/stations/{station}.csv")
        output_file = (model_data_dir+station+"/hres_validation_data.csv")


        # ------------------------------------------------------------------
        # Load data
        # ------------------------------------------------------------------

        era5 = pd.read_csv(era5_file)
        hres_2024 = pd.read_csv(hres_2024_file)
        hres_2025 = pd.read_csv(hres_2025_file)

        for df in [era5, hres_2024, hres_2025]:
            df["DATETIME"] = pd.to_datetime(df["DATETIME"])

        # ------------------------------------------------------------------
        # Convert variables to numeric
        # ------------------------------------------------------------------

        for var in vars_to_correct:

            for name, df in [
                ("ERA5", era5),
                ("HRES 2024", hres_2024),
                ("HRES 2025", hres_2025),
            ]:
                if var not in df.columns:
                    raise ValueError(
                        f"{var} not found in {name} data for {station}"
                    )

                df[var] = pd.to_numeric(
                    df[var],
                    errors="coerce",
                )

        # ------------------------------------------------------------------
        # Fit EQM using 2024 HRES against ERA5
        # ------------------------------------------------------------------

        merged = pd.merge(
            hres_2024[["DATETIME", *vars_to_correct]],
            era5[["DATETIME", *vars_to_correct]],
            on="DATETIME",
            how="inner",
            suffixes=("_hres", "_era5"),
        )

        if merged.empty:
            print(
                f"WARNING: no overlapping 2024 HRES/ERA5 data "
                f"for {station}"
            )
            continue

        qm_params[station] = {}

        for var in vars_to_correct:

            training = merged[
                [
                    f"{var}_hres",
                    f"{var}_era5",
                ]
            ].dropna()

            if training.empty:
                print(
                    f"WARNING: no valid training data "
                    f"for {station} / {var}"
                )
                continue

            (
                hres_sorted,
                era5_sorted,
                probabilities,
            ) = fit_empirical_qm(
                training[f"{var}_hres"],
                training[f"{var}_era5"],
            )

            qm_params[station][var] = {
                "hres_sorted": hres_sorted,
                "era5_sorted": era5_sorted,
                "probabilities": probabilities,
            }

            # ------------------------------------------------------------------
            # Apply 2024-fitted EQM to 2025 HRES
            # ------------------------------------------------------------------

            hres_2025[var] = apply_empirical_qm(
                hres_2025[var],
                hres_sorted,
                era5_sorted,
                probabilities,
            )

            print(
                f"  {var}: "
                f"{len(training):,} training points"
            )

        # ------------------------------------------------------------------
        # Save corrected 2025 HRES
        # ------------------------------------------------------------------

        output_file.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        hres_2025.to_csv(
            output_file,
            index=False,
        )

        print(
            f"  Saved: {output_file}"
        )

    return qm_params


def main(stations = None):


    if stations == None:
        ## if subset of stations not provided, run for all stations
        stations = ['Aranmore', 'Ballycotton', 'Ballyglass', 'Castletownbere', 'Dunmore', 'Carrigaholt',  'Dublin Port',  'Galway Port', 'Howth', 'Inishmore', 
                'Killybegs Port', 'Malin Head',  'Skerries Harbour', 'Sligo', 'Wexford', 'Fenit','Ferry Bridge Maigue', 'Foynes', 'Moneycashen', 'Port Bridge Swilly', 'Port Oriel', 'Ringaskiddy NMCI','Rossaveel Pier']


    folder_base = '../Data/'
    folder_created = '../Data/PreProcessed/' 

    print('Preprocessing gauge data for: ', stations)
    
    for gauge in stations: 

        output_dir = folder_created+gauge
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        models_dir = '../Data/Model_Data/' + gauge
        models_dir = Path(models_dir)
        models_dir.mkdir(parents=True, exist_ok=True)

        df_tide_gauge, station_report = process_tide_gauge(folder_base + 'tide_gauge/'+gauge+'/')
        df_tide_gauge.to_csv(output_dir / 'tide_gauge.csv', index = False)
        station_report.to_csv(output_dir / 'tide_gauge_report.csv', index = False)

        df_ecmwf = process_ecmwf(folder_base + 'ecmwf/' + gauge + '/')
        df_ecmwf.to_csv(output_dir / 'ecmwf.csv', index = False)

    excluded_stations = [
        "Ballyglass",
        "Castletownbere",
    ]

    qm_params = empirical_quantile_mapping_2025(
        stations=stations,
        vars_to_correct=['u_s', 'v_s', 'msl'],
        hres_dir="../Data/HRES/",
        model_data_dir="../Data/Model_Data/",
        excluded_stations=excluded_stations,
    )

if __name__ == '__main__':
    main()

