from pathlib import Path
import ecmwfapi
import numpy as np
import pandas as pd
import xarray as xr
from dotenv import load_dotenv
import os

PARAMETERS = {
    "u": "165.128",      # 10 m u-component of wind
    "v": "166.128",      # 10 m v-component of wind
    "msl": "151.128",    # mean sea-level pressure
}

DEFAULT_STEP = 24
DEFAULT_AREA = "55/349/51/355" #to cover Ireland


def download_weather_data(
    start_date,
    end_date,
    output_file,
    step=DEFAULT_STEP,
    area=DEFAULT_AREA,
):
    """
    Download HRES forecast data for a date range.

    Data are downloaded from MARS in one request for the supplied date range.
    The caller should keep the date range reasonably small (e.g. one month).

    Parameters
    ----------
    start_date : str
        Start date, e.g. "2025-01-01".
    end_date : str
        End date, e.g. "2025-02-01".
    output_file : Path
        Destination GRIB file.
    step : int
        Maximum forecast lead time in hours.
    area : str
        ECMWF area string.

    Notes
    -----
    MARS date ranges are inclusive, so when processing monthly chunks it is
    usually preferable to use the first day of the following month as the
    end date only if your API/query semantics have been tested accordingly.
    """

    output_file = Path(output_file)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    param_string = "/".join(PARAMETERS.values())
    step_range = f"0/to/{step}/by/1"

    date_range = f"{start_date}/to/{end_date}"

    print(
        f"Downloading HRES:\n"
        f"  dates : {date_range}\n"
        f"  output: {output_file}"
    )
    mars_url = 'https://api.ecmwf.int/v1'
    server = ecmwfapi.ECMWFService(
        service="mars",
        key=os.environ["MARS_KEY"],
        url=mars_url,
        email=os.environ["MARS_EMAIL"],
    )

    server.execute(
        {
            "class": "od",
            "stream": "oper",
            "expver": "1",
            "levtype": "sfc",
            "param": param_string,
            "step": step_range,
            "time": "00:00:00",
            "type": "fc",
            "target": "output",
            "grid": "0.25/0.25",
            "area": area,
            "date": date_range,
            "expect": "any",
        },
        str(output_file),
    )

    return output_file


def find_nearest_grid_points(ds, station_locs):
    """
    Find the nearest HRES grid point to each station.

    The station coordinates used for finding the nearest HRES grid point
    come from:
        station_locs["ECMWF Latitude"]
        station_locs["ECMWF Longitude"]

    Returns
    -------
    DataFrame
        Columns:
            Station
            ECMWF Latitude
            ECMWF Longitude
            ecmwf_latitude
            ecmwf_longitude
    """

    grid = (
        ds[["latitude", "longitude"]]
        .to_dataframe()
        .reset_index()[["latitude", "longitude"]]
        .drop_duplicates()
        .reset_index(drop=True)
    )

    # Determine longitude convention used by the GRIB.
    grib_uses_360 = grid["longitude"].max() > 180

    stations = station_locs[
        ["Station", "ECMWF Latitude", "ECMWF Longitude"]
    ].copy()

    # Keep the original ECMWF longitude so it is not overwritten.
    stations["lookup_longitude"] = stations["ECMWF Longitude"]

    # Convert longitude to the convention used by the GRIB.
    if grib_uses_360:
        stations["lookup_longitude"] %= 360
    else:
        stations.loc[
            stations["lookup_longitude"] > 180,
            "lookup_longitude",
        ] -= 360

    nearest_points = []

    for _, station in stations.iterrows():

        distance_squared = (
            (
                grid["latitude"]
                - station["ECMWF Latitude"]
            ) ** 2
            +
            (
                grid["longitude"]
                - station["lookup_longitude"]
            ) ** 2
        )

        nearest_idx = distance_squared.idxmin()
        nearest = grid.loc[nearest_idx]

        nearest_points.append(
            {
                "Station": station["Station"],
                "ECMWF Latitude": station["ECMWF Latitude"],
                "ECMWF Longitude": station["ECMWF Longitude"],
                "ecmwf_latitude": nearest["latitude"],
                "ecmwf_longitude": nearest["longitude"],
            }
        )

    return pd.DataFrame(nearest_points)



def convert_weather(grib_file, station_locs):
    """
    Convert one HRES GRIB file into a station-level DataFrame.

    Only the nearest HRES grid point to each station is retained.

    Returns
    -------
    DataFrame
        One row per station/time.
    """

    grib_file = Path(grib_file)

    print(f"Converting {grib_file} ...")

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

    try:
        # Find the nearest grid point for every station.
        nearest = find_nearest_grid_points(ds, station_locs)

        # Convert only the variables we need.
        required_variables = ["u10", "v10", "msl"]

        available_variables = [
            variable
            for variable in required_variables
            if variable in ds.data_vars
        ]

        missing = set(required_variables) - set(available_variables)

        if missing:
            raise ValueError(
                f"Missing variables in {grib_file}: {sorted(missing)}"
            )

        weather = (
            ds[available_variables]
            .to_dataframe()
            .reset_index()
        )

    finally:
        ds.close()

    weather = weather.rename(
        columns={
            "u10": "u",
            "v10": "v",
        }
    )

    weather["DATETIME"] = pd.to_datetime(weather["valid_time"])

    # Convert pressure from Pa to hPa.
    weather["msl"] = weather["msl"] / 100.0

    # Keep only the columns we actually need.
    weather = weather[
        [
            "DATETIME",
            "latitude",
            "longitude",
            "u",
            "v",
            "msl",
        ]
    ]

    # Match each station to its nearest ECMWF grid point.
    weather = weather.merge(
        nearest,
        left_on=["latitude", "longitude"],
        right_on=["ecmwf_latitude", "ecmwf_longitude"],
        how="inner",
    )

    print(weather.head(6))

    weather = weather[
        [
            "DATETIME",
            "Station",
            "ECMWF Latitude",
            "ECMWF Longitude",
            "ecmwf_latitude",
            "ecmwf_longitude",
            "u",
            "v",
            "msl",
        ]
    ]

    weather = weather.dropna(
        subset=[
            "u",
            "v",
            "msl",
        ]
    )

    weather = (
        weather
        .drop_duplicates(
            subset=["Station", "DATETIME"],
            keep="last",
        )
        .sort_values(["Station", "DATETIME"])
        .reset_index(drop=True)
    )

    return weather


def calculate_wind_stress(df):
    """
    Calculate wind stress components from 10 m wind.

    Adds:
        abs_w
        c
        u_s
        v_s
    """

    df = df.copy()

    df["abs_w"] = np.hypot(df["u"], df["v"])

    df["c"] = np.where(
        df["abs_w"] < 7.5,
        0.001,
        0.00061 + 0.000063 * df["abs_w"],
    )

    air_density = 1.225  # kg/m^3

    df["u_s"] = (
        air_density
        * df["c"]
        * df["abs_w"]
        * df["u"]
    )

    df["v_s"] = (
        air_density
        * df["c"]
        * df["abs_w"]
        * df["v"]
    )

    return df


def join_validation_data(
    weather,
    gauge_info,
    validation_root="Model_Data",
):
    """
    Add Station-specific data to the weather DataFrame.

    The validation files are expected at:

        Model_Data/<station>/validation_data.csv
        and
        Model_Data/<station>/training_data.csv
    """

    weather = weather.copy()
    weather["DATETIME"] = pd.to_datetime(weather["DATETIME"])

    output = []

    for station in gauge_info["Station"].unique():

        station_weather = weather[
            weather["Station"] == station
        ].copy()

        if station_weather.empty:
            print(f"WARNING: no weather data for {station}")
            continue

        station_dir = Path(validation_root) / station
        training_file = station_dir / "training_data.csv"
        validation_file = station_dir / "validation_data.csv"
        data_frames = []

        for file, dataset_type in [
            (training_file, "training"),
            (validation_file, "validation"),
        ]:

            if not file.exists():
                print(
                    f"WARNING: {dataset_type} file missing for {station}: "
                    f"{file}"
                )
                continue

            df = pd.read_csv(file)

            if "DATETIME" not in df.columns:
                print(
                    f"WARNING: DATETIME missing from {file}"
                )
                continue

            df["DATETIME"] = pd.to_datetime(df["DATETIME"])

            df["data_split"] = dataset_type

            data_frames.append(df)

        if not data_frames:
            continue

        station_data = pd.concat(
            data_frames,
            ignore_index=True,
        )
        station_data["DATETIME"] = pd.to_datetime(
            station_data["DATETIME"]
        )

        station_data = station_data[
            [
                "DATETIME",
                "TWL_OD",
                "TIDE",
                "SURGE",
                "Water_Level_trend",
            ]
        ]

        station_weather = station_weather.merge(
            station_data,
            on="DATETIME",
            how="inner",
        )

        output.append(station_weather)

    if not output:
        return pd.DataFrame()

    return (
        pd.concat(output, ignore_index=True)
        .sort_values(["Station", "DATETIME"])
        .drop_duplicates(
            subset=["Station", "DATETIME"],
            keep="last",
        )
        .reset_index(drop=True)
    )

def save_station_files(df, output_dir):
    """
    Save one CSV per station.

    Files are written as:

        output_dir/<station>.csv
    """

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for station, station_df in df.groupby("Station"):

        station_df = (
            station_df
            .sort_values("DATETIME")
            .drop_duplicates(
                subset="DATETIME",
                keep="last",
            )
            .reset_index(drop=True)
        )

        output_file = output_dir / f"{station}.csv"

        station_df.to_csv(
            output_file,
            index=False,
        )

        print(
            f"Saved {station}: "
            f"{len(station_df):,} rows -> {output_file}"
        )




def process_weather_chunk(
    grib_file,
    station_locs,
):
    """
    Downloaded GRIB -> station-level weather DataFrame.
    """

    weather = convert_weather(
        grib_file,
        station_locs,
    )

    weather = calculate_wind_stress(weather)

    return weather[
        [
            "DATETIME",
            "Station",
            "u_s",
            "v_s",
            "msl",
        ]
    ]



def process_year(
    year,
    station_locs,
    gauge_info,
    output_root="HRES",
    step=DEFAULT_STEP,
    area=DEFAULT_AREA,
    validation_root="Model_Data",
):
    """
    Download and process a complete year of HRES data.

    The year is processed month-by-month to avoid creating a very large
    GRIB file or DataFrame.

    Final output:

        HRES/<year>/<station>.csv
    """

    year = int(year)

    year_dir = Path(output_root) / str(year)
    grib_dir = year_dir / "grib"
    station_dir = year_dir / "stations"

    grib_dir.mkdir(parents=True, exist_ok=True)
    station_dir.mkdir(parents=True, exist_ok=True)

    monthly_weather = []

    dates = pd.date_range(
        start=f"{year}-01-01",
        end=f"{year+1}-01-01",
        freq="MS",
    )

    for start, end in zip(dates[:-1], dates[1:]):

        # Last day actually requested.
        end_date = end - pd.Timedelta(days=1)

        chunk_name = (
            f"hres_{start:%Y-%m}.grib"
        )

        grib_file = grib_dir / chunk_name

        print()
        print("=" * 70)
        print(
            f"Processing {start:%Y-%m}"
        )
        print("=" * 70)

        # ---------------------------------------------------------------
        # Download
        # ---------------------------------------------------------------

        if not grib_file.exists():

            download_weather_data(
                start_date=start.strftime("%Y-%m-%d"),
                end_date=end_date.strftime("%Y-%m-%d"),
                output_file=grib_file,
                step=step,
                area=area,
            )

        else:
            print(
                f"Already downloaded: {grib_file}"
            )

        # ---------------------------------------------------------------
        # Convert to csv
        # ---------------------------------------------------------------

        weather = process_weather_chunk(
            grib_file,
            station_locs,
        )

        monthly_weather.append(weather)

        print(
            f"Converted {len(weather):,} station/time records"
        )

    print()
    print("Joining monthly weather data...")

    weather = (
        pd.concat(
            monthly_weather,
            ignore_index=True,
        )
        .sort_values(["Station", "DATETIME"])
        .drop_duplicates(
            subset=["Station", "DATETIME"],
            keep="last",
        )
        .reset_index(drop=True)
    )


    print("Joining validation data...")

    weather = join_validation_data(
        weather,
        gauge_info,
        validation_root=validation_root,
    )
    print("Saving station files...")

    save_station_files(
        weather,
        station_dir,
    )

    print()
    print(f"Finished {year}")

    return weather

def main():
    load_dotenv()  # containing key and email for mars-api

    year = 2025

    gauge_info = pd.read_csv(
        "paper_results/gauge_info.csv"
    )

    process_year(
        year=year,
        station_locs=gauge_info,
        gauge_info=gauge_info,
        output_root="Data/HRES/",
        step=24,
        area="55/349/51/355",
        validation_root="Data/Model_Data/",
    )


if __name__ == "__main__":
    main()