# -*- coding: utf-8 -*-
"""
This module contains functions that read (and plot) various
meteorological instrument data. This includes:
- MET Automatic Weather Station (AWS)
- UNIS AWS
- Radiation
- Sonic Anemometer
- TinyTags
- IWIN
- AROME Arctic

The functions were developed at the University Centre in Svalbard. They were
optimized for the file formats typically used in the UNIS courses.
"""

from . import universal_func as uf
import pandas as pd
import dask.dataframe as ddf
import xarray as xr
import glob
import geopandas as gpd
import numpy as np
import numpy.typing as npt
import cmocean as cmo
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.contour import QuadContourSet
from matplotlib.colorbar import Colorbar
from matplotlib.figure import Figure
from shapely.geometry import LineString
from shapely.errors import GEOSException
import rioxarray as rxr
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
from cartopy.mpl.geoaxes import GeoAxes
import os
import datetime
from collections import defaultdict
from typing import Literal, Any, Self, get_args
import warnings
import logging
import re

############################################################################
# READING FUNCTIONS
############################################################################


def read_MET_AWS(filepath: str) -> pd.DataFrame | dict[str, pd.DataFrame]:
    """Reads data from a CSV file downloaded from seklima.met.no.
    Can handle CSV with multiple stations and one.
    Standard variable names and convention are not used!

    Args:
        filepath (str): String with path to CSV file.

    Returns:
        pandas.DataFrame or dict[str, pandas dataframe]:
            - A pandas.DataFrame with time as index and the individual variables as columns.
            - A dictionary of pandas.DataFrames (keyed by station name) if multiple stations are present.
    """

    if not isinstance(filepath, str):
        raise TypeError(
            f"Expected filepath as a string, but got {type(filepath).__name__}."
        )
    if not os.path.isfile(filepath):
        raise FileNotFoundError(f"File not found: {filepath}")

    if filepath.endswith(".csv"):
        df: pd.DataFrame = pd.read_csv(
            filepath,
            skipfooter=1,
            sep=";",
            engine="python",
            na_values="-",
            dayfirst=True,
            parse_dates=[2],
            header=0,
            decimal=",",
        )
    else:
        raise ValueError(f"Invalid file format: {filepath}. Expected a .csv file.")

    try:
        df["TIMESTAMP"] = df["Tid(norsk normaltid)"] - pd.Timedelta("1h")
        df["TIMESTAMP"] = pd.to_datetime(df["TIMESTAMP"])
        df.set_index("TIMESTAMP", inplace=True)
        df.drop(["Tid(norsk normaltid)"], axis=1, inplace=True)
        stations: list = df["Navn"].unique().tolist()
        if len(stations) == 1:
            df.drop(["Navn", "Stasjon"], axis=1, inplace=True)
            return df
        else:
            dic: dict = {
                i: df[df["Navn"] == i].drop(["Navn", "Stasjon"], axis=1)
                for i in stations
            }
            return dic

    except KeyError:
        df["TIMESTAMP"] = df["Time(norwegian mean time)"] - pd.Timedelta("1h")
        df["TIMESTAMP"] = pd.to_datetime(df["TIMESTAMP"])
        df.set_index("TIMESTAMP", inplace=True)
        df.drop(["Time(norwegian mean time)"], axis=1, inplace=True)
        stations: list = df["Name"].unique().tolist()
        if len(stations) == 1:
            df.drop(["Name", "Station"], axis=1, inplace=True)
            return df
        else:
            dic: dict[str, pd.DataFrame] = {
                i: df[df["Name"] == i].drop(["Name", "Station"], axis=1)
                for i in stations
            }
            return dic


def read_Campbell_TOA5(filepath: str) -> pd.DataFrame:
    """Reads data from one or several TOA5 files from Campbell data loggers.
    Standard variable names and convention are used (e.g. T_1 [degC], T_air [degC]).

    Args:
        filepath (str): Path to one or more '.dat' file(s).
            - For multiple files, use UNIX-style wildcards ('*' for any character(s), '?' for single character, etc.)

    Returns:
        pandas.DataFrame: a Dataframe with time as index and the individual variables as columns.
    """

    if not isinstance(filepath, str):
        raise TypeError(
            f"Expected filepath as a string, but got {type(filepath).__name__}."
        )
    files: list[str] = glob.glob(filepath)
    if len(files) == 0:
        raise ValueError(f"No such file(s):'{filepath}'")
    for i in files:
        if not os.path.isfile(i):
            raise ValueError(
                f"Invalid input: '{i}'. Expected valid file(s) name with .dat extension."
            )
        if not i.endswith(".dat"):
            raise ValueError(f"Invalid file format: '{i}'. Expected a .dat file.")

    one_file: str = sorted(glob.glob(filepath))[0]

    with open(one_file, "r") as f:
        _, col_names, units = [
            line.strip().replace('"', "").split(",") for line in f.readlines()[:3]
        ]
    col_names: list[str] = [i.replace(" ", "_") for i in col_names]
    units: list[str]

    table_headers: list[str] = []
    for cn, u in zip(col_names, units):
        if cn not in ["TIMESTAMP", "RECORD"]:
            table_headers.append(" ".join((cn.strip(), "[" + u.strip() + "]")))
        else:
            table_headers.append(cn)

    dtypes_dict = defaultdict(lambda: np.float32)
    dtypes_dict["RECORD"] = np.int32

    d_df: ddf.DataFrame = ddf.read_csv(
        filepath,
        skiprows=4,
        dtype=dtypes_dict,
        parse_dates=["TIMESTAMP"],
        names=table_headers,
        date_format="%Y-%m-%d %H:%M:%S",
        na_values=["NAN"],
    )
    df: pd.DataFrame = d_df.compute()
    df["TIMESTAMP"] = pd.to_datetime(df["TIMESTAMP"])
    df.set_index("TIMESTAMP", inplace=True)
    df.sort_index(inplace=True)

    df = uf.std_names(df)

    return df


def read_EddyPro_full_output(filepath: str) -> pd.DataFrame:
    """Reads data from one or several EddyPro full output file(s).
    Standard variable names and convention are used (e.g. T_1 [degC], T_air [degC]).

    Args:
        filepath (str): Path to one or more '.csv' file(s).
            - For multiple files, use UNIX-style wildcards ('*' for any character(s), '?' for single character, etc.).

    Returns:
        pandas.DataFrame: Dataframe with time as index and the individual variables as columns.
    """

    if not isinstance(filepath, str):
        raise TypeError(
            f"Expected filepath as a string, but got {type(filepath).__name__}."
        )
    files: list[str] = glob.glob(filepath)
    if len(files) == 0:
        raise ValueError(f"No such file(s):'{filepath}'")
    for i in files:
        if not os.path.isfile(i):
            raise ValueError(
                f"Invalid input: '{i}'. Expected valid file(s) name with .csv extension."
            )
        if not i.endswith(".csv"):
            raise ValueError(f"Invalid file format: '{i}'. Expected a .csv file.")

    one_file: str = sorted(files)[0]
    with open(one_file, "r") as f:
        f.readline()
        col_names: list[str] = f.readline().split(",")
        units: list[str] = f.readline().split(",")

    dtypes_dict = defaultdict(lambda: np.float32)
    dtypes_dict["file_records"] = np.int64
    dtypes_dict["used_records"] = np.int64
    dtypes_dict["daytime"] = "int8"
    dtypes_dict["filename"] = "str"
    dtypes_dict["date"] = "str"
    dtypes_dict["time"] = "str"

    table_headers: list[str] = []
    for cn, u in zip(col_names, units):
        if cn not in [
            "filename",
            "date",
            "time",
            "DOY",
            "daytime",
            "file_records",
            "used_records",
        ]:
            table_headers.append(" ".join((cn.strip(), u.strip())))
        else:
            table_headers.append(cn)
        if cn == "date":
            date_format: str = (
                u.strip("[]")
                .replace("yyyy", "%Y")
                .replace("mm", "%m")
                .replace("dd", "%d")
            )
        elif cn == "time":
            time_format: str = u.strip("[]").replace("HH", "%H").replace("MM", "%M")

    d_df: ddf.DataFrame = ddf.read_csv(
        filepath, skiprows=4, dtype=dtypes_dict, names=table_headers, na_values=["NAN"]
    )
    df: pd.DataFrame = d_df.compute()
    df.drop("filename", axis=1, inplace=True)
    df["TIMESTAMP"] = pd.to_datetime(
        df.pop("date") + " " + df.pop("time"), format=date_format + " " + time_format
    )
    df.set_index("TIMESTAMP", inplace=True)
    df.sort_index(inplace=True)

    df = uf.std_names(df)

    return df


def read_Tinytag(
    filepath: str, sensor: None | Literal["TT", "TH", "CEB"] = None
) -> pd.DataFrame:
    """Reads data from one or several data files from the Tinytag output files.
    Standard variable names and convention are used (e.g. T_1 [degC], T_air [degC]).

    Args:
        filepath (str): Path to one or more '.txt' file(s).
            - For multiple files, use UNIX-style wildcards ('*' for any character(s), '?' for single character, etc.).
        sensor (None or {'TT', 'TH', 'CEB'}, optional): Defaults to None.
            - None for automatic detection. Works if 'TT', 'TH' or 'CEB' is split by '_' (e.g. '*_TH_*.txt', 'TH_*.txt', *_TH.txt).
            - 'TT' for devices with 2 temparature measurments.
            - 'TH' for devices with temperature and relativ humidity.
            - 'CEB' for devices with just temperature.

    Returns:
        pandas.DataFrame: DataFrame with time as index and the individual variables as columns.
    """

    if not isinstance(filepath, str):
        raise TypeError(
            f"Expected filepath as a string, but got {type(filepath).__name__}."
        )
    files: list[str] = glob.glob(filepath)
    if len(files) == 0:
        raise ValueError(f"No such file(s):'{filepath}'")
    for i in files:
        if not os.path.isfile(i):
            raise ValueError(
                f"Invalid input: '{i}'. Expected valid file(s) name with .txt extension."
            )
        if not i.endswith(".txt"):
            raise ValueError(f"Invalid file format: '{i}'. Expected a .txt file.")

    if sensor == None:
        filename: str = (
            os.path.basename(files[0])
            .translate(str.maketrans("", "", "1234567890"))
            .split("_")
        )
        filename[-1] = filename[-1][:-4]  # get rid of the .txt
        if "TT" in filename:
            sensor = "TT"
        if "TH" in filename:
            if sensor:
                raise ValueError(
                    f"More then one sensor detected. Specify with arg: sensor="
                )
            sensor = "TH"
        if "CEB" in filename:
            if sensor:
                raise ValueError(
                    f"More then one sensor detected. Specify with arg: sensor="
                )
            sensor = "CEB"
        if not sensor:
            raise ValueError(f"Sensor information not found. Specify with arg: sensor=")
        logging.info(f"Using {sensor} as a sensor.")

    if sensor == "TT":
        d_df: ddf.DataFrame = ddf.read_csv(
            filepath,
            delimiter="\t",
            skiprows=5,
            parse_dates=[1],
            date_format="%d %b %Y %H:%M:%S",
            names=["RECORD", "TIMESTAMP", "T_black", "T_white"],
            encoding="ISO-8859-1",
        )
    elif sensor == "TH":
        d_df: ddf.DataFrame = ddf.read_csv(
            filepath,
            delimiter="\t",
            skiprows=5,
            parse_dates=[1],
            date_format="%d %b %Y %H:%M:%S",
            names=["RECORD", "TIMESTAMP", "T", "RH"],
            encoding="ISO-8859-1",
        )
    elif sensor == "CEB":
        d_df: ddf.DataFrame = ddf.read_csv(
            filepath,
            delimiter="\t",
            skiprows=5,
            parse_dates=[1],
            date_format="%d %b %Y %H:%M:%S",
            names=["RECORD", "TIMESTAMP", "T"],
            encoding="ISO-8859-1",
        )
    else:
        raise ValueError(
            'Sensortype of Tinytag not known. Should be one of "TT", "TH" or "CEB".'
        )

    df: pd.DataFrame = d_df.compute()
    df["TIMESTAMP"] = pd.to_datetime(df["TIMESTAMP"])
    df.set_index("TIMESTAMP", inplace=True)

    df = uf.std_names(df, add_units=True)

    for key in df.columns:
        if not key == "RECORD":
            data: list[float] = [float(i.split(" ")[0]) for i in df[key]]
            df[key] = data

    return df


def read_HOBO(filepath: str, get_sn: bool = False) -> pd.DataFrame:
    """Reads data from one or several data files from the HOBO output files.
    Standard variable names and convention are used (e.g. T_1 [degC], T_air [degC]).

    Args:
        filepath (str): Path to one or more '.txt' file(s).
            - For multiple files, use UNIX-style wildcards ('*' for any character(s), '?' for single character, etc.).
        get_sn (bool): Whether the serial number should be included in the column name. Defaults to False.

    Returns:
        pandas.DataFrame: Dataframe with time as index and the individual variables as columns.
    """

    if not isinstance(filepath, str):
        raise TypeError(
            f"Expected filepath as a string, but got {type(filepath).__name__}."
        )
    files: list[str] = glob.glob(filepath)
    if len(files) == 0:
        raise ValueError(f"No such file(s):'{filepath}'")
    for i in files:
        if not os.path.isfile(i):
            raise ValueError(
                f"Invalid input: '{i}'. Expected valid file(s) name with .txt extension."
            )
        if not i.endswith(".txt"):
            raise ValueError(f"Invalid file format: '{i}'. Expected a .txt file.")

    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="Could not infer format, so each element will be parsed individually, falling back to `dateutil`",
            category=UserWarning,
            module="dask.dataframe.io.csv",
        )
        warnings.filterwarnings(
            "ignore",
            message="Parsing dates in %Y.%m.%d %H:%M:%S format when dayfirst=True was specified. Pass `dayfirst=False` or specify a format to silence this warning.",
            category=UserWarning,
            module="dask.dataframe.io.csv",
        )
        d_df: ddf.DataFrame = ddf.read_csv(
            filepath,
            delimiter=";",
            skiprows=1,
            parse_dates=["Date Time, GMT+00:00"],
            dayfirst=True,
            encoding="ISO-8859-1",
            thousands=",",
        )
        df: pd.DataFrame = d_df.compute()

    df.rename({"Date Time, GMT+00:00": "TIMESTAMP"}, axis=1, inplace=True)
    df["TIMESTAMP"] = pd.to_datetime(df["TIMESTAMP"])
    df.set_index("TIMESTAMP", inplace=True)
    df.sort_index(inplace=True)

    df = uf.std_names(df, bonus=get_sn)

    return df


def read_Raingauge(filepath: str, get_sn: bool = False) -> pd.DataFrame:
    """Reads data from one or several data files from the raingauge output files.

    Args:
        filepath (str): Path to one or more '.txt' file(s).
            - For multiple files, use UNIX-style wildcards ('*' for any character(s), '?' for single character, etc.).
        get_sn (bool): Whether the serial number should be included in the column name. Defaults to False.

    Returns:
        pandas.DataFrame: Dataframe with time as index and the individual variables as columns.
    """

    if not isinstance(filepath, str):
        raise TypeError(
            f"Expected filepath as a string, but got {type(filepath).__name__}."
        )
    files: list[str] = glob.glob(filepath)
    if len(files) == 0:
        raise ValueError(f"No such file(s):'{filepath}'")
    for i in files:
        if not os.path.isfile(i):
            raise ValueError(
                f"Invalid input: '{i}'. Expected valid file(s) name with .txt extension."
            )
        if not i.endswith(".txt"):
            raise ValueError(f"Invalid file format: '{i}'. Expected a .txt file.")

    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="Could not infer format, so each element will be parsed individually, falling back to `dateutil`",
            category=UserWarning,
            module="dask.dataframe.io.csv",
        )
        warnings.filterwarnings(
            "ignore",
            message="Parsing dates in %Y.%m.%d %H:%M:%S format when dayfirst=True was specified. Pass `dayfirst=False` or specify a format to silence this warning.",
            category=UserWarning,
            module="dask.dataframe.io.csv",
        )
        d_df: ddf.DataFrame = ddf.read_csv(
            filepath,
            delimiter=";",
            skiprows=1,
            parse_dates=["Date Time, GMT+00:00"],
            dayfirst=True,
            encoding="ISO-8859-1",
            thousands=",",
        )
        df: pd.DataFrame = d_df.compute()

    df.rename({"Date Time, GMT+00:00": "TIMESTAMP"}, axis=1, inplace=True)
    df["TIMESTAMP"] = pd.to_datetime(df["TIMESTAMP"])
    df.set_index("TIMESTAMP", inplace=True)
    df.sort_index(inplace=True)

    df = uf.std_names(df, bonus=get_sn)

    return df


def read_IWIN(filepath: str) -> xr.Dataset:
    """Reads data from one or several netCDF data files from IWIN stations.

    Args:
        filepath (str): Path to one or more '.nc' file(s).
            - For multiple files, use UNIX-style wildcards ('*' for any character(s), '?' for single character, etc.).

    Returns:
        xarray.Dataset: Dataset representing the netCDF file(s).
    """

    if not isinstance(filepath, str):
        raise TypeError(
            f"Expected filepath as a string, but got {type(filepath).__name__}."
        )
    files: list[str] = sorted(glob.glob(filepath))
    if len(files) == 0:
        raise ValueError(f"No such file(s):'{filepath}'")
    for i in files:
        if not os.path.isfile(i):
            raise ValueError(
                f"Invalid input: '{i}'. Expected valid file(s) name with .nc extension."
            )
        if not i.endswith(".nc"):
            raise ValueError(f"Invalid file format: '{i}'. Expected a .nc file.")
    if len(files) == 1:
        with xr.open_dataset(files[0]) as f:
            ds: xr.Dataset = f.load()
    elif len(files) > 1:
        with xr.open_mfdataset(files) as f:
            ds: xr.Dataset = f.load()

    ds = uf.std_names(ds)

    return ds


def read_AROME(filepath: str) -> xr.Dataset:
    """Reads data from one or several netCDF data files from AROME-Arctic.

    Args:
        filepath (str): Path to one or more '.nc' file(s).
            - For multiple files, use UNIX-style wildcards ('*' for any character(s), '?' for single character, etc.).

    Returns:
        xarray.Dataset: Dataset representing the netCDF file(s).
    """

    if not isinstance(filepath, str):
        raise TypeError(
            f"Expected filepath as a string, but got {type(filepath).__name__}."
        )
    files: list[str] = sorted(glob.glob(filepath))
    if len(files) == 0:
        raise ValueError(f"No such file(s):'{filepath}'")
    for i in files:
        if not os.path.isfile(i):
            raise ValueError(
                f"Invalid input: '{i}'. Expected valid file(s) name with .nc extension."
            )
        if not i.endswith(".nc"):
            raise ValueError(f"Invalid file format: '{i}'. Expected a .nc file.")
    if len(files) == 1:
        with xr.open_dataset(filepath) as f:
            ds: xr.Dataset = f.load()
    elif len(files) > 1:
        with xr.open_mfdataset(filepath) as f:
            ds: xr.Dataset = f.load()

    ds = uf.std_names(ds)

    return ds


def read_radiosonde(
    filepath: str, date: str = pd.Timestamp.now().strftime("%Y%m%d")
) -> pd.DataFrame:
    """Reads data from one or several data files from the small re-usable radiosondes.

    Args:
        filepath (str):  Path to one or more '.csv' file(s).
            - For multiple files, use UNIX-style wildcards ('*' for any character(s), '?' for single character, etc.).
        date (str, optional): String specifying the date of the sounding (the output file doesn't include the date) Format: YYYYmmdd, default: today.

    Returns:
        pd.DataFrame: Dataframe with time as index and the individual variables as columns.
    """

    if not isinstance(filepath, str):
        raise TypeError(
            f"Expected filepath as a string, but got {type(filepath).__name__}."
        )
    files: list[str] = glob.glob(filepath)
    if len(files) == 0:
        raise ValueError(f"No such file(s):'{filepath}'")
    for i in files:
        if not os.path.isfile(i):
            raise ValueError(
                f"Invalid input: '{i}'. Expected valid file(s) name with .csv extension."
            )
        if not i.endswith(".csv"):
            raise ValueError(f"Invalid file format: '{i}'. Expected a .csv file.")

    if not isinstance(date, str):
        raise TypeError(f"Expected date as a string, but got {type(date).__name__}.")
    try:
        datetime.datetime.strptime(date, "%Y%m%d")
        # checks for the right format
    except ValueError:
        raise ValueError(f"Invalid date format '{date}'. Use YYYYmmdd.")

    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="Could not infer format, so each element will be parsed individually, falling back to `dateutil`",
            category=UserWarning,
            module="dask.dataframe.io.csv",
        )
        warnings.filterwarnings(
            "ignore",
            message="Parsing dates in %Y.%m.%d %H:%M:%S format when dayfirst=True was specified. Pass `dayfirst=False` or specify a format to silence this warning.",
            category=UserWarning,
            module="dask.dataframe.io.csv",
        )
        d_df: ddf.DataFrame = ddf.read_csv(
            filepath,
            delimiter=",",
            parse_dates=["UTC time"],
            encoding="ISO-8859-1",
            na_values=["", " ", "  ", "   "],
        )
    df: pd.DataFrame = d_df.compute()
    df["UTC time"] = [
        dt.replace(year=int(date[:4]), month=int(date[4:6]), day=int(date[6:]))
        for dt in df["UTC time"]
    ]

    df.rename({"UTC time": "TIMESTAMP"}, axis=1, inplace=True)
    df["TIMESTAMP"] = pd.to_datetime(df["TIMESTAMP"])
    df.set_index("TIMESTAMP", inplace=True)
    df.sort_index(inplace=True)

    df = uf.std_names(df)

    if "p [Pa]" in df.columns:
        df["p [hPa]"] = df["p [Pa]"] * 0.01
        df.drop("p [Pa]", inplace=True, axis=1)

    return df


def read_iMet(filepath: str) -> pd.DataFrame:
    """Reads data from one or several data files from the iMet sensors.

    Args:
        filepath (str): Path to one or more '.csv' file(s).
            - For multiple files, use UNIX-style wildcards ('*' for any character(s), '?' for single character, etc.).

    Returns:
        pandas.DataFrame: Dataframe with time as index and the individual variables as columns.
    """

    if not isinstance(filepath, str):
        raise TypeError(
            f"Expected filepath as a string, but got {type(filepath).__name__}."
        )
    files: list[str] = glob.glob(filepath)
    if len(files) == 0:
        raise ValueError(f"No such file(s):'{filepath}'")
    for i in files:
        if not os.path.isfile(i):
            raise ValueError(
                f"Invalid input: '{i}'. Expected valid file(s) name with .csv extension."
            )
        if not i.endswith(".csv"):
            raise ValueError(f"Invalid file format: '{i}'. Expected a .csv file.")

    d_df: ddf.DataFrame = ddf.read_csv(
        filepath, delimiter=",", encoding="ISO-8859-1"
    )  # , parse_dates=[["Date", "Time"]], dayfirst=True
    df: pd.DataFrame = d_df.compute()
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="Parsing dates in %Y.%m.%d %H:%M:%S format when dayfirst=True was specified. Pass `dayfirst=False` or specify a format to silence this warning.",
            category=UserWarning,
        )
        df["TIMESTAMP"] = pd.to_datetime(df["Date"] + " " + df["Time"], dayfirst=True)
    df.drop(["ID", "Date", "Time"], axis=1, inplace=True)
    df.rename(
        dict(
            zip(
                list(df.keys()),
                [k.replace("XQ-iMet-XQ ", "").strip() for k in list(df.keys())],
            )
        ),
        axis=1,
        inplace=True,
    )
    df.set_index("TIMESTAMP", inplace=True)
    df.sort_index(inplace=True)

    unit_conversion: dict[str, float] = {
        "Pressure": 100.0,
        "Air Temperature": 100.0,
        "Humidity": 10.0,
        "Longitude": 1.0e7,
        "Latitude": 1.0e7,
        "Altitude": 1000.0,
    }
    for k, factor in unit_conversion.items():
        df[k] = df[k] / factor
    try:
        df["Humidity Temp"] = df["Humidity Temp"] / 100.0
    except KeyError:
        pass

    for col_name in df.columns:
        if col_name == "Lon":
            if df["Longitude"].equals(df["Lon"]):
                df.drop("Lon", axis=1, inplace=True)
        if col_name == "Lat":
            if df["Latitude"].equals(df["Lat"]):
                df.drop("Lat", axis=1, inplace=True)

    df = uf.std_names(df, add_units=True)

    return df


############################################################################
# DOWNLOADING FUNCTIONS
############################################################################

iwin_stations = Literal[
    "MSBard",
    "MSBerg",
    "MSBillefjord",
    "MSPolargirl",
    "RVHannaResvoll",
    "Narveneset",
    "Bohemanneset",
    "Daudmannsodden",
    "Gasoyane",
    "KappThordsen",
]


def download_IWIN_from_THREDDS(
    station_name: str | iwin_stations,
    start_time: str,
    end_time: str,
    local_out_path: str = os.getcwd(),
    resolution: str | Literal["20sec", "1min", "10min"] = "1min",
) -> None:
    """Function to download data from one IWIN station and save it locally in a netCDF file.

    Args:
        station_name (str): String specifying the station.
            - Available options are MSBerg (2023-), MSBard (2021-2022), MSPolargirl, MSBillefjord, RVHannaResvoll (2024-), Bohemanneset (2021-), Narveneset (2022-), Daudmannsodden (2022-), Gåsøyane (2022-), KappThordsen (2023-).
        start_time (str): String specifying the first hour to download. Format YYYY-MM-DD HH.
        end_time (str): String specifying the last hour (included) to download. Format YYYY-MM-DD HH.
            - If you wish to download data from exactly one day, set HH=23, 00 from the following day.
        local_out_path (str, optional): String specifying the path to the folder where the data should be saved. The default is the current working directory.
            - Don't add a file name here, the file name will be given automatically from the script.
        resolution (str, optional): String specifying the temporal resolution of the time series to download. Defaults to "1min".
            - Available options are '1min' and '10min' for lighthouse stations and additionally '20sec' for mobile stations.

    Returns:
        None
    """

    if not isinstance(local_out_path, str):
        raise TypeError(
            f"Expected filepath as a string, but got {type(local_out_path).__name__}."
        )
    elif not os.path.isdir(local_out_path):
        raise NotADirectoryError(f"'{local_out_path}' is not a valid directory.")

    start_time_dt: datetime.datetime = datetime.datetime.strptime(
        start_time, "%Y-%m-%d %H"
    ).replace(tzinfo=datetime.timezone.utc)
    end_time_dt: datetime.datetime = datetime.datetime.strptime(
        end_time, "%Y-%m-%d %H"
    ).replace(tzinfo=datetime.timezone.utc)

    path_base: str = "https://thredds.met.no/thredds/dodsC/met.no/observations/unis/"

    if station_name in [
        "MSBard",
        "MSBerg",
        "MSBillefjord",
        "MSPolargirl",
        "RVHannaResvoll",
    ]:
        if resolution in ["20sec", "1min", "10min"]:
            path_data: str = f"{path_base}mobile_AWS_{station_name}_{resolution}"
            path_out: str = os.path.join(
                local_out_path,
                f"mobile_AWS_{station_name}_{resolution}_{start_time_dt.strftime('%Y%m%d%H')}_{end_time_dt.strftime('%Y%m%d%H')}.nc",
            )
        else:
            raise ValueError(
                f"Requested resolution not available for '{station_name}'. Please choose '20sec', '1min' or '10min'."
            )
    elif station_name in [
        "Narveneset",
        "Bohemanneset",
        "Daudmannsodden",
        "Gasoyane",
        "KappThordsen",
    ]:
        if resolution in ["1min", "10min"]:
            path_data: str = f"{path_base}lighthouse_AWS_{station_name}_{resolution}"
            path_out: str = os.path.join(
                local_out_path,
                f"lighthouse_AWS_{station_name}_{resolution}_{start_time_dt.strftime('%Y%m%d%H')}_{end_time_dt.strftime('%Y%m%d%H')}.nc",
            )
        else:
            raise ValueError(
                f"Requested resolution not available for '{station_name}'. Please choose '1min' or '10min'."
            )
    else:
        raise ValueError(
            f"Requested station name '{station_name}' not recognized. Please choose from 'MSBard', 'MSBerg', 'MSBillefjord', 'MSPolargirl', 'RVHannaResvoll', 'Narveneset', 'Bohemanneset', 'Daudmannsodden', 'Gasoyane', 'KappThordsen'."
        )

    logging.info("Download starting...")
    with xr.open_dataset(path_data) as f:
        ds: xr.Dataset = f.sel(time=slice(start_time, end_time))

    ds.to_netcdf(path_out, unlimited_dims=["time"])

    logging.info(
        f"The following dataset was successfully downloaded and saved in {path_out}."
    )
    logging.info(ds)

    return None


############################################################################
# PLOTTING FUNCTIONS
############################################################################


class MapGenerator:
    def __init__(
        self,
        lat_limits: list[int | float],
        lon_limits: list[int | float],
        nrows: int = 1,
        ncols: int = 1,
        path_mapdata: str | None = ...,
        subplots_parameters: dict[str, Any] = None,
        gridlines_parameters: dict[str, Any] = None,
        aspect: Literal["auto", "equal"] | float = None,
    ) -> None:
        """Function to initialize an empty map using a Mercator projection.
        This function neither draws features like coastlines, topography etc.
        nor actual meteorological data. It should be used before adding the
        above-mentioned elements using the respective functions included in
        this package.

        Args:
            lat_limits (list[int|float]): List with two elements: [lat_min, lat_max].
            lon_limits (list[int|float]): List with two elements: [lon_min, lon_max].
            nrows (int, optional): Number of rows in the subplot grid. Defaults to 1.
            ncols (int, optional): Number of columns in the subplot grid. Defaults to 1.
            figsize (tuple[int,int], optional): Size of figure. Defaults to (12,12).
            path_mapdata (str, optional): Absolute or relative path to the directory containing the map data. Defaults to not set.
                - If specified, this path will be used as the default for all map features.
                - Individual functions can override this setting by specifying their own path_mapdata parameter.
            subplots_parameters (dict[str,Any], optional): Additional parameters passed to plt.subplots. Defaults to None.
                - For available options check matplotlib.pyplot.subplots.
            gridlines_parameters (dict[str,Any],optional): Additional parameters passed to GeoAxes.gridlines. Defaults to None.
                - For available options check cartopy.mpl.geoaxes.GeoAxes.gridlines.
                - If not specified, draw_labels=False is always set by default.
            aspect (str, optional): Aspect ratio of the map. Defaults to None.
                - For available options check cartopy.mpl.geoaxes.GeoAxes.set_aspect.
        """

        if subplots_parameters is None:
            subplots_parameters = {}
        elif not isinstance(subplots_parameters, dict):
            raise TypeError("'subplots_parameters' should be a dictionary.")

        if gridlines_parameters is None:
            gridlines_parameters = {"draw_labels": False}
        elif not isinstance(gridlines_parameters, dict):
            raise TypeError("'gridlines_parameters' should be a dictionary.")
        gridlines_parameters.setdefault("draw_labels", False)

        if len(lon_limits) != 2:
            raise ValueError(f"'lon_limits' should contain exactly two values.")
        if not all(isinstance(x, (int, float)) for x in lon_limits):
            raise TypeError(f"'lon_limits' should contain just floats.")

        if len(lat_limits) != 2:
            raise ValueError(f"'lat_limits' should contain exactly two values.")
        if not all(isinstance(x, (int, float)) for x in lat_limits):
            raise TypeError(f"'lat_limits' should contain just floats.")

        if not isinstance(nrows, int):
            raise TypeError(f"'nrows' should be a int, not a {type(nrows).__name__}.")

        if not isinstance(ncols, int):
            raise TypeError(f"'nrows' should be a int, not a {type(ncols).__name__}.")

        if not (isinstance(path_mapdata, str) or path_mapdata == ...):
            raise TypeError(
                f"'path_mapdata' should be a str, not a {type(path_mapdata).__name__}."
            )
        elif not path_mapdata == ...:
            if not path_mapdata.endswith("/"):
                path_mapdata += "/"
            if not os.path.isdir(path_mapdata):
                raise NotADirectoryError(f"'{path_mapdata}' is not a valid directory.")

        self.lat_limits: list[float] = lat_limits
        self.lon_limits: list[float] = lon_limits

        self.path_mapdata: str | None = path_mapdata

        self.fig: Figure
        self.fig, self.ax = plt.subplots(
            nrows,
            ncols,
            subplot_kw={"projection": ccrs.Mercator()},
            **subplots_parameters,
        )
        self.ax: np.ndarray[GeoAxes]
        self.ax = np.array(self.ax) if (nrows > 1 or ncols > 1) else np.array([self.ax])

        if aspect is not None:
            if isinstance(aspect, str):
                if aspect not in ["auto", "equal"]:
                    raise ValueError(
                        f"'aspect' should be 'auto', 'equal' or a number, not '{aspect}'."
                    )
            elif not isinstance(aspect, (int, float)):
                raise TypeError(
                    f"'aspect' should be a str or a number, not a {type(aspect).__name__}."
                )
        else:
            aspect = self.ax.flat[0].get_aspect()

        self.aspect: str = aspect

        for ax in self.ax.flat:
            ax.set_extent(self.lon_limits + self.lat_limits, crs=ccrs.PlateCarree())
            ax.set_aspect(aspect)
            gl = ax.gridlines(**gridlines_parameters)
            gl.left_labels = True
            gl.bottom_labels = True
            gl.xformatter = LONGITUDE_FORMATTER
            gl.yformatter = LATITUDE_FORMATTER

    def __str__(self) -> str:
        if len(self.ax) == 1:
            return f"The map is in the limits of lon: {self.lon_limits} and lat: {self.lat_limits}."
        else:
            return f"{len(self.ax)} maps in the limits of lon: {self.lon_limits} and lat: {self.lat_limits}."

    def add_coastline(
        self,
        option: int,
        color: (
            str | tuple[float, float, float] | tuple[float, float, float, float]
        ) = "black",
        ax: int = 0,
        path_mapdata: str = ...,
        custom_path: str = ...,
        plot_parameters: dict[str, Any] = None,
    ) -> Self:
        """Function to add a coastline to a previously created MapGenerator object.
        Several options are available for the resolution of the data used to draw
        the coastline (see below).

        Args:
            option (int): Switch to distinguish different resolutions. Valid options:
                0 : Low resolution coastline based on the cartopy database.
                1 : Medium resolution (250 m) based on data from the Norwegian Polar Institute.
                2 : High resolution (100 m) based on data from the Norwegian Polar Institute.
                3 : Uses 'custom_path' for map data.
                    - Needs a .shp file with the coastline data.
            color (str or RGB or RGBA, optional): Color for the coastline. Defaults to "black".
            ax (int, optional): Gives the position for which map the coastline is added. Defaults to 0.
                - Starts to count from 0 and continues like the normal reading flow.
            path_mapdata (str, optional): Absolute or relative path to the directory including the map data. Defaults to not set.
                - Will be set as default for the map(s), if not set in the class initialisation.
            custom_path (str, optional): Absolute or relative path to a shape file. Defaults to not set.
            plot_parameters (dict[str,Any], optional):
                option:
                    0 : - For available options check cartopy.mpl.geoaxes.coastlines.
                        - Always sets color = color.
                        - If not specified,  linewidth = 1.5, resolution = "10m" is always set by default.
                    1-3 : - For available options check geopandas.GeoDataFrame.plot.
                          - Always sets edgecolor = color.
                          - If not specified, facecolor = "none", zorder = 20, lw = 1 is always set by default.

        Returns:
            MapGenerator
        """
        if not isinstance(option, int):
            raise TypeError(f"'option' should be a int, not a {type(option).__name__}.")

        if not isinstance(color, (str, tuple)):
            raise TypeError(
                f"'color' should be a str or tuple, not a {type(color).__name__}."
            )

        if not isinstance(ax, int):
            raise TypeError(f"'ax' should be a int, not a {type(ax).__name__}.")
        if ax >= len(self.ax.flat):
            raise ValueError(
                f"Invalid axis index {ax}, max index is {len(self.ax.flat)-1}"
            )

        if path_mapdata == ... and option != 0:
            if self.path_mapdata != ...:
                path_mapdata = self.path_mapdata
            else:
                raise ValueError(
                    f"'path_mapdata' needs to be set, if option 1, 2 or 3 is selected."
                )
        if not (isinstance(path_mapdata, str) or option == 0):
            raise TypeError(
                f"'path_mapdata' should be a str, not a {type(path_mapdata).__name__}."
            )
        elif not (path_mapdata == ... or option == 0):
            if not path_mapdata.endswith("/"):
                path_mapdata += "/"
            if not os.path.isdir(path_mapdata):
                raise NotADirectoryError(f"'{path_mapdata}' is not a valid directory.")
        if not path_mapdata == ... and self.path_mapdata == ...:
            self.path_mapdata = path_mapdata
            logging.info(
                f"Changing the path_mapdata for the object to '{path_mapdata}'. "
            )

        if plot_parameters == None:
            plot_parameters = {}
        if not isinstance(plot_parameters, dict):
            raise TypeError(
                f"'plot_parameters' should be a dictionary, not a {type(plot_parameters).__name__}."
            )

        title: str = self.ax.flat[ax].get_title()
        xlabel: str = self.ax.flat[ax].get_xlabel()
        ylabel: str = self.ax.flat[ax].get_ylabel()

        if option == 0:
            plot_parameters.setdefault("resolution", "10m")
            plot_parameters["color"] = color
            plot_parameters.setdefault("linewidth", 1.5)
            self.ax.flat[ax].coastlines(**plot_parameters)
        elif option == 1:
            input_file: str = f"{path_mapdata}NP_S250_SHP/S250_Land_l.shp"
            df_maplayer = gpd.read_file(input_file)
            df_maplayer = df_maplayer.to_crs(ccrs.Mercator().proj4_init)
            plot_parameters["edgecolor"] = color
            plot_parameters.setdefault("facecolor", "none")
            plot_parameters.setdefault("zorder", 20)
            plot_parameters.setdefault("lw", 1.0)
            df_maplayer.plot(ax=self.ax.flat[ax], **plot_parameters)
            self.ax.flat[ax].set_extent(
                self.lon_limits + self.lat_limits, crs=ccrs.PlateCarree()
            )
        elif option == 2:
            input_file = f"{path_mapdata}NP_S100_SHP/S100_Land_l.shp"
            df_maplayer = gpd.read_file(input_file)
            df_maplayer = df_maplayer.to_crs(ccrs.Mercator().proj4_init)
            plot_parameters["edgecolor"] = color
            plot_parameters.setdefault("facecolor", "none")
            plot_parameters.setdefault("zorder", 20)
            plot_parameters.setdefault("lw", 1.0)
            df_maplayer.plot(ax=self.ax.flat[ax], **plot_parameters)
            self.ax.flat[ax].set_extent(
                self.lon_limits + self.lat_limits, crs=ccrs.PlateCarree()
            )
        elif option == 3:
            if custom_path == ...:
                raise ValueError(f"'custom_path' needs to be set for option 3.")
            if not isinstance(custom_path, str):
                raise TypeError(
                    f"'custom_path' should be a str, not a {type(custom_path).__name__}."
                )
            if not os.path.isfile(custom_path):
                raise FileNotFoundError(f"Expected file not found: {custom_path}")
            if not custom_path.endswith(".shp"):
                raise ValueError(
                    f"Invalid file format: {custom_path}. Expected a .shp file."
                )
            df_maplayer = gpd.read_file(custom_path)
            df_maplayer = df_maplayer.to_crs(ccrs.Mercator().proj4_init)
            plot_parameters["edgecolor"] = color
            plot_parameters.setdefault("facecolor", "none")
            plot_parameters.setdefault("zorder", 20)
            plot_parameters.setdefault("lw", 1.0)
            df_maplayer.plot(ax=self.ax.flat[ax], **plot_parameters)
            self.ax.flat[ax].set_extent(
                self.lon_limits + self.lat_limits, crs=ccrs.PlateCarree()
            )
        else:
            raise ValueError(f"{option} not a valid option!")

        self.ax.flat[ax].set_title(title)
        self.ax.flat[ax].set_xlabel(xlabel)
        self.ax.flat[ax].set_ylabel(ylabel)
        self.ax.flat[ax].set_aspect(self.aspect)

        return self

    def add_land_filled(
        self,
        option: int,
        color: str | tuple[float, float, float] | tuple[float, float, float, float],
        ax: int = 0,
        path_mapdata: str = ...,
        custom_path: str = ...,
        plot_parameters: dict[str, Any] = None,
    ) -> Self:
        """Function to fill land areas in a previously created MapGenerator object with the specified color.
        Several options are available for the resolution of the data used to define
        the land area (see below).

        Args:
            option (int): Switch to distinguish different resolutions. Valid options:
                0 : Low resolution based on the cartopy database
                1 : Medium resolution (250 m) based on data from the Norwegian Polar Institute
                2 : High resolution (100 m) based on data from the Norwegian Polar Institute
                3 : Uses 'custom_path' for map data.
                    - Needs a .shp file with the land data.
            color (str or RGB or RGBA): Color for the land patches.
            ax (int, optional): Gives the position for which map the land fill is added. Defaults to 0.
                - Starts to count from 0 and continues like the normal reading flow.
            path_mapdata (str, optional): Absolute or relative path to the directory including the map data. Defaults to not set.
                - Will be set as default for the map(s), if not set in the class initialisation.
            custom_path (str, optional): Absolute or relative path to a shape file. Defaults to not set.
            plot_parameters (dict[str, Any], optional):
                option:
                    0 : - For available options check cartopy.mpl.geoaxes.coastlines.
                        - Always sets facecolor = color.
                        - If not specified, feature = cfeature.LAND.with_scale('10m') is always set by default.
                    1-3: - For available options check geopandas.GeoDataFrame.plot.
                         - Always sets facecolor=color.

        Returns:
            MapGenerator
        """
        if not isinstance(option, int):
            raise TypeError(f"'option' should be a int, not a {type(option).__name__}.")

        if not isinstance(color, (str, tuple)):
            raise TypeError(
                f"'color' should be a str or tuple, not a {type(color).__name__}."
            )
        if not isinstance(ax, int):
            raise TypeError(f"'ax' should be a int, not a {type(ax).__name__}.")
        if ax >= len(self.ax.flat):
            raise ValueError(
                f"Invalid axis index {ax}, max index is {len(self.ax.flat)-1}"
            )

        if path_mapdata == ... and option != 0:
            if self.path_mapdata != ...:
                path_mapdata = self.path_mapdata
            else:
                raise ValueError(
                    f"'path_mapdata' needs to be set, if option 1 or 2 is selected."
                )
        if not (isinstance(path_mapdata, str) or option == 0):
            raise TypeError(
                f"'path_mapdata' should be a str, not a {type(path_mapdata).__name__}."
            )
        elif not (path_mapdata == ... or option == 0):
            if not path_mapdata.endswith("/"):
                path_mapdata += "/"
            if not os.path.isdir(path_mapdata):
                raise NotADirectoryError(f"'{path_mapdata}' is not a valid directory.")
        if not path_mapdata == ... and self.path_mapdata == ...:
            self.path_mapdata = path_mapdata
            logging.info(
                f"Changing the 'path_mapdata' for the object to '{path_mapdata}'. "
            )

        if plot_parameters == None:
            plot_parameters = {}
        if not isinstance(plot_parameters, dict):
            raise TypeError(f"'plot_parameters' should be a dictionary.")

        title: str = self.ax.flat[ax].get_title()
        xlabel: str = self.ax.flat[ax].get_xlabel()
        ylabel: str = self.ax.flat[ax].get_ylabel()

        plot_parameters["facecolor"] = color
        if option == 0:
            plot_parameters.setdefault("feature", cfeature.LAND.with_scale("10m"))
            self.ax.flat[ax].add_feature(**plot_parameters)
        elif option == 1:
            input_file: str = f"{path_mapdata}NP_S250_SHP/S250_Land_f.shp"
            df_maplayer = gpd.read_file(input_file)
            df_maplayer = df_maplayer.to_crs(ccrs.Mercator().proj4_init)
            df_maplayer.plot(ax=self.ax.flat[ax], **plot_parameters)
            self.ax.flat[ax].set_extent(
                self.lon_limits + self.lat_limits, crs=ccrs.PlateCarree()
            )
        elif option == 2:
            input_file = f"{path_mapdata}NP_S100_SHP/S100_Land_f.shp"
            df_maplayer = gpd.read_file(input_file)
            df_maplayer = df_maplayer.to_crs(ccrs.Mercator().proj4_init)
            df_maplayer.plot(ax=self.ax.flat[ax], **plot_parameters)
            self.ax.flat[ax].set_extent(
                self.lon_limits + self.lat_limits, crs=ccrs.PlateCarree()
            )
        elif option == 3:
            if custom_path == ...:
                raise ValueError(f"'custom_path' needs to be set for option 3.")
            if not isinstance(custom_path, str):
                raise TypeError(
                    f"'custom_path' should be a str, not a {type(custom_path).__name__}."
                )
            if not os.path.isfile(custom_path):
                raise FileNotFoundError(f"Expected file not found: {custom_path}")
            if not custom_path.endswith(".shp"):
                raise ValueError(
                    f"Invalid file format: {custom_path}. Expected a .shp file."
                )
            df_maplayer = gpd.read_file(custom_path)
            df_maplayer = df_maplayer.to_crs(ccrs.Mercator().proj4_init)
            df_maplayer.plot(ax=self.ax.flat[ax], **plot_parameters)
            self.ax.flat[ax].set_extent(
                self.lon_limits + self.lat_limits, crs=ccrs.PlateCarree()
            )
        else:
            raise ValueError(f"{option} not a valid option!")

        self.ax.flat[ax].set_title(title)
        self.ax.flat[ax].set_xlabel(xlabel)
        self.ax.flat[ax].set_ylabel(ylabel)
        self.ax.flat[ax].set_aspect(self.aspect)

        return self

    def add_bathymetry(
        self,
        option: int,
        color_contourlines: (
            str
            | tuple[float, float, float]
            | tuple[float, float, float, float]
            | LinearSegmentedColormap
        ),
        contour_params: float | npt.ArrayLike,
        ax: int = 0,
        path_mapdata: str = ...,
        plot_parameters: dict[str, Any] = None,
        label_parameters: dict[str, Any] = None,
        more_custom: bool = False,
        used_EPSG: int = 3996,
    ) -> Self | tuple[Self, QuadContourSet, Colorbar | None]:
        """Function to plot either contour lines of the bathymetry of an ocean area
        with the specified color or colored contours of bathymetry.
        Several options are available for the style of
        the bathymetry contours (see below).

        Args:
            option (int): Switch to distinguish different styles. Valid options:
                0 : Bathymetry as contour lines.
                1 : Bathymetry as filled contours, no colorbar.
                2 : Bathymetry as filled contours, with colorbar.
                3X: Uses path_mapdata as costom path for map data.
                    - X = 0, 1, 2 : Same as above.
                    - Needs a .tif file. (Projection should be EPSG:3996)
            color_contourlines (str or RGB or RGBA or LinearSegmentedColormap): Color for the topography contour lines (only used with option 0).
            contour_params (float or array_like): At which levels contour levels shoud be.
                - single value : Resolution (distance between contour levels) of the bathymetry.
                - array_like : Contour levels of the bathymetry.
            ax (int, optional): Gives the position for which map the bathymetry is added. Defaults to 0.
                - Starts to count from 0 and continues like the normal reading flow.
            path_mapdata (str, optional): Absolute or relative path to the directory including the map data. Defaults to not set.
                - Will be set as default for the map(s), if not set in the class initialisation.
            plot_parameters (dict[str, Any], optional):
                option:
                    0 : - For available options check xarray.DataArray.plot.contour.
                        - Always sets colors = color_contourlines, levels = contour_params.
                        - If not specified, linestyles = "-", linewidths = 0.5, is always set by default.
                    1-2 : - For available options check xarray.DataArray.plot.imshow.
                          - Always sets levels = contour_params.
                          - If not specified, cmap = cmocean.cm.deep_r, interpolation = None, add_colorbar = False, is always set by default.
            label_parameters (dict[str, Any], optional):
                option:
                    0 : - For available options check cartopy.mpl.geoaxes.clabel.
                        - Always sets levels = contour_params.
                        - If not specified, inline = True, fmt = "%.0f", fontsize = 10 is always set by default.
                    2 : - For available options check matplotlib.pyplot.colorbar.
                        - If not specified, pad = 0.02, extend = "neither", tick_params = {}, set_ylabel = {}, is always set by default.
                        - To modify matplotlib.colorbar.Colorbar.ax.tick_params or .ax.set_ylabel, use:
                            - 'tick_params = {}' (default: axis = "y", labelsize = 10).
                            - 'set_ylabel = {}' (default: ylabel = "Height [m]", fontsize = 10).
            more_custom (bool, optional): If True, returns additional objects ('QuadContourSet' and 'Colorbar'). Defaults to False.
            used_EPSG (int, optional): EPSG code of the map data. Defaults to 3996.
                - Don't change unless you are providing your own map data.

        Returns:
            MapGenerator or tuple[MapGenerator, QuadContourSet, Colorbar or None]:
                - If more_custom = False: Returns self.
                - If more_custom = True: Returns a tuple with extra objects.
        """
        if not isinstance(option, int):
            raise TypeError(f"'option' should be a int, not a {type(option).__name__}.")

        if not isinstance(
            color_contourlines, (str, tuple, mpl.colors.LinearSegmentedColormap)
        ):
            raise TypeError(
                f"'color' should be a str or tuple, not a {type(color_contourlines).__name__}."
            )

        if not isinstance(ax, int):
            raise TypeError(f"'ax' should be a int, not a {type(ax).__name__}.")
        if ax >= len(self.ax.flat):
            raise ValueError(
                f"Invalid axis index {ax}, max index is {len(self.ax.flat)-1}"
            )

        if path_mapdata == ...:
            if self.path_mapdata != ...:
                path_mapdata = self.path_mapdata
                if option > 3:
                    logging.warning(
                        f"Using 'path_mapdata' ({path_mapdata}) from the class. Be aware that you chose a option with a costum path."
                    )
            else:
                raise ValueError(f"'path_mapdata' needs to be set.")
        if not isinstance(path_mapdata, str):
            raise TypeError(
                f"'path_mapdata' should be a str, not a {type(path_mapdata).__name__}."
            )
        elif not path_mapdata == ... and option < 3:
            if not path_mapdata.endswith("/"):
                path_mapdata += "/"
            if not os.path.isdir(path_mapdata):
                raise NotADirectoryError(f"'{path_mapdata}' is not a valid directory.")
        if not path_mapdata == ... and self.path_mapdata == ... and option < 3:
            self.path_mapdata = path_mapdata
            logging.info(
                f"Changing the 'path_mapdata' for the object to '{path_mapdata}'. "
            )

        if plot_parameters == None:
            plot_parameters = {}
        if not isinstance(plot_parameters, dict):
            raise TypeError(
                f"'plot_parameters' should be a dictionary, not a {type(plot_parameters).__name__}."
            )

        if label_parameters == None:
            label_parameters = {}
        if not isinstance(label_parameters, dict):
            raise TypeError(
                f"'label_parameters' should be a dictionary, not a {type(label_parameters).__name__}."
            )

        if not isinstance(more_custom, bool):
            raise TypeError(
                f"'more_custom' should be a bool, not a {type(more_custom).__name__}."
            )

        if not isinstance(used_EPSG, int):
            raise TypeError(
                f"'used_EPSG' should be a int, not a {type(used_EPSG).__name__}."
            )

        title: str = self.ax.flat[ax].get_title()
        xlabel: str = self.ax.flat[ax].get_xlabel()
        ylabel: str = self.ax.flat[ax].get_ylabel()

        if int(option / 10) == 3:
            if not os.path.isfile(path_mapdata):
                raise FileNotFoundError(f"File not found: {path_mapdata}")
            if not path_mapdata.endswith(".tif"):
                raise ValueError(
                    f"Invalid file format: {path_mapdata}. Expected a .tif file."
                )
            path_ibcao: str = path_mapdata
            option = option - 30
        else:
            if not used_EPSG == 3996:
                logging.warning(
                    f"Do not change the used_EPSG, if you are not unsing the costum path option. used_ESPG is set back to 3996."
                )
                used_EPSG = 3996
            path_ibcao: str = f"{self.path_mapdata}IBCAO/IBCAO_100m_v5_Svalbard.tif"

        bathy = rxr.open_rasterio(path_ibcao, masked=True).squeeze()
        bathy.rio.write_crs(used_EPSG, inplace=True)
        bathy = bathy.rio.reproject("EPSG:4326")
        bathy = bathy.rio.clip_box(
            minx=self.lon_limits[0],
            miny=self.lat_limits[0],
            maxx=self.lon_limits[1],
            maxy=self.lat_limits[1],
        )
        bathy = bathy.rio.reproject(ccrs.Mercator().proj4_init)
        bathy = bathy.where(bathy <= 10.0)

        if isinstance(contour_params, (float, int)):
            levels: np.ndarray[float] = np.arange(
                contour_params * np.floor(np.nanmin(bathy) / contour_params),
                1.0,
                contour_params,
            )
        elif pd.api.types.is_list_like(contour_params):
            if not all([isinstance(x, (int, float)) for x in contour_params]):
                raise ValueError(f"List of 'contour_params' shout just contain floats.")
            else:
                fail_values: list[float] = [
                    x
                    for x in contour_params
                    if not (x <= np.nanmax(bathy) and x >= np.nanmin(bathy))
                ]
                if len(fail_values) != 0:
                    raise ValueError(
                        f"The following 'contour_params' values are out of range: {fail_values}"
                    )
                levels = np.array(contour_params)
        else:
            raise TypeError(
                f"'contour_params' should be a tuple or a list, not a {type(contour_params).__name__}."
            )

        if option == 0:
            plot_parameters["colors"] = color_contourlines
            plot_parameters["levels"] = levels
            plot_parameters["ax"] = self.ax.flat[ax]
            plot_parameters.setdefault("linestyles", "-")
            plot_parameters.setdefault("linewidths", 0.5)
            pic = bathy.plot.contour(**plot_parameters)
            label_parameters["CS"] = pic
            label_parameters["levels"] = pic.levels
            label_parameters.setdefault("inline", True)
            label_parameters.setdefault("fmt", "%.0f")
            label_parameters.setdefault("fontsize", 10)
            self.ax.flat[ax].clabel(**label_parameters)
        elif option == 1:
            plot_parameters["ax"] = self.ax.flat[ax]
            plot_parameters["levels"] = levels
            plot_parameters.setdefault("cmap", cmo.cm.deep_r)
            plot_parameters.setdefault("interpolation", None)
            plot_parameters.setdefault("add_colorbar", False)
            bathy.plot.imshow(**plot_parameters)
        elif option == 2:
            plot_parameters["ax"] = self.ax.flat[ax]
            plot_parameters["levels"] = levels
            plot_parameters.setdefault("cmap", cmo.cm.deep_r)
            plot_parameters.setdefault("interpolation", None)
            plot_parameters.setdefault("add_colorbar", False)
            pic = bathy.plot.imshow(**plot_parameters)
            label_parameters["mappable"] = pic
            label_parameters["ax"] = self.ax.flat[ax]
            label_parameters.setdefault("pad", 0.02)
            label_parameters.setdefault("extend", "neither")
            tick_parameters: dict[str, Any] = label_parameters.pop("tick_params", {})
            if not isinstance(tick_parameters, dict):
                raise TypeError(
                    f"'tick_parameters' in 'label_parameters' should be a dict, not a {type(tick_parameters).__name__}."
                )
            set_ylabel_parameters: dict[str, Any] = label_parameters.pop(
                "set_ylabel", {}
            )
            if not isinstance(set_ylabel_parameters, dict):
                raise TypeError(
                    f"'set_ylabel_parameters' in 'label_parameters' should be a dict, not a {type(set_ylabel_parameters).__name__}."
                )
            cbar: plt.Colorbar = plt.colorbar(**label_parameters)
            tick_parameters.setdefault("axis", "y")
            tick_parameters.setdefault("labelsize", 10)
            cbar.ax.tick_params(**tick_parameters)
            set_ylabel_parameters.setdefault("ylabel", "Height [m]")
            set_ylabel_parameters.setdefault("fontsize", 10)
            cbar.ax.set_ylabel(**set_ylabel_parameters)
        else:
            raise ValueError(f"{option} not a valid option!")

        self.ax.flat[ax].set_title(title)
        self.ax.flat[ax].set_xlabel(xlabel)
        self.ax.flat[ax].set_ylabel(ylabel)
        self.ax.flat[ax].set_aspect(self.aspect)

        if more_custom:
            if option == 0:
                return self, pic
            elif option == 2:
                return self, pic, cbar
            else:
                warnings.warn(
                    f"There is nothing more to return for further customization of option '{option}'."
                )

        return self

    def add_total_topography(
        self,
        option: int,
        color_contourlines: (
            str
            | tuple[float, float, float]
            | tuple[float, float, float, float]
            | LinearSegmentedColormap
        ),
        contour_params: float | npt.ArrayLike,
        ax: int = 0,
        path_mapdata: str = ...,
        plot_parameters: dict[str, Any] = None,
        label_parameters: dict[str, Any] = None,
        more_custom: bool = False,
        used_EPSG: int = 3996,
    ) -> Self | tuple[Self, QuadContourSet, Colorbar | None]:
        """Function to plot either contour lines of the total topography (above and below sea level)
        with the specified color or colored contours of the total topography.
        Several options are available for the style of
        the topography contours (see below).

        Args:
            option (int): Switch to distinguish different styles. Valid options:
                0 : Topography as contour lines.
                1 : Topography as filled contours, no colorbar.
                2 : Topography as filled contours, with colorbar.
                3X: Uses path_mapdata as costom path for map data.
                    - X = 0, 1, 2 : Same as above.
                    - Needs a .tif file. (Projection should be EPSG:3996)
            color_contourlines (str or RGB or RGBA or LinearSegmentedColormap): Color for the topography contour lines (only used with option 0).
            contour_params (float or array_like): At which levels contour levels shoud be.
                - single value : Resolution (distance between contour levels) of the bathymetry.
                - array_like : Contour levels of the bathymetry. Can just be used with option 0.
            ax (int, optional): Gives the position for which map the topography is added. Defaults to 0.
                - Starts to count from 0 and continues like the normal reading flow.
            path_mapdata (str, optional): Absolute or relative path to the directory including the map data. Defaults to not set.
                - Will be set as default for the map(s), if not set in the class initialisation.
            plot_parameters (dict[str, Any], optional):
                option:
                    0 : - For available options check xarray.DataArray.plot.contour.
                        - Always sets colors = color_contourlines, levels = contour_params.
                        - If not specified, linestyles = "-", linewidths = 0.5, is always set by default.
                    1-2 : - For available options check xarray.DataArray.plot.imshow.
                          - Always sets norm = contour_params.
                          - If not specified, cmap = cmocean.cm.topo, interpolation = None, add_colorbar = False, is always set by default.
            label_parameters (dict[str, Any], optional):
                option:
                    0 : - For available options check cartopy.mpl.geoaxes.clabel.
                        - Always sets levels = contour_params.
                        - If not specified, inline = True, fmt = "%.0f", fontsize = 10 is always set by default.
                    2 : - For available options check matplotlib.pyplot.colorbar.
                        - If not specified, pad = 0.02, extend = "neither", tick_params = {}, set_ylabel = {}, is always set by default.
                        - To modify matplotlib.colorbar.Colorbar.ax.tick_params or .ax.set_ylabel, use:
                            - 'tick_params = {}' (default: axis = "y", labelsize = 10).
                            - 'set_ylabel = {}' (default: ylabel = "Height [m]", fontsize = 10).
            more_custom (bool, optional): If True, returns additional objects ('QuadContourSet' and 'Colorbar'). Defaults to False.
            used_EPSG (int, optional): EPSG code of the map data. Defaults to 3996.
                - Don't change unless you are providing your own map data.

        Returns:
            MapGenerator or tuple[MapGenerator, QuadContourSet, Colorbar or None]:
                - If more_custom = False: Returns self.
                - If more_custom = True: Returns a tuple with extra objects.
        """
        if not isinstance(option, int):
            raise TypeError(f"'option' should be a int, not a {type(option).__name__}.")

        if not isinstance(
            color_contourlines, (str, tuple, mpl.colors.LinearSegmentedColormap)
        ):
            raise TypeError(
                f"'color' should be a str or tuple, not a {type(color_contourlines).__name__}."
            )

        if not isinstance(ax, int):
            raise TypeError(f"'ax' should be a int, not a {type(ax).__name__}.")
        if ax >= len(self.ax.flat):
            raise ValueError(
                f"Invalid axis index {ax}, max index is {len(self.ax.flat)-1}"
            )

        if path_mapdata == ...:
            if self.path_mapdata != ...:
                path_mapdata = self.path_mapdata
                if option > 3:
                    logging.warning(
                        f"Using 'path_mapdata' ({path_mapdata}) from the class. Be aware that you chose a option with a costum path."
                    )
            else:
                raise ValueError(f"'path_mapdata' needs to be set.")
        if not isinstance(path_mapdata, str):
            raise TypeError(
                f"'path_mapdata' should be a str, not a {type(path_mapdata).__name__}."
            )
        elif not path_mapdata == ... and option < 3:
            if not path_mapdata.endswith("/"):
                path_mapdata += "/"
            if not os.path.isdir(path_mapdata):
                raise NotADirectoryError(f"'{path_mapdata}' is not a valid directory.")
        if not path_mapdata == ... and self.path_mapdata == ... and option < 3:
            self.path_mapdata = path_mapdata
            logging.info(
                f"Changing the 'path_mapdata' for the object to '{path_mapdata}'. "
            )

        if plot_parameters == None:
            plot_parameters = {}
        if not isinstance(plot_parameters, dict):
            raise TypeError(
                f"'plot_parameters' should be a dictionary, not a {type(plot_parameters).__name__}."
            )

        if label_parameters == None:
            label_parameters = {}
        if not isinstance(label_parameters, dict):
            raise TypeError(
                f"'label_parameters' should be a dictionary, not a {type(label_parameters).__name__}."
            )

        if not isinstance(more_custom, bool):
            raise TypeError(
                f"'more_custom' should be a bool, not a {type(more_custom).__name__}."
            )

        if not isinstance(used_EPSG, int):
            raise TypeError(
                f"'used_EPSG' should be a int, not a {type(used_EPSG).__name__}."
            )

        title: str = self.ax.flat[ax].get_title()
        xlabel: str = self.ax.flat[ax].get_xlabel()
        ylabel: str = self.ax.flat[ax].get_ylabel()

        if int(option / 10) == 3:
            if not os.path.isfile(path_mapdata):
                raise FileNotFoundError(f"File not found: {path_mapdata}")
            if not path_mapdata.endswith(".tif"):
                raise ValueError(
                    f"Invalid file format: {path_mapdata}. Expected a .tif file."
                )
            path_ibcao: str = path_mapdata
            option = option - 30
        else:
            if not used_EPSG == 3996:
                logging.warning(
                    f"Do not change the used_EPSG, if you are not unsing the costum path option. used_ESPG is set back to 3996."
                )
                used_EPSG = 3996
            path_ibcao: str = f"{self.path_mapdata}IBCAO/IBCAO_100m_v5_Svalbard.tif"

        bathy = rxr.open_rasterio(path_ibcao, masked=True).squeeze()
        bathy.rio.write_crs(used_EPSG, inplace=True)
        bathy = bathy.rio.reproject("EPSG:4326")
        bathy = bathy.rio.clip_box(
            minx=self.lon_limits[0],
            miny=self.lat_limits[0],
            maxx=self.lon_limits[1],
            maxy=self.lat_limits[1],
        )
        bathy = bathy.rio.reproject(ccrs.Mercator().proj4_init)

        if isinstance(contour_params, (float, int)):
            if option == 0:
                levels: np.ndarray[float] = np.arange(
                    contour_params * np.floor(np.nanmin(bathy) / contour_params),
                    contour_params * np.ceil(np.nanmax(bathy) / contour_params) + 1.0,
                    contour_params,
                )
            else:
                levels = None
                norm = mpl.colors.TwoSlopeNorm(
                    0.0,
                    contour_params * np.floor(np.nanmin(bathy) / contour_params),
                    contour_params * np.ceil(np.nanmax(bathy) / contour_params),
                )
        elif pd.api.types.is_list_like(contour_params):
            if option != 0:
                raise ValueError(f"Lists or tuples can only be used with option 0.")
            if not all([isinstance(x, (int, float)) for x in contour_params]):
                raise ValueError(f"List of 'contour_params' shout just contain floats.")
            else:
                fail_values: list[float] = [
                    x
                    for x in contour_params
                    if not (x <= np.nanmax(bathy) and x >= np.nanmin(bathy))
                ]
                if len(fail_values) != 0:
                    raise ValueError(
                        f"The following 'contour_params' values are out of range: {fail_values}"
                    )
                levels = np.array(contour_params)
        else:
            raise TypeError(
                f"'contour_params' should be a tuple or a list, not a {type(contour_params).__name__}."
            )

        if option == 0:
            plot_parameters["colors"] = color_contourlines
            plot_parameters["levels"] = levels
            plot_parameters["ax"] = self.ax.flat[ax]
            plot_parameters.setdefault("linestyles", "-")
            plot_parameters.setdefault("linewidths", 0.5)
            pic = bathy.plot.contour(**plot_parameters)
            label_parameters["CS"] = pic
            label_parameters["levels"] = pic.levels
            label_parameters.setdefault("inline", True)
            label_parameters.setdefault("fmt", "%.0f")
            label_parameters.setdefault("fontsize", 10)
            self.ax.flat[ax].clabel(**label_parameters)
        elif option == 1:
            plot_parameters["ax"] = self.ax.flat[ax]
            plot_parameters["norm"] = norm
            plot_parameters.setdefault("cmap", cmo.cm.topo)
            plot_parameters.setdefault("interpolation", None)
            plot_parameters.setdefault("add_colorbar", False)
            bathy.plot.imshow(**plot_parameters)
        elif option == 2:
            plot_parameters["ax"] = self.ax.flat[ax]
            plot_parameters["norm"] = norm
            plot_parameters.setdefault("cmap", cmo.cm.topo)
            plot_parameters.setdefault("interpolation", None)
            plot_parameters.setdefault("add_colorbar", False)
            pic = bathy.plot.imshow(**plot_parameters)
            label_parameters["mappable"] = pic
            label_parameters["ax"] = self.ax.flat[ax]
            label_parameters.setdefault("pad", 0.02)
            label_parameters.setdefault("extend", "neither")
            tick_parameters: dict[str, Any] = label_parameters.pop("tick_params", {})
            if not isinstance(tick_parameters, dict):
                raise TypeError(
                    f"'tick_parameters' in 'label_parameters' should be a dict, not a {type(tick_parameters).__name__}."
                )
            set_ylabel_parameters: dict[str, Any] = label_parameters.pop(
                "set_ylabel", {}
            )
            if not isinstance(set_ylabel_parameters, dict):
                raise TypeError(
                    f"'set_ylabel_parameters' in 'label_parameters' should be a dict, not a {type(set_ylabel_parameters).__name__}."
                )
            cbar: plt.Colorbar = plt.colorbar(**label_parameters)
            tick_parameters.setdefault("axis", "y")
            tick_parameters.setdefault("labelsize", 10)
            cbar.ax.tick_params(**tick_parameters)
            set_ylabel_parameters.setdefault("ylabel", "Height [m]")
            set_ylabel_parameters.setdefault("fontsize", 10)
            cbar.ax.set_ylabel(**set_ylabel_parameters)
        else:
            raise ValueError(f"{option} not a valid option!")

        self.ax.flat[ax].set_title(title)
        self.ax.flat[ax].set_xlabel(xlabel)
        self.ax.flat[ax].set_ylabel(ylabel)
        self.ax.flat[ax].set_aspect(self.aspect)

        if more_custom:
            if option == 0:
                return self, pic
            elif option == 2:
                return self, pic, cbar
            else:
                warnings.warn(
                    f"There is nothing more to return for further customization of option '{option}'."
                )

        return self

    def add_topography(
        self,
        option: int,
        color_contourlines: (
            str
            | tuple[float, float, float]
            | tuple[float, float, float, float]
            | LinearSegmentedColormap
        ),
        contour_params: float | npt.ArrayLike,
        ax: int = 0,
        path_mapdata: str = ...,
        plot_parameters: dict[str, Any] = None,
        label_parameters: dict[str, Any] = None,
        more_custom: bool = False,
    ) -> Self | tuple[Self, QuadContourSet, Colorbar | None]:
        """Function to plot either contour lines of the topography
        with the specified color or colored contours of the topography.
        Several options are available for the style of
        the topography contours (see below).

        Args:
            option (int): Switch to distinguish different styles. Valid options:
                0 : Topography as contour lines, low resolution.
                1 : Topography as contour lines, high resolution.
                2 : Topography as contour lines, costum .tif file.
                3 : Topography as filled contours, no colorbar, low resolution.
                4 : Topography as filled contours, no colorbar, high resolution.
                5 : Topography as filled contours, no colorbar, costum .tif file.
                6 : Topography as filled contours, with colorbar, low resolution.
                7 : Topography as filled contours, with colorbar, high resolution.
                8 : Topography as filled contours, with colorbar, costum .tif file.
            color_contourlines (str or RGB or RGBA or LinearSegmentedColormap): Color for the topography contour lines (only used with options 0 or 1)
            contour_params (float or array_like): At which levels contour levels shoud be.
                - single value : Resolution (distance between contour levels) of the bathymetry.
                - array_like : Contour levels of the bathymetry.
            ax (int, optional): Gives the position for which map the topography is added. Defaults to 0.
                - Starts to count from 0 and continues like the normal reading flow.
            path_mapdata (str, optional): Absolute or relative path to the directory including the map data. Defaults to not set.
                - Will be set as default for the map(s), if not set in the class initialisation.
            plot_parameters (dict[str, Any], optional):
                option:
                    0-2 : - For available options check xarray.DataArray.plot.contour.
                          - Always sets colors = color_contourlines, levels = contour_params.
                          - If not specified, linestyles = "-", linewidths = 0.5, is always set by default.
                    3-8 : - For available options check xarray.DataArray.plot.imshow.
                          - Always sets levels = contour_params.
                          - If not specified, cmap = cmocean.cm.turbid, interpolation = None, add_colorbar = False, is always set by default.
            label_parameters (dict[str, Any], optional):
                option:
                    0-2 : - For available options check cartopy.mpl.geoaxes.clabel.
                          - Always sets levels = contour_params.
                          - If not specified, inline = True, fmt = "%.0f", fontsize = 10 is always set by default.
                    3-8 : - For available options check matplotlib.pyplot.colorbar.
                          - If not specified, pad = 0.02, extend = "neither", tick_params = {}, set_ylabel = {}, is always set by default.
                          - To modify matplotlib.colorbar.Colorbar.ax.tick_params or .ax.set_ylabel, use:
                          - 'tick_params = {}' (default: axis = "y", labelsize = 10).
                          - 'set_ylabel = {}' (default: ylabel = "Height [m]", fontsize = 10).
            more_custom (bool, optional): If True, returns additional objects ('QuadContourSet' and 'Colorbar'). Defaults to False.

        Returns:
            MapGenerator or tuple[MapGenerator, QuadContourSet, Colorbar or None]:
                - If more_custom = False: Returns self.
                - If more_custom = True: Returns a tuple with extra objects.
        """
        if not isinstance(option, int):
            raise TypeError(f"'option' should be a int, not a {type(option).__name__}.")

        if not isinstance(
            color_contourlines, (str, tuple, mpl.colors.LinearSegmentedColormap)
        ):
            raise TypeError(
                f"'color' should be a str or tuple, not a {type(color_contourlines).__name__}."
            )

        if not isinstance(ax, int):
            raise TypeError(f"'ax' should be a int, not a {type(ax).__name__}.")
        if ax >= len(self.ax.flat):
            raise ValueError(
                f"Invalid axis index {ax}, max index is {len(self.ax.flat)-1}"
            )

        if path_mapdata == ...:
            if self.path_mapdata != ...:
                path_mapdata = self.path_mapdata
                if option % 3 == 2:
                    logging.warning(
                        f"Using 'path_mapdata' ({path_mapdata}) from the class. Be aware that you chose a option with a costum path."
                    )
            else:
                raise ValueError(f"'path_mapdata' needs to be set.")
        if not isinstance(path_mapdata, str):
            raise TypeError(
                f"'path_mapdata' should be a str, not a {type(path_mapdata).__name__}."
            )
        elif not path_mapdata == ... and option % 3 != 2:
            if not path_mapdata.endswith("/"):
                path_mapdata += "/"
            if not os.path.isdir(path_mapdata):
                raise NotADirectoryError(f"'{path_mapdata}' is not a valid directory.")
        if not path_mapdata == ... and self.path_mapdata == ... and option % 3 != 2:
            self.path_mapdata = path_mapdata
            logging.info(
                f"Changing the 'path_mapdata' for the object to '{path_mapdata}'. "
            )

        if plot_parameters == None:
            plot_parameters = {}
        if not isinstance(plot_parameters, dict):
            raise TypeError(
                f"'plot_parameters' should be a dictionary, not a {type(plot_parameters).__name__}."
            )

        if label_parameters == None:
            label_parameters = {}
        if not isinstance(label_parameters, dict):
            raise TypeError(
                f"'label_parameters' should be a dictionary, not a {type(label_parameters).__name__}."
            )

        if not isinstance(more_custom, bool):
            raise TypeError(
                f"'more_custom' should be a bool, not a {type(more_custom).__name__}."
            )

        title: str = self.ax.flat[ax].get_title()
        xlabel: str = self.ax.flat[ax].get_xlabel()
        ylabel: str = self.ax.flat[ax].get_ylabel()

        if option % 3 == 0:  # for 0,3,6
            dem = rxr.open_rasterio(
                f"{path_mapdata}NP_S0_DTM50/S0_DTM50.tif", masked=True
            ).squeeze()
        elif option % 3 == 1:  # for 1,4,7
            dem = rxr.open_rasterio(
                f"{path_mapdata}NP_S0_DTM20/S0_DTM20.tif", masked=True
            ).squeeze()
        elif option % 3 == 2:  # for 2,5,8
            if not os.path.isfile(path_mapdata):
                raise FileNotFoundError(f"File not found: {path_mapdata}")
            if not path_mapdata.endswith(".tif"):
                raise ValueError(
                    f"Invalid file format: {path_mapdata}. Expected a .tif file."
                )
            dem = rxr.open_rasterio(path_mapdata, masked=True).squeeze()
        else:
            raise ValueError(f"{option} not a valid option!")

        dem = dem.rio.reproject("EPSG:4326")
        dem = dem.rio.clip_box(
            minx=self.lon_limits[0],
            miny=self.lat_limits[0],
            maxx=self.lon_limits[1],
            maxy=self.lat_limits[1],
        )
        dem = dem.rio.reproject(ccrs.Mercator().proj4_init)

        if isinstance(contour_params, (float, int)):
            levels: np.ndarray[float] = np.arange(
                0,
                contour_params * np.ceil(np.nanmax(dem) / contour_params) + 1.0,
                contour_params,
            )
        elif pd.api.types.is_list_like(contour_params):
            if not all([isinstance(x, (int, float)) for x in contour_params]):
                raise ValueError(f"List of 'contour_params' shout just contain floats.")
            else:
                fail_values: list[float] = [
                    x
                    for x in contour_params
                    if not (x <= np.nanmax(dem) and x >= np.nanmin(dem))
                ]
                if len(fail_values) != 0:
                    raise ValueError(
                        f"The following 'contour_params' values are out of range: {fail_values}"
                    )
                levels = np.array(contour_params)
        else:
            raise TypeError(
                f"'contour_params' should be a tuple or a list, not a {type(contour_params).__name__}."
            )

        if (option == 0) | (option == 1) | (option == 2):
            dem = dem.where(dem >= 0.0)
            plot_parameters["ax"] = self.ax.flat[ax]
            plot_parameters["levels"] = levels
            plot_parameters["colors"] = color_contourlines
            plot_parameters.setdefault("linestyles", "-")
            plot_parameters.setdefault("linewidths", 0.5)
            pic = dem.plot.contour(**plot_parameters)
            label_parameters["CS"] = pic
            label_parameters["levels"] = pic.levels
            label_parameters.setdefault("inline", True)
            label_parameters.setdefault("fmt", "%.0f")
            label_parameters.setdefault("fontsize", 10)
            self.ax.flat[ax].clabel(**label_parameters)
        elif (option == 3) | (option == 4) | (option == 5):
            dem = dem.where(dem > 0.0)
            plot_parameters["ax"] = self.ax.flat[ax]
            plot_parameters["levels"] = levels
            plot_parameters.setdefault("cmap", cmo.cm.turbid)
            plot_parameters.setdefault("interpolation", None)
            plot_parameters.setdefault("add_colorbar", False)
            dem.plot.imshow(**plot_parameters)
        elif (option == 6) | (option == 7) | (option == 8):
            dem = dem.where(dem > 0.0)
            plot_parameters["ax"] = self.ax.flat[ax]
            plot_parameters["levels"] = levels
            plot_parameters.setdefault("cmap", cmo.cm.turbid)
            plot_parameters.setdefault("interpolation", None)
            plot_parameters.setdefault("add_colorbar", False)
            pic = dem.plot.imshow(**plot_parameters)
            label_parameters["mappable"] = pic
            label_parameters["ax"] = self.ax.flat[ax]
            label_parameters.setdefault("pad", 0.02)
            label_parameters.setdefault("extend", "neither")
            tick_parameters: dict[str, Any] = label_parameters.pop("tick_params", {})
            if not isinstance(tick_parameters, dict):
                raise TypeError(
                    f"'tick_parameters' in 'label_parameters' should be a dict, not a {type(tick_parameters).__name__}."
                )
            set_ylabel_parameters: dict[str, Any] = label_parameters.pop(
                "set_ylabel", {}
            )
            if not isinstance(set_ylabel_parameters, dict):
                raise TypeError(
                    f"'set_ylabel_parameters' in 'label_parameters' should be a dict, not a {type(set_ylabel_parameters).__name__}."
                )
            cbar: plt.Colorbar = plt.colorbar(**label_parameters)
            tick_parameters.setdefault("axis", "y")
            tick_parameters.setdefault("labelsize", 10)
            cbar.ax.tick_params(**tick_parameters)
            set_ylabel_parameters.setdefault("ylabel", "Height [m]")
            set_ylabel_parameters.setdefault("fontsize", 10)
            cbar.ax.set_ylabel(**set_ylabel_parameters)
        else:
            raise ValueError(f"{option} not a valid option!")

        self.ax.flat[ax].set_title(title)
        self.ax.flat[ax].set_xlabel(xlabel)
        self.ax.flat[ax].set_ylabel(ylabel)
        self.ax.flat[ax].set_aspect(self.aspect)

        if more_custom:
            if option % 3 == 0:
                return self, pic
            elif option % 3 == 2:
                return self, pic, cbar
            else:
                warnings.warn(
                    f"There is nothing more to return for further customization of option '{option}'."
                )

        return self

    def add_surface_cover(
        self,
        option: int,
        ax: int = 0,
        path_mapdata: str = ...,
        plot_parameters: dict[str, dict[str, Any]] = None,
    ) -> Self:
        """Function to colorize different types of surface cover on a previously created
        MapGenerator object. Several options are available for the resolution of the data used
        to define the surface cover areas (see below).

            Args:
                option (int): Switch to distinguish different resolutions. Valid options:
                    0 : Low resolution (1km) based on data from the Norwegian Polar Institute.
                        - Used properties: 'Land', 'Vann', 'Isbreer'
                    1 : Medium resolution (250 m) based on data from the Norwegian Polar Institute.
                        - Used properties: 'Land', 'Vann', 'Isbreer', 'Morener', 'TekniskSituasjon', 'Elvesletter'
                    2 : High resolution (100 m) based on data from the Norwegian Polar Institute.
                        - Used properties: 'Land', 'Vann', 'Isbreer', 'Morener', 'TekniskSituasjon', 'Elvesletter', 'Hav'
                ax (int, optional): Gives the position for which map the surface cover is added. Defaults to 0.
                    - Starts to count from 0 and continues like the normal reading flow.
                path_mapdata (str, optional): Absolute or relative path to the directory including the map data. Defaults to not set.
                    - Will be set as default for the map(s), if not set in the class initialisation.
                plot_parameters (dict[str, dict[str, Any]], optional): Parameters for plotting
                    - For available options check GeoAxes.plot.
                    - Provide parameters to .plot use: {'name_of_property':{'arg_name':arg_value}}.
                    - Property names: 'Land', 'Vann', 'Isbreer', 'Morener', 'TekniskSituasjon', 'Elvesletter', 'Hav'
                        - Translation: 'Land', 'Water', 'Glaciers', 'Morains', 'TechnicalSituation', 'Floodplains', 'Sea'
                        - Don't use the translation!
                        - To exclude a property, use: {'name_of_property':{"color":"none","facecolor":None}}.


            Returns:
                MapGenerator
        """

        if not isinstance(option, int):
            raise TypeError(f"'option' should be a int, not a {type(option).__name__}.")

        if not isinstance(ax, int):
            raise TypeError(f"'ax' should be a int, not a {type(ax).__name__}.")
        if ax >= len(self.ax.flat):
            raise ValueError(
                f"Invalid axis index {ax}, max index is {len(self.ax.flat)-1}"
            )

        if path_mapdata == ...:
            if self.path_mapdata != ...:
                path_mapdata = self.path_mapdata
            else:
                raise ValueError(f"'path_mapdata' needs to be set.")
        if not (isinstance(path_mapdata, str) or option == 0):
            raise TypeError(
                f"'path_mapdata' should be a str, not a {type(path_mapdata).__name__}."
            )
        elif not path_mapdata == ...:
            if not path_mapdata.endswith("/"):
                path_mapdata += "/"
            if not os.path.isdir(path_mapdata):
                raise NotADirectoryError(f"'{path_mapdata}' is not a valid directory.")
        if not path_mapdata == ... and self.path_mapdata == ...:
            self.path_mapdata = path_mapdata
            logging.info(
                f"Changing the 'path_mapdata' for the object to '{path_mapdata}'. "
            )

        if plot_parameters == None:
            plot_parameters = {}
        if not isinstance(plot_parameters, dict):
            raise TypeError(
                f"'plot_parameters' should be a dictionary, not a {type(plot_parameters).__name__}."
            )

        colors: dict[str, str] = {
            "Elvesletter": "#CCBDA9",
            "Hav": "#FFFFFF",
            "Isbreer": "#B3FFFF",
            "Land": "#E7D3B8",
            "Morener": "#CED8D9",
            "TekniskSituasjon": "#FF8080",
            "Vann": "#99CAEB",
        }

        if option == 0:
            layers: list[str] = ["Land", "Vann", "Isbreer"]
            res = "1000"
        elif option == 1:
            layers = [
                "Land",
                "Vann",
                "Elvesletter",
                "Isbreer",
                "Morener",
                "TekniskSituasjon",
            ]
            res = "250"
        elif option == 2:
            layers = [
                "Land",
                "Vann",
                "Elvesletter",
                "Isbreer",
                "Morener",
                "TekniskSituasjon",
            ]
            res = "100"
        else:
            raise ValueError(f"{option} not a valid option!")

        title: str = self.ax.flat[ax].get_title()
        xlabel: str = self.ax.flat[ax].get_xlabel()
        ylabel: str = self.ax.flat[ax].get_ylabel()

        self.ax.flat[ax].set_facecolor("#FFFFFF")
        for layer in layers:
            input_file: str = f"{path_mapdata}NP_S{res}_SHP/S{res}_{layer}_f.shp"
            with warnings.catch_warnings():
                warnings.filterwarnings(
                    "ignore",
                    message="Non closed ring detected. To avoid accepting it, set the OGR_GEOMETRY_ACCEPT_UNCLOSED_RING configuration option to NO",
                    category=RuntimeWarning,
                )
                try:
                    df_layer = gpd.read_file(input_file)
                except UnicodeDecodeError:
                    df_layer = gpd.read_file(input_file, encoding="Windows-1252")
            try:
                df_layer = df_layer.to_crs(ccrs.Mercator().proj4_init)
            except GEOSException:
                df_layer = (
                    df_layer["geometry"]
                    .buffer(1e-100)
                    .to_crs(ccrs.Mercator().proj4_init)
                )
            layer_plot_parameters: dict[str, Any] = plot_parameters.pop(layer, {})
            if not isinstance(layer_plot_parameters, dict):
                raise TypeError(
                    f"In 'plot_parameters' the key '{layer}' as an invailed value. The value should be a dictionary, not a {type(plot_parameters).__name__}."
                )
            layer_plot_parameters["ax"] = self.ax.flat[ax]
            layer_plot_parameters.setdefault("edgecolor", None)
            layer_plot_parameters.setdefault("facecolor", colors[layer])
            df_layer.plot(**layer_plot_parameters)
            self.ax.flat[ax].set_extent(
                self.lon_limits + self.lat_limits, crs=ccrs.PlateCarree()
            )

        self.ax.flat[ax].set_title(title)
        self.ax.flat[ax].set_xlabel(xlabel)
        self.ax.flat[ax].set_ylabel(ylabel)
        self.ax.flat[ax].set_aspect(self.aspect)

        return self

    def add_crosssection_line(
        self,
        lat: npt.ArrayLike,
        lon: npt.ArrayLike,
        color: (
            str | tuple[float, float, float] | tuple[float, float, float, float]
        ) = "k",
        ax: int = 0,
        plot_parameters: dict[str, Any] = None,
    ) -> Self:
        """Function to add a line indicating the position of a cross section
        on a previously created MapGenerator object.

        Args:
            lat (array_like): List with the latitude coordinates.
                - [lat_0, lat_1, ...]
            lon (array_like): List with the lonitute coordinates.
                - [lon_0, lon_1, ...]
            color (str or RGB or RGBA, optional): Color for the cross section line. Defaults to "k".
            ax (int, optional): Gives the position for which map the crosssection line is added. Defaults to 0.
                    - Starts to count from 0 and continues like the normal reading flow.
            plot_parameters (dict[str,Any], optional): Additional parameters passed to geopandas.GeoDataFrame.plot.
                - For available options check matplotlib.pyplot.subplots.
                - Always sets color = color.

        Returns:
            MapGenerator
        """

        if not pd.api.types.is_list_like(lon):
            raise TypeError(
                f"'lon' should be a array_like, not a {type(lon).__name__}."
            )
        if not all(isinstance(x, (int, float)) for x in lon):
            raise TypeError(f"'lon' should contain just float.")
        if not pd.api.types.is_list_like(lat):
            raise TypeError(
                f"'lat' should be a array_like, not a {type(lat).__name__}."
            )
        if not all(isinstance(x, (int, float)) for x in lat):
            raise TypeError(f"'lat' should contain just float.")
        if len(lat) != len(lon):
            raise ValueError(f"'lon' and 'lat' should have the same lenght.")

        if not isinstance(color, (str, tuple)):
            raise TypeError(
                f"'color' should be a str or tuple, not a {type(color).__name__}."
            )

        if not isinstance(ax, int):
            raise TypeError(f"'ax' should be a int, not a {type(ax).__name__}.")
        if ax >= len(self.ax.flat):
            raise ValueError(
                f"Invalid axis index {ax}, max index is {len(self.ax.flat)-1}"
            )

        if plot_parameters == None:
            plot_parameters = {}
        if not isinstance(plot_parameters, dict):
            raise TypeError(
                f"'plot_parameters' should be a dictionary, not a {type(plot_parameters).__name__}."
            )

        title: str = self.ax.flat[ax].get_title()
        xlabel: str = self.ax.flat[ax].get_xlabel()
        ylabel: str = self.ax.flat[ax].get_ylabel()

        df: pd.DataFrame = pd.DataFrame(
            {"latitude": lat, "longitude": lon, "section": 1}
        )
        gdf = gpd.GeoDataFrame(
            df, geometry=gpd.points_from_xy(df.longitude, df.latitude)
        )
        gdf = gdf.groupby(["section"])["geometry"].apply(
            lambda x: LineString(x.tolist())
        )
        gdf = gpd.GeoDataFrame(gdf, geometry="geometry", crs="EPSG:4326")
        gdf: gpd.GeoDataFrame = gdf.to_crs(ccrs.Mercator().proj4_init)
        plot_parameters["ax"] = self.ax.flat[ax]
        plot_parameters["color"] = color
        gdf.plot(**plot_parameters)

        self.ax.flat[ax].set_title(title)
        self.ax.flat[ax].set_xlabel(xlabel)
        self.ax.flat[ax].set_ylabel(ylabel)
        self.ax.flat[ax].set_aspect(self.aspect)

        return self

    def add_points(
        self,
        lat: npt.ArrayLike,
        lon: npt.ArrayLike,
        color: str | tuple[float, float, float] | tuple[float, float, float, float],
        size: float | npt.ArrayLike,
        ax: int = 0,
        cbar_label: str = ...,
        point_label_title: str = ...,
        point_labels: list[str] = ...,
        plot_parameters: dict[str, Any] = None,
        label_parameters: dict[str, Any] = None,
        point_label_parameters: dict[str, Any | dict[str, Any]] = None,
        more_custom: bool = False,
    ) -> Self | tuple[Self, QuadContourSet, Colorbar | None]:
        """Function to add a set of points on a previously created MapGenerator object.
        The points can either all be in the same color and size, e.g. to indicate the locations
        of instruments, or their color and/or size can be set according to data,
        e.g. colorcode the temperature at different locations.

        Args:
            lat (array_like): List with the latitude coordinates.
                - [lat_0, lat_1, ...]
            lon (array_like): List with the lonitude coordinates.
                - [lon_0, lon_1, ...]
            color (str or RGB or RGBA, optional):
                Color for the points. Either one fixed color for all, fixed individual colors for all or data that is going to be
                represented by the coloring.
            size (float or array_like):
                Size of the points. Either one fixed value or data that is going to be
                represented by the size.
            ax (int, optional): Gives the position for which map the crosssection line is added. Defaults to 0.
                    - Starts to count from 0 and continues like the normal reading flow.
            cbar_label (str, optional): Label for the colorbar. Defaults to ....
                - Has to be set to show a colorbar.
            point_label_title (str, optional): Title for the legend. Defaults to ....
                - Has to be set to show automatically show a legend.
                - The legend displayes the scale of the dots.
            point_labels (list[str], optional): List of names for the different legend entries. Default to ....
            plot_parameters (dict[str,Any], optional): Additional parameters passed to GeoAxes.scatter.
                - For available options check GeoAxes.scatter.
                - Always sets color = color, size = size, transform = ccrs.PlateCarree().
                - If not specified, zorder = 10, cmap = cmo.cm.theramal is always set by default.
            label_parameters (dict[str, Any], optional):
                - For available options check matplotlib.pyplot.colorbar.
                            - If not specified, pad = 0.1, orientation = "horizontal", tick_params = {}, set_ylabel = {}, is always set by default.
                            - To modify matplotlib.colorbar.Colorbar.ax.tick_params or .ax.set_xlabel, use:
                                - 'tick_params = {}' (default: axis = "x", labelsize = 10).
                                - 'set_ylabel = {}' (default: xlabel = cbar_label, fontsize = 10).
            point_label_parameters (dict[str, Any], optional):
                        - For available options check matplotlib.collections.PathCollection.legend_elements.
                        - If not specified, prop = "sizes", color = "black", legend_parameters = {} is always set by default.
                        - Sets num = 3 if not specified and 'sizes' is not in 'point_label_parameters'.
                        - To modify GeoAxes.legend.
                            - 'legend_parameters = {}' (default: axis = "y", labels = point_labels|labels (labels is generated by matplotlib.collections.PathCollection.legend_elements)).
            more_custom (bool, optional): If True, returns additional objects ('matplotlib.collections.PathCollection' and 'Colorbar'). Defaults to False.

        Returns:
            MapGenerator or tuple[MapGenerator, matplotlib.collections.PathCollection, Colorbar or None]:
                - If more_custom = False: Returns self.
                - If more_custom = True: Returns a tuple with extra objects.
        """

        if not pd.api.types.is_list_like(lon):
            raise TypeError(
                f"'lon' should be a array_like, not a {type(lon).__name__}."
            )
        if not all(isinstance(x, (int, float)) for x in lon):
            raise TypeError(f"'lon' should contain just float.")
        if not pd.api.types.is_list_like(lat):
            raise TypeError(
                f"'lat' should be a array_like, not a {type(lat).__name__}."
            )
        if not all(isinstance(x, (int, float)) for x in lat):
            raise TypeError(f"'lat' should contain just float.")
        if len(lat) != len(lon):
            raise ValueError(f"'lon' and 'lat' should have the same lenght.")

        if not (isinstance(color, str) or pd.api.types.is_list_like(color)):
            raise TypeError(
                f"'color' should be a str or array_like, not a {type(color).__name__}."
            )
        else:
            single_color = True
            color_values = False
        if pd.api.types.is_list_like(color):
            single_color: bool = False  # to see if it is a RGB or RGBA
            if len(color) == 3 or len(color) == 4:
                if not any(
                    [
                        (pd.api.types.is_list_like(x) or isinstance(x, str))
                        for x in color
                    ]
                ):
                    if all([(x <= 1 and x >= 0) for x in color]):
                        single_color = True
            elif len(color) != len(lat) and not single_color:
                raise ValueError(
                    f"'color' has not enough values for every point. Needs {len(lat)}, but has {len(color)}."
                )
            elif all([isinstance(x, (float, int)) for x in color]):
                color_values: bool = True

        if not (isinstance(size, (float, int)) or pd.api.types.is_list_like(size)):
            raise TypeError(
                f"'size' should be a float or array_like, not a {type(color).__name__}."
            )
        single_size: bool = True
        if pd.api.types.is_list_like(size):
            if len(size) != len(lat):
                raise ValueError(
                    f"'size' has not enough values for every point. Needs {len(lat)}, but has {len(size)}."
                )
            else:
                single_size = False

        if not isinstance(ax, int):
            raise TypeError(f"'ax' should be a int, not a {type(ax).__name__}.")
        if ax >= len(self.ax.flat):
            raise ValueError(
                f"Invalid axis index {ax}, max index is {len(self.ax.flat)-1}"
            )

        if not isinstance(cbar_label, str) and not cbar_label == ...:
            raise TypeError(
                f"'cbar_label' should be a str, not a {type(cbar_label).__name__}."
            )

        if not isinstance(point_label_title, str) and not point_label_title == ...:
            raise TypeError(
                f"'point_label' should be a str, not a {type(point_label_title).__name__}."
            )

        if plot_parameters == None:
            plot_parameters = {}
        if not isinstance(plot_parameters, dict):
            raise TypeError(
                f"'plot_parameters' should be a dictionary, not a {type(plot_parameters).__name__}."
            )

        if label_parameters == None:
            label_parameters = {}
        if not isinstance(label_parameters, dict):
            raise TypeError(
                f"'label_parameters' should be a dictionary, not a {type(label_parameters).__name__}."
            )

        if point_label_parameters == None:
            point_label_parameters = {}
        if not isinstance(point_label_parameters, dict):
            raise TypeError(
                f"'point_label_parameters' should be a dictionary, not a {type(point_label_parameters).__name__}."
            )

        if point_labels != ...:
            if not pd.api.types.is_list_like(point_labels):
                raise TypeError(f"'point_labels' should be a list.")
            if "size" in point_label_parameters.keys():
                if not pd.api.types.is_list_like(point_label_parameters["size"]):
                    raise TypeError(
                        f"'size' in 'point_label_parameters' should be array_like."
                    )
                else:
                    len_legend: int = len(point_label_parameters["size"])
            elif "num" in point_label_parameters.keys():
                if not isinstance(point_label_parameters["num"]):
                    raise TypeError(
                        f"'num' in 'point_label_parameters' should be a int, not a {type(point_label_parameters['num']).__name__}."
                    )
                else:
                    len_legend = point_label_parameters["num"]
            else:
                len_legend = 3
            if len(point_labels) != len_legend:
                raise ValueError(
                    f"'point_labels' should have a length of {len_legend}, but has {len(point_labels)}."
                )

        if single_size and single_color:
            color = [color] * len(lat)
            size = [size] * len(lat)
        elif single_color and not single_size:
            color = [color] * len(lat)
        elif not single_color and single_size:
            size = [size] * len(lat)
        elif not (single_color and single_size):
            pass
        else:
            raise ValueError(f"Something went wrong, this shouldn't happen :(.")
        df = pd.DataFrame(
            {"latitude": lat, "longitude": lon, "color": color, "size": size}
        )

        gdf = gpd.GeoDataFrame(
            df, geometry=gpd.points_from_xy(df.longitude, df.latitude), crs="EPSG:4326"
        )
        gdf: gpd.GeoDataFrame = gdf.to_crs(ccrs.Mercator().proj4_init)

        title: str = self.ax.flat[ax].get_title()
        xlabel: str = self.ax.flat[ax].get_xlabel()
        ylabel: str = self.ax.flat[ax].get_ylabel()

        plot_parameters["x"] = gdf["longitude"]
        plot_parameters["y"] = gdf["latitude"]
        plot_parameters["c"] = gdf["color"]
        plot_parameters["s"] = gdf["size"]
        plot_parameters["transform"] = ccrs.PlateCarree()
        plot_parameters.setdefault("zorder", 10)
        if color_values:
            plot_parameters.setdefault("cmap", cmo.cm.thermal)
        pic = self.ax.flat[ax].scatter(**plot_parameters)

        if not single_color and color_values and cbar_label != ...:
            label_parameters["mappable"] = pic
            label_parameters["ax"] = self.ax.flat[ax]
            label_parameters.setdefault("pad", 0.1)
            label_parameters.setdefault("orientation", "horizontal")
            tick_params: dict[str, Any] = label_parameters.pop("tick_params", {})
            if not isinstance(tick_params, dict):
                raise TypeError(
                    f"'tick_params' in 'label_parameters' should be a dict, not a {type(tick_params).__name__}."
                )
            set_xlabel_parameters: dict[str, Any] = label_parameters.pop(
                "set_xlabel", {}
            )
            if not isinstance(set_xlabel_parameters, dict):
                raise TypeError(
                    f"'set_xlabel_parameters' in 'label_parameters' should be a dict, not a {type(set_ylabel_parameters).__name__}."
                )
            cbar: Colorbar = plt.colorbar(**label_parameters)
            tick_params.setdefault("axis", "x")
            tick_params.setdefault("labelsize", 10)
            cbar.ax.tick_params(**tick_params)
            set_xlabel_parameters.setdefault("xlabel", cbar_label)
            set_xlabel_parameters.setdefault("fontsize", 10)
            cbar.ax.set_xlabel(**set_xlabel_parameters)
        if point_label_title != ...:
            legend_parameters: dict[str, Any] = point_label_parameters.pop(
                "legend_parameters", {}
            )
            if not isinstance(legend_parameters, dict):
                raise TypeError(
                    f"'legend_parameters' in 'point_label_parameters' should be a dict, not a {type(legend_parameters).__name__}."
                )
            point_label_parameters.setdefault("prop", "sizes")
            point_label_parameters.setdefault("color", "black")
            if not "sizes" in point_label_parameters.keys():
                point_label_parameters.setdefault("num", 3)
            else:
                point_label_parameters.setdefault("num", None)
            handles, labels = pic.legend_elements(**point_label_parameters)
            legend_parameters["handles"] = handles
            legend_parameters["title"] = point_label_title
            if point_labels != ...:
                legend_parameters["labels"] = point_labels
            legend_parameters.setdefault("labels", labels)
            self.ax.flat[ax].legend(**legend_parameters)

        self.ax.flat[ax].set_title(title)
        self.ax.flat[ax].set_xlabel(xlabel)
        self.ax.flat[ax].set_ylabel(ylabel)
        self.ax.flat[ax].set_aspect(self.aspect)

        if more_custom:
            if not single_color and color_values and cbar_label != ...:
                return self, pic, cbar
            else:
                return self, pic

        return self

    def add_wind_arrows(
        self,
        lat: npt.ArrayLike,
        lon: npt.ArrayLike,
        u: list[float],
        v: list[float],
        length: float = 10,
        lw: float = 1,
        ax: int = 0,
        plot_parameters: dict[str, Any] = None,
    ) -> Self:
        """Function to add meteorological wind arrows on a previously created MapGenerator object.
        The scaling of the arrows follows the meteorological convention:
        small bar = 5 knots
        large bar = 10 knots
        tiangle = 50 knots

        Args:
            lat (array_like): List with the latitude coordinates.
                - [lat_0, lat_1, ...]
            lon (array_like): List with the lonitude coordinates.
                - [lon_0, lon_1, ...]
            u (list[float]): List or array with the x component of the winds.
            v (list[float]): List or array with the y component of the winds.
            length (float, optional): Reference for the length of the arrows. Defaults to 10.
            lw (float, optional): Line thickness of the arrows. Defaults to 1.
            ax (int, optional): Gives the position for which map the bathymetry is added. Defaults to 0.
                - Starts to count from 0 and continues like the normal reading flow.
            plot_parameters (dict[str, Any], optional):
                - For available options check cartopy.mpl.geoaxes.barbs.
                - Always sets x = lat, y = lon, u = u, v = v, lenght = length, linewidth = lw.

        Returns:
            MapGenerator
        """

        if not pd.api.types.is_list_like(lon):
            raise TypeError(f"'lon' should be array_like, not a {type(lon).__name__}.")
        if not all(isinstance(x, (int, float)) for x in lon):
            raise TypeError(f"'lon' should contain just float.")
        if not pd.api.types.is_list_like(lat):
            raise TypeError(f"'lat' should be array_like, not a {type(lat).__name__}.")
        if not all(isinstance(x, (int, float)) for x in lat):
            raise TypeError(f"'lat' should contain just float.")
        if len(lat) != len(lon):
            raise TypeError(f"'lon' and 'lat' should have the same lenght.")

        if not pd.api.types.is_list_like(u):
            raise TypeError(f"'u' should be a tuple or list, not a {type(u).__name__}.")
        if not all(isinstance(x, (int, float)) for x in u):
            raise TypeError(f"'u' should contain just float.")
        if len(u) != len(lat):
            raise ValueError(f"'u' should have a length of {lat}, but has {len(u)}.")

        if not pd.api.types.is_list_like(v):
            raise TypeError(f"'v' should be a tuple or list, not a {type(v).__name__}.")
        if not all(isinstance(x, (int, float)) for x in v):
            raise TypeError(f"'v' should contain just float.")
        if len(v) != len(lat):
            raise ValueError(f"'v' should have a length of {lat}, but has {len(v)}.")

        if not isinstance(length, (float, int)):
            raise TypeError(
                f"'lenght' should be a float, not a {type(length).__name__}."
            )

        if not isinstance(lw, (float, int)):
            raise TypeError(f"'lw' should be a float, not a {type(lw).__name__}.")

        if not isinstance(ax, int):
            raise TypeError(f"'ax' should be a int, not a {type(ax).__name__}.")
        if ax >= len(self.ax.flat):
            raise ValueError(
                f"Invalid axis index {ax}, max index is {len(self.ax.flat)-1}"
            )

        if plot_parameters == None:
            plot_parameters = {}
        if not isinstance(plot_parameters, dict):
            raise TypeError(
                f"'plot_parameters' should be a dictionary, not a {type(plot_parameters).__name__}."
            )

        u = np.array(u) * 1.94384
        v = np.array(v) * 1.94384

        df = pd.DataFrame({"latitude": lat, "longitude": lon, "u": u, "v": v})
        gdf = gpd.GeoDataFrame(
            df, geometry=gpd.points_from_xy(df.longitude, df.latitude), crs="EPSG:4326"
        )
        gdf: gpd.GeoDataFrame = gdf.to_crs(ccrs.Mercator().proj4_init)

        title: str = self.ax.flat[ax].get_title()
        xlabel: str = self.ax.flat[ax].get_xlabel()
        ylabel: str = self.ax.flat[ax].get_ylabel()

        plot_parameters["x"] = gdf["geometry"].x
        plot_parameters["y"] = gdf["geometry"].y
        plot_parameters["u"] = gdf["u"]
        plot_parameters["v"] = gdf["v"]
        plot_parameters["length"] = length
        plot_parameters["linewidth"] = lw
        self.ax.flat[ax].barbs(**plot_parameters)

        self.ax.flat[ax].set_title(title)
        self.ax.flat[ax].set_xlabel(xlabel)
        self.ax.flat[ax].set_ylabel(ylabel)
        self.ax.flat[ax].set_aspect(self.aspect)

        return self

    def get_ax(self) -> np.ndarray[GeoAxes]:
        """Returns an array of the axis/axes.

        Returns:
            np.ndarray[GeoAxes]: An array of the axis/axes.
        """
        return self.ax

    def get_fig(self) -> Figure:
        """Returns the mathplotlib.figure.Figure

        Returns:
            mathplotlib.figure.Figure: Figure of the plt.subplot.
        """
        return self.fig
