# -*- coding: utf-8 -*-
"""
This module contains functions that read (and plot) various
oceanographic instrument data. This includes:
- CTD
- ADCP
- mooring

The functions were oroginally developed at the Geophysical Institute at the
University of Bergen, Norway. They were adapted for the file formats typically used
in student cruises at UNIS.
"""

from __future__ import print_function, annotations

from numpy._typing._array_like import NDArray
import universal_func as uf
from seabird.cnv import fCNV
import gsw
import numpy as np
import numpy.typing as npt
import matplotlib.pyplot as plt
import matplotlib
from netCDF4 import Dataset
import glob
from scipy.interpolate import interp1d, griddata
import scipy.io as spio
from scipy.io import loadmat
from matplotlib.dates import date2num, datestr2num
import cmocean
import cartopy.crs as ccrs
import cartopy.feature
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
import pandas as pd
from adjustText import adjust_text as adj_txt
from pyrsktools import RSK
import xarray as xr
import datetime
import os
import plotly.express as px
from plotly.offline import plot as pplot

# from mpl_toolkits.axes_grid1.inset_locator import InsetPosition will look into that later
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import uptide
import spectrum
from scipy import signal

import re
import pathlib
import zipfile
import posixpath
import pyTMD.utilities
import pyTMD.compute

import numbers as num
import time
from typing import Literal, Any, get_args
import logging
import warnings
import sounddevice as sd


############################################################################
# MISCELLANEOUS FUNCTIONS
############################################################################
def cal_dist_dir_on_sphere(
    longitude: pd.Series, latitude: pd.Series
) -> tuple[pd.Series, pd.Series]:
    """Function to calculate a series of distances between
    coordinate points (longitude and latitude)
    of the drifter between sequential timesteps.

    Args:
        longitude (pd.Series): Time Series of logitudinal coordinates [deg] of the ship.
        latitude (pd.Series): Time Series of latitudinal coordinates [deg] of the ship.

    Returns:
        tuple(pd.Series, pd.Series):
            - Speed the drifter travelled between each of the timesteps
            - Direction drifter headed between each of the timesteps
    """

    if not isinstance(longitude, pd.Series):
        raise TypeError(
            f"'longnitude' should be a pd.Series, not a {type(longitude).__name__}."
        )
    if not isinstance(latitude, pd.Series):
        raise TypeError(
            f"'longnitude' should be a pd.Series, not a {type(latitude).__name__}."
        )

    # Define the Earths Radius (needed to estimate distance on Earth's sphere)
    R: float = 6378137.0  # [m]

    # Convert latitude and logitude to radians
    lon: pd.Series = longitude * np.pi / 180.0
    lat: pd.Series = latitude * np.pi / 180.0

    # Calculate the differential of lon and lat between the timestamps
    dlon: pd.Series = lon.diff()
    dlat: pd.Series = lat.diff()

    # Create a shifted time Series
    lat_t1: pd.Series = lat.shift(periods=1)
    lat_t2: pd.Series = lat.copy()

    # Calculate interim stage
    alpha: pd.Series = (
        np.sin(dlat / 2.0) ** 2
        + np.cos(lat_t1) * np.cos(lat_t2) * np.sin(dlon / 2.0) ** 2
    )

    distance: pd.Series = (
        2 * R * np.arctan2(np.sqrt(alpha), np.sqrt(1 - alpha))
    )  # (np.arcsin(np.sqrt(alpha))

    time_delta: pd.Series = pd.Series(
        (lat.index[1:] - lat.index[0:-1]).seconds, index=lat.index[1::]
    )
    speed: pd.Series = distance / time_delta

    # Calculate the ships heading
    arg1: pd.Series = np.sin(dlon) * np.cos(lat_t2)
    arg2: pd.Series = np.cos(lat_t1) * np.sin(lat_t2) - np.sin(lat_t1) * np.cos(
        lat_t2
    ) * np.cos(dlon)

    heading: pd.Series = np.arctan2(arg1, arg2) * (-180.0 / np.pi) + 90.0
    heading[heading < 0.0] = heading + 360.0
    heading[heading > 360.0] = heading - 360.0

    return speed, heading


__ctype__ = Literal["math", "ocean", "meteo"]


def cart2pol(
    u: num.Real | npt.ArrayLike,
    v: num.Real | npt.ArrayLike,
    ctype: __ctype__ = "math",
) -> tuple[npt.ArrayLike | num.Real]:
    """Converts cartesian velocity (u,v) to polar velocity (angle,speed),
    using either mathematical, oceanographical or meteorological
    definition.

    Args:
        u (numeric or array_like): u-Component of velocity.
        v (numeric or array_like): v-Component of velocity.
        ctype (str, optional): Type of definitition, 'math', 'ocean' or 'meteo'. Defaults to "math".

    Returns:
        tuple[npt.ArrayLike | numeric]:
            - Angle of polar velocity.
            - Speed of polar velocity.
    """

    if not (pd.api.types.is_array_like(u) or isinstance(u, num.Real)):
        raise TypeError(
            f"'u' should be numeric or array_like, not a {type(u).__name__}."
        )
    if not (pd.api.types.is_array_like(v) or isinstance(v, num.Real)):
        raise TypeError(
            f"'u' should be numeric or array_like, not a {type(v).__name__}."
        )
    if ctype not in get_args(__ctype__):
        raise ValueError(f"'ctype' should be 'math', 'ocean' or 'meteo', not {ctype}.")

    speed: num.Number | npt.ArrayLike = np.sqrt(u**2 + v**2)
    if ctype == "math":
        angle: num.Number | npt.ArrayLike = 180 / np.pi * np.arctan2(v, u)
    if ctype in ["meteo", "ocean"]:
        angle = 180 / np.pi * np.arctan2(u, v)
        if ctype == "meteo":
            angle = (angle + 180) % 360

    return angle, speed


def pol2cart(
    angle: num.Real | npt.ArrayLike,
    speed: num.Real | npt.ArrayLike,
    ctype: __ctype__ = "math",
) -> tuple[npt.ArrayLike | num.Real]:
    """Converts polar velocity (angle,speed) to cartesian velocity (u,v),
    using either mathematical, oceanographical or meteorological
    definition.

    Args:
        angle (numeric | npt.ArrayLike): Angle of polar velocity.
        speed (numeric | npt.ArrayLike): Speed of polar velocity.
        ctype (__ctype__, optional): Type of definitition, 'math', 'ocean' or 'meteo'. Defaults to "math".

    Returns:
        tuple[npt.ArrayLike | numeric]:
            - u-component of velocity.
            - v-component of velocity.
    """

    if not (pd.api.types.is_array_like(angle) or isinstance(angle, num.Real)):
        raise TypeError(
            f"'u' should be numeric or array_like, not a {type(angle).__name__}."
        )
    if not (pd.api.types.is_array_like(speed) or isinstance(speed, num.Real)):
        raise TypeError(
            f"'u' should be numeric or array_like, not a {type(speed).__name__}."
        )
    if ctype not in get_args(__ctype__):
        raise ValueError(f"'ctype' should be 'math', 'ocean' or 'meteo', not {ctype}.")

    if ctype == "math":
        u: num.Number | npt.ArrayLike = speed * np.cos(angle * np.pi / 180.0)
        v: num.Number | npt.ArrayLike = speed * np.sin(angle * np.pi / 180.0)
    elif ctype == "meteo":
        u = -speed * np.sin(angle * np.pi / 180.0)
        v = -speed * np.cos(angle * np.pi / 180.0)
    elif ctype == "ocean":
        u = speed * np.sin(angle * np.pi / 180.0)
        v = speed * np.cos(angle * np.pi / 180.0)

    return u, v


def create_latlon_text(lat: num.Real, lon: num.Real) -> tuple[str, str]:
    """Creates two strings which contain a text for latitude and longitude.

    Args:
        lat (scalar): Latitude value.
        lon (scalar): Longitude value.

    Returns:
        tuple[str, str]:
            - The string for the latitude.
            - The string for the longitude.
    """

    lat_minutes: str = str(np.round((np.abs(lat - int(lat))) * 60, 5))
    if lat < 0:
        lat_letter: str = "S"
    else:
        lat_letter = "N"
    latstring: str = str(int(np.abs(lat))) + " " + lat_minutes + " " + lat_letter

    lon_minutes: str = str(np.round((np.abs(lon - int(lon))) * 60, 5))
    if lon < 0:
        lon_letter: str = "W"
    else:
        lon_letter = "E"
    lonstring: str = str(int(np.abs(lon))) + " " + lon_minutes + " " + lon_letter

    return latstring, lonstring


__x_type__ = Literal["time", "distance"]


def CTD_to_grid(
    CTD: dict[dict],
    stations: npt.ArrayLike = None,
    interp_opt: int = 1,
    x_type: __x_type__ = "distance",
    z_fine: bool = False,
) -> tuple[dict, np.ndarray, np.ndarray, np.ndarray]:
    """This function accepts a CTD dict of dicts, finds out the maximum
    length of the depth vectors for the given stations, and fills all
    fields to that maximum length, using np.nan values.

    Args:
        CTD (dict[dict]): CTD data.
            - Is created by 'read_CTD'.
        stations (array_like, optional): List of stations to select from CTD. Defaults to None.
            - If set to None all stations are used.
        interp_opt (int, optional): flag how to interpolate over X. Defaults to 1.
            0 : no interpolation.
            1 : linear interpolation, fine grid.
            2 : linear interpolation, coarse grid.
        x_type (str, optional): Whether X is 'time' or 'distance'. Defaults to "distance".
        z_fine (bool, optional): Whether to use a fine z grid. Defaults to False.
            - If True, will be 10 cm, otherwise 1 m.

    Returns:
        tuple[dict, np.ndarray, np.ndarray, np.ndarray]:
            - Dict with the gridded CTD data.
            - Common depth vector.
            - Common X vector.
            - Locations of the stations as X units.
    """

    # if no stations are given, take all stations available
    if not isinstance(CTD, dict):
        raise TypeError(f"'CTD' should be a dict, not a {type(CTD).__name__}")
    if stations is None:
        stations = list(CTD.keys())
    else:
        if not pd.api.types.is_list_like(stations):
            raise TypeError(
                f"'stations' should be a array_like, not a {type(stations).__name__}."
            )
        if type(stations) != list:
            stations = [x for x in stations]
        notfound_stations: list = [
            key for key in stations if not key in list(CTD.keys())
        ]
        if len(notfound_stations) != 0:
            logging.info(
                f"The stations '{notfound_stations}' are not in the CTD data. Proceeding without them."
            )
            for i in notfound_stations:
                stations.remove(i)
            if len(stations) == 0:
                raise ValueError(f"There are no CTD stations left.")
        CTD = {key: CTD[key] for key in stations}
    if len(stations) == 0:
        raise ValueError(f"The CTD is empty.")

    if not isinstance(interp_opt, int):
        raise TypeError(
            f"'interp_opt' should be a int, not a {type(interp_opt).__name__}."
        )
    if interp_opt < 0 or interp_opt > 2:
        raise ValueError(f"'interp_opt' should be a 0,1,2, not {interp_opt}.")

    if x_type not in get_args(__x_type__):
        raise ValueError(f"'ctype' should be 'time' or 'distance', not {x_type}.")

    if not isinstance(z_fine, bool):
        raise TypeError(f"'z_fine' should be a bool, not a {type(z_fine).__name__}.")

    # construct the Z-vector from the max and min depth of the given stations
    maxdepth: float = np.nanmax([np.nanmax(-CTD[i]["z"]) for i in stations])
    mindepth: float = np.nanmin([np.nanmin(-CTD[i]["z"]) for i in stations])
    if z_fine:
        Z: np.ndarray = np.linspace(
            mindepth, maxdepth, int((maxdepth - mindepth) * 10) + 1
        )
    else:
        Z = np.linspace(mindepth, maxdepth, int(maxdepth - mindepth) + 1)

    # construct the X-vector, either distance or time
    if x_type == "distance":
        LAT: np.ndarray = np.asarray([d["LAT"] for d in CTD.values()])
        LON: np.ndarray = np.asarray([d["LON"] for d in CTD.values()])
        X: np.ndarray = np.insert(np.cumsum(gsw.distance(LON, LAT) / 1000), 0, 0)
    elif x_type == "time":
        X = np.array([date2num(d["datetime"]) for d in CTD.values()])
        X = (X - X[0]) * 24

    # this X vector is where the stations are located, so save that
    station_locs: np.ndarray = X[:]
    fields: list = list(
        set(
            [
                field
                for field in CTD[stations[0]]
                if np.size(CTD[stations[0]][field]) > 1
            ]
        )
    )

    # original grids
    X_orig, Z_orig = [f.ravel() for f in np.meshgrid(X, Z)]
    # new grids in case of 2-d interpolation
    if interp_opt == 1:
        X_int: np.ndarray = np.linspace(
            np.min(X), np.max(X), len(X) * 20
        )  # create fine X grid
        Z_int: np.ndarray = Z[:]
    elif interp_opt == 2:
        X_int = np.linspace(np.min(X), np.max(X), 20)  # create coarse X grid
        Z_int = np.linspace(mindepth, maxdepth, 50)

    fCTD: dict = {}
    for field in fields:
        try:
            # grid over Z
            temp_array: list = []
            for value in CTD.values():
                if field in value:
                    temp_array.append(
                        interp1d(-value["z"], value[field], bounds_error=False)(Z)
                    )
                else:
                    temp_array.append(interp1d(Z, Z * np.nan, bounds_error=False)(Z))
            temp_array = np.array(temp_array).transpose()

            if interp_opt == 0:  # only grid over Z
                fCTD[field] = temp_array
            else:  # grid over Z and X
                temp_array = temp_array.ravel()
                mask = np.where(~np.isnan(temp_array))  # NaN mask
                # grid in X and Z
                fCTD[field] = griddata(
                    (X_orig[mask], Z_orig[mask]),  # old grid
                    temp_array[mask],  # data
                    tuple(np.meshgrid(X_int, Z_int)),
                )  # new grid

            if field == "water_mass":
                fCTD["water_mass"] = np.round(fCTD["water_mass"])
        except:
            logging.warning(
                f"Warning: No gridding possible for '{field}'. Maybe no valid data? Setting to nan..."
            )
            if interp_opt == 0:
                fCTD[field] = np.meshgrid(X, Z)[0] * np.nan
            else:
                fCTD[field] = np.meshgrid(X_int, Z_int)[0] * np.nan

    if interp_opt > 0:
        X, Z = X_int, Z_int

    return fCTD, Z, X, station_locs


__switch_xdim__ = Literal["station", "time"]


def CTD_to_xarray(
    CTD: dict[dict], switch_xdim: __switch_xdim__ = "station"
) -> xr.Dataset:
    """Function to store CTD data in a xarray dataset instead of a dictionary.

    Args:
        CTD (dict[dict]): CTD data. Is created by `read_CTD`
        switch_xdim (str, optional): Keyword to switch between time and station as x dimension for the returned data set. Defaults to "station".
            -  'station' means UNIS station number.

    Returns:
        xr.Dataset: Dataset with two dimensions depth and distance along the section and all measured variables.
    """
    if not isinstance(CTD, dict):
        raise TypeError(f"'CTD' should be a dict, not a {type(CTD).__name__}")

    if switch_xdim not in get_args(__switch_xdim__):
        raise ValueError(
            f"'switch_xdim' should be 'time' or 'distance', not {switch_xdim}."
        )

    # take all stations available
    stations: list = list(CTD.keys())
    if len(stations) == 0:
        raise ValueError(f"The CTD is empty.")

    # construct the Z-vector from the max and min depth of the given stations
    maxdepth: np.ndarray = np.nanmax([np.nanmax(-CTD[i]["z"]) for i in stations])
    mindepth: np.ndarray = np.nanmin([np.nanmin(-CTD[i]["z"]) for i in stations])

    Z: np.ndarray = np.linspace(mindepth, maxdepth, int(maxdepth - mindepth) + 1)

    # collect station numbers and other metadata
    ship_station: np.ndarray = np.array([d["st"] for d in CTD.values()])
    station: np.ndarray = np.array([d["unis_st"] for d in CTD.values()])
    lat: np.ndarray = np.array([d["LAT"] for d in CTD.values()])
    lon: np.ndarray = np.array([d["LON"] for d in CTD.values()])
    bdepth: np.ndarray = np.array([d["BottomDepth"] for d in CTD.values()])

    # construct the X-vector
    X: np.ndarray = np.array([d["datetime"] for d in CTD.values()])

    fields: list = list(
        set(
            [
                field
                for field in CTD[stations[0]]
                if np.size(CTD[stations[0]][field]) > 1
            ]
        )
    )

    fCTD: dict = {}
    for field in fields:
        try:
            # grid over Z
            temp_array: list = []
            for value in CTD.values():
                if field in value:
                    temp_array.append(
                        interp1d(-value["z"], value[field], bounds_error=False)(Z)
                    )
                else:
                    temp_array.append(interp1d(Z, Z * np.nan, bounds_error=False)(Z))
            temp_array = np.array(temp_array).transpose()

            fCTD[field] = temp_array

            if field == "water_mass":
                fCTD["water_mass"] = np.round(fCTD["water_mass"])
        except:
            logging.warning(
                f"Warning: No gridding possible for '{field}'. Maybe no valid data? Setting to nan..."
            )

            fCTD[field] = np.ones([len(X), len(Z)]) * np.nan

    list_da: list = []
    for vari in fCTD.keys():
        list_da.append(
            xr.DataArray(
                data=fCTD[vari],
                dims=["depth", "time"],
                coords={
                    "depth": Z,
                    "time": X,
                    "station": ("time", station),
                    "ship_station": ("time", ship_station),
                    "lat": ("time", lat),
                    "lon": ("time", lon),
                    "bottom_depth": ("time", bdepth),
                },
                name=vari,
            )
        )

    ds: xr.Dataset = xr.merge(list_da)

    ds = ds.sortby("time")
    ds = ds.interp(depth=np.arange(np.ceil(ds.depth[0]), np.floor(ds.depth[-1]) + 1.0))

    if switch_xdim == "station":
        ds = ds.swap_dims({"time": "station"})

    ds["SA"].attrs["long_name"] = "Absolute Salinity [g/kg]"
    ds["S"].attrs["long_name"] = "Salinity [PSU]"
    ds["CT"].attrs["long_name"] = "Conservative Temperature [°C]"
    ds["T"].attrs["long_name"] = "Temperature [°C]"
    ds["C"].attrs["long_name"] = "Conductivity [S/cm]"
    ds["P"].attrs["long_name"] = "Pressure [dbar]"
    ds["SIGTH"].attrs["long_name"] = "Density (sigma-theta) [kg/m3]"
    ds["OX"].attrs["long_name"] = "Oxygen [ml/l]"

    return ds


def section_to_xarray(
    ds: xr.Dataset,
    stations: list = None,
    time_periods: list = None,
    ship_speed_threshold: float = 1.0,
) -> xr.Dataset:
    """Function to extract one section from the CTD/ADCP dataset from the whole cruise and return a new dataset, where distance along the section is the new dimension.

    Args:
        ds (xr.Dataset): Data from CTD or ADCP, read and transformed with the respective functions (see example notebook).
        stations (list, optional): List with the UNIS station numbers in the section. This is used for CTD and LADCP. Defaults to None.
        time_periods (list, optional): List with the start and end points for each time period that contributes to the section. This is used for the VM-ADCPs. Defaults to None.
        ship_speed_threshold (float, optional): Threshold value for the ship speed (m/s) for use of VM-ADCP. Data during times with ship speeds lower than the threshold will be discarded. Only applies for VM-ADCP. Defaults to 1.0.

    Returns:
        xr.Dataset: Dataset with two dimensions depth and distance along the section, and all measured variables.
    """
    if not isinstance(ds, xr.Dataset):
        raise TypeError(f"'ds' should be a xr.Dataset, not a {type(ds).__name__}.")
    if not pd.api.types.is_number(ship_speed_threshold):
        raise TypeError(
            f"'ship_speed_threshold' should be a float, not a {type(ship_speed_threshold).__name__}."
        )

    if (stations == None) & (time_periods != None):  # for VM-ADCP
        ds_section: list = []
        for start, end in time_periods:
            if end > start:
                ds_section.append(ds.sel(time=slice(start, end)))
            elif start > end:
                ds_section.append(ds.sel(time=slice(end, start)))
        ds_section = xr.concat(ds_section, dim="time")
        if len(time_periods) == 1:
            if time_periods[0][0] > time_periods[0][-1]:
                ds_section = ds_section.sortby("time", ascending=False)
        else:
            if time_periods[0][0] > time_periods[-1][0]:
                ds_section = ds_section.sortby("time", ascending=False)

        ds_section = ds_section.where(
            ds_section["speed_ship"] > ship_speed_threshold, drop=True
        )

        ds_section["distance"] = xr.DataArray(
            np.insert(
                np.cumsum(
                    gsw.distance(ds_section.lon.values, ds_section.lat.values) / 1000
                ),
                0,
                0,
            ),
            dims=["time"],
            coords={"distance": ds_section.time},
        )
        ds_section = ds_section.swap_dims({"time": "distance"}).dropna(
            "depth", how="all"
        )
        ds_section = ds_section.transpose("depth", "distance")
        return ds_section

    elif (stations != None) & (time_periods == None):  # for CTD and L-ADCP
        ds_section = ds.sel(station=stations)
        ds_section["distance"] = xr.DataArray(
            np.insert(
                np.cumsum(
                    gsw.distance(ds_section.lon.values, ds_section.lat.values) / 1000
                ),
                0,
                0,
            ),
            dims=["station"],
            coords={"distance": ds_section.station},
        )
        ds_section = ds_section.swap_dims({"station": "distance"}).dropna(
            "depth", how="all"
        )
        ds_section = ds_section.transpose("depth", "distance")
        return ds_section

    else:
        raise ValueError(
            f"Please specify either stations (for CTD and L-ADCP) or time_periods (for VM_ADCP)!"
        )


def mooring_into_xarray(
    dict_of_instr: dict[pd.DataFrame],
    transfer_vars: list[str] = ["T", "S", "SIGTH", "U", "V", "OX", "P"],
) -> xr.Dataset:
    """Function to store mooring data from a mooring in an xarray dataset.
    The returned dataset can be regridded onto a regular time/depth grid using the xarray methods interpolate_na and interp.

    Args:
        dict_of_instr (dict[pd.DataFrame]): Dictionary with the dataframes returned from the respective read functions for the different instruments, keys: depth levels.
        transfer_vars (list[str], optional): Variables to read into the dataset. Defaults to ["T", "S", "SIGTH", "U", "V", "OX", "P"].

    Returns:
        xr.Dataset: Dataset with two dimensions depth and time, and the variables from transfer_vars (Units are striped).
    """

    if not isinstance(dict_of_instr, dict):
        raise TypeError(
            f"'dict_of_instr' should be a dict, not a {type(dict_of_instr).__name__}."
        )
    for i in dict_of_instr.keys():
        if not isinstance(dict_of_instr[i], pd.DataFrame):
            raise TypeError(
                f"'{i}' in dict_of_instr should be a pd.DataFrame, not a {type(dict_of_instr[i]).__name__}."
            )

    if not isinstance(transfer_vars, list):
        raise TypeError(
            f"'transfer_vars' should be a list, not a {type(transfer_vars).__name__}."
        )
    for i in transfer_vars:
        if not isinstance(i, str):
            raise TypeError(
                f"'{i}' in transfer_vars should be a str, not a {type(i).__name__}."
            )

    # working on which variables to keep
    dict_vars: dict[str, list[str]] = {}
    for d in dict_of_instr.keys():
        varis_instr: list[str] = [
            v
            for v in dict_of_instr[d].columns
            if (var := re.match(r"^(.*?)\s*(?:\[[^\]]*\])?$", v))
            and var.group(1) in transfer_vars
        ]
        dict_vars[d] = varis_instr

    # check for vars with diff units
    set_vars: set[str] = set(
        [
            item
            for sublist in [name[1] for name in dict_vars.items()]
            for item in sublist
        ]
    )
    dict_find_double_var: dict[str, list[str]] = {}
    for obj in set_vars:
        if var := re.match(r"^(.*?)\s*(?:\[[^\]]*\])?$", obj):
            var: str = var.group(1)
            if not var in dict_find_double_var.keys():
                dict_find_double_var[var] = [obj]
            else:
                dict_find_double_var[var].append(obj)

    # find the most use unit and use that one
    for item in dict_find_double_var.items():
        if len(item[1]) > 1:
            num_it: list[int] = []
            for i in item[1]:
                num: int = 0
                for d in dict_vars.values():
                    if i in d:
                        num += 1
                num_it.append(num)
            keep: str = item[1][num_it.index(max(num_it))]
            dict_find_double_var[item[0]] = [keep]

    return dict_find_double_var

    list_da: list[xr.DataArray] = []
    for vari in transfer_vars:
        list_df: list[pd.DataFrame] = []
        for d, df_instr in dict_of_instr.items():
            if vari in list(df_instr.keys()):
                list_df.append(df_instr[vari].rename(d))
        if len(list_df) == 0:  # to continue if no data is available
            logging.warning(f"No data for '{vari}' in the mooring data.")
            df_vari: pd.DataFrame = pd.DataFrame(index=pd.DatetimeIndex([]))
        else:
            df_vari = pd.concat(list_df, axis=1)
        df_vari = df_vari.resample("20min").mean()

        list_da.append(
            xr.DataArray(
                data=df_vari,
                dims=["time", "depth"],
                coords={
                    "depth": np.array(list(df_vari.columns), dtype=float),
                    "time": df_vari.index.values,
                },
                name=vari,
            )
        )

    ds: xr.Dataset = xr.merge(list_da)

    var_name: list[str] = []
    for name in transfer_vars:
        split_name: list[str] = name.split(" ")
        if len(split_name) == 2:
            ds[name].attrs["unit"] = split_name[1].strip("[]")
            split_name[0] = " ".join(split_name[0].split("_"))
        var_name.append(split_name[0])

    return ds


def calc_freshwater_content(
    salinity: npt.ArrayLike, depth: npt.ArrayLike, ref_salinity: float = 34.8
) -> float:
    """Calculates the freshwater content from a profile of salinity and depth.

    Args:
        salinity (array_like): The salinity vector.
        depth (array_like): The depth vector.
        ref_salinity (float, optional): The reference salinity. Defaults to 34.8.
            - make sure it is the same unit as 'salinity'.

    Returns:
        float: The freshwater content for the profile, same unit as 'depth'.
    """

    sal: npt.ArrayLike = salinity.copy()

    idx: np.ndarray = np.where(sal > ref_salinity)[0]
    sal[idx] = ref_salinity

    sal = 0.5 * (sal[1:] + sal[:-1])

    dz: np.ndarray = np.diff(depth)

    return -1.0 * np.sum(((sal - ref_salinity) / ref_salinity) * dz)


def myloadmat(filename: str) -> dict:
    """This function should be called instead of direct spio.loadmat
    as it cures the problem of not properly recovering python dictionaries
    from mat files. It calls the function check keys to cure all entries
    which are still mat-objects.

    Args:
        filename (str): Name of the mat file.

    Returns:
        dict: Dictionary with variable names as keys and loaded matrices as values.
    """

    def _check_keys(d: dict) -> dict:
        """
        checks if entries in dictionary are mat-objects. If yes
        todict is called to change them to nested dictionaries
        """
        for key in d:
            if isinstance(d[key], spio.matlab.mat_struct):
                d[key] = _todict(d[key])
        return d

    def _todict(matobj: spio.matlab.mat_struct) -> dict:
        """
        A recursive function which constructs from matobjects nested dictionaries
        """
        d: dict = {}
        for strg in matobj._fieldnames:
            elem = matobj.__dict__[strg]
            if isinstance(elem, spio.matlab.mat_struct):
                d[strg] = _todict(elem)
            elif isinstance(elem, np.ndarray):
                d[strg] = _tolist(elem)
            else:
                d[strg] = elem

        return d

    def _tolist(ndarray: np.ndarray) -> NDArray:
        """
        A recursive function which constructs lists from cellarrays
        (which are loaded as numpy ndarrays), recursing into the elements
        if they contain matobjects.
        """
        elem_list: list = []
        for sub_elem in ndarray:
            if isinstance(sub_elem, spio.matlab.mat_struct):
                elem_list.append(_todict(sub_elem))
            elif isinstance(sub_elem, np.ndarray):
                elem_list.append(_tolist(sub_elem))
            else:
                elem_list.append(sub_elem)
        return np.asarray(elem_list)

    data: dict = spio.loadmat(filename, struct_as_record=False, squeeze_me=True)
    return _check_keys(data)


def mat2py_time(matlab_dnum: np.array) -> pd.DatetimeIndex:
    """
    Converts matlab datenum to python datetime objects.

    Args:
        matlab_dnum (np.array): The matlab datenum.

    Returns:
        pd.DatetimeIndex: The python datetime
    """

    return pd.to_datetime(np.asarray(matlab_dnum) - 719529, unit="D").round("1s")
    # try:
    #     len(matlab_dnum)
    # except:
    #     matlab_dnum = [matlab_dnum]
    # return [datetime.fromordinal(int(t)) + timedelta(days=t%1) - \
    #                         timedelta(days = 366) for t in matlab_dnum]


def present_dict(d: dict, offset="") -> None:
    """Iterative function to present the contents of a dictionary. Prints in
    the terminal.

    Args:
        d (dict): The dictionary.
        offset (str, optional): Offset used for iterative calls. Defaults to "".

    Returns:
        None
    """
    if not isinstance(d, dict):
        raise TypeError(f"'d' should be a dict, not a {type(d).__name__}.")
    if not isinstance(offset, str):
        raise TypeError(f"'offset' should be a dict, not a {type(offset).__name__}.")

    if len(d.keys()) > 50:
        print(offset, "keys:", list(d.keys()))
        print(offset, "first one containing:")
        f: Any = d[list(d.keys())[0]]
        if type(f) == dict:
            present_dict(f, offset=" |" + offset + "       ")
        else:
            print(" |" + offset + "       ", type(f), ", size:", np.size(f))
    else:
        for i, k in d.items():
            if type(k) == dict:
                print(offset, i, ": dict, containing:")
                present_dict(k, offset=" |" + offset + "       ")
                print()
            elif (1 < np.size(k) < 5) and (type(k[0]) != dict):
                print(offset, i, ":", k)
            elif np.size(k) == 1:
                print(offset, i, ":", k)
            elif np.size(k) > 1 and type(k[0]) == dict:
                print(offset, i, ": array of dicts, first one containing:")
                present_dict(k[0], offset=" |" + offset + "       ")
            else:
                print(offset, i, ":", type(k), ", size:", np.size(k))

    return None


__output__ = Literal["pd.DataFrame", "csv", "df_func", "None"]


def create_water_mass_DataFrame(
    output: __output__ | str,
    Abbr: list = ...,
    T_min: list = ...,
    T_max: list = ...,
    S_psu_min: list = ...,
    S_psu_max: list = ...,
) -> None | pd.DataFrame | str:
    """Creates a DataFrame containing the abbreviation of the water mass,
    as well as the minimum and maximum temperature/salinity values (as used by ctd_identify_water_masses()).
    If 'Abbr', 'T_min', 'T_max', 'S_psu_min', and 'S_psu_max' are provided,
    the function skips user input and directly assigns these lists to the DataFrame.
    The resulting DataFrame can be saved as a .csv file and later read using pd.read_csv(filepath).
    To close the input, enter: "done", "quit", "exit" or "save". Press ESC to exit without saving or making any changes.

    Args:
        output (str): Used to define which dataformat is required.
            - 'pd.DataFrame'  : Returns a pandas DataFrame.
            - 'csv'           : Creates a .csv with the Dataframe in the current working directory. Returns None.
            - 'df_func'       : Prints and returns the string which is used to create the DataFrame with pd.DataFrame().
            - 'None'          : Returns None.
            - output directory: The path to the directory the .csv should be saved. Returns None.
        Abbr (list, optional): List with abbreviations. Defaults to None.
            - Needs to be set, to skip the inputs.
        T_min (list, optional): List with the minimum temperatures. Defaults to None.
            - Needs to be set, to skip the inputs.
        T_max (list, optional): List with the maximum temperatures. Defaults to None.
            - Needs to be set, to skip the inputs.
        S_psu_min (list, optional): List with the minimum salinities. Defaults to None.
            - Needs to be set, to skip the inputs.
        S_psu_max (list, optional): List with the maximum salinities. Defaults to None.
            - Needs to be set, to skip the inputs.

    Returns:
        None or pd.DataFrame or str: See 'output'.
    """

    def _createListStr(List: list) -> str:
        """Creates a list to recreate the list in a normal command.

        Args:
            List (list): List thats supposed to be converted.

        Returns:
            str: Converted list.
        """

        List_str: str = "["
        for x in List:
            if not np.isnan(x) and x != float("inf") and x != float("-inf"):
                List_str += str(x) + ","
            elif x == float("inf") or np.isnan(x):
                List_str += "np." + str(x) + ","
            else:
                List_str += "-np.inf,"

        List_str = List_str[:-1]
        List_str += "]"
        return List_str

    if output == None:
        output = "None"
    if output not in get_args(__output__) and not os.path.isdir(output):
        raise ValueError(
            f"'switch_xdim' should be 'pd.DataFrame', 'csv', 'df_func', 'None' or a path, not {output}."
        )

    userinput: str = ""
    if os.path.isdir(output) or output == "csv":
        userinput = input("Please enter a filename (without .csv).")
        if os.path.isdir(output):
            output = os.path.join(output, userinput + ".csv")
        elif "csv" == output:
            output = os.getcwd() + "/" + userinput + ".csv"
        else:
            raise ValueError(f"Something went wrong. This shouldn't happen.")
        userinput: str = ""

    if (
        Abbr == ...
        and T_min == ...
        and T_max == ...
        and S_psu_max == ...
        and S_psu_min == ...
    ):
        Abbr: list = []
        T_min: list = []
        T_max: list = []
        S_psu_min: list = []
        S_psu_max: list = []
        go: bool = True
    else:
        go = False
        if not pd.api.types.is_list_like(Abbr):
            raise TypeError(
                f"'Abbr' should be a list_like, not a {type(Abbr).__name__}."
            )
        if not pd.api.types.is_list_like(T_min):
            raise TypeError(
                f"'T_min' should be a list_like, not a {type(T_min).__name__}."
            )
        if len(Abbr) != len(T_min):
            raise ValueError(
                f"'T_min' should have the same size as 'Abbr'. Has size {len(T_min)} should {len(Abbr)}."
            )
        if not pd.api.types.is_list_like(T_max):
            raise TypeError(
                f"'T_max' should be a list_like, not a {type(T_max).__name__}."
            )
        if len(Abbr) != len(T_max):
            raise ValueError(
                f"'T_max' should have the same size as 'Abbr'. Has size {len(T_max)} should {len(Abbr)}."
            )
        if not pd.api.types.is_list_like(S_psu_min):
            raise TypeError(
                f"'S_psu_min' should be a list_like, not a {type(S_psu_min).__name__}."
            )
        if len(Abbr) != len(S_psu_min):
            raise ValueError(
                f"'S_psu_min' should have the same size as 'Abbr'. Has size {len(S_psu_min)} should {len(Abbr)}."
            )
        if not pd.api.types.is_list_like(S_psu_max):
            raise TypeError(
                f"'S_psu_max' should be a list_like, not a {type(S_psu_max).__name__}."
            )
        if len(Abbr) != len(S_psu_max):
            raise ValueError(
                f"'S_psu_max' should have the same size as 'Abbr'. Has size {len(S_psu_max)} should {len(Abbr)}."
            )

    exitstrs: list = ["quit", "exit", "letmefree", "done", "save"]
    conv: dict = {"np.nan": np.nan, "np.inf": np.inf, "-np.inf": -np.inf}

    while go:
        userinput = input("Please enter the abbreviation of the water mass.")
        if userinput.lower().replace(" ", "") in exitstrs:
            go = False
            break
        else:
            Abbr.append(userinput)

        repeat: int = 1  # 0:finished successfully, 1: first time, 2: repeated run
        while repeat > 0 and go:
            if repeat == 1:
                userinput = input(
                    f"Please enter the minimum temperature for '{Abbr[-1]}'."
                )
            elif repeat == 2:
                userinput = input(
                    f"WARNING: The last input couldn't be converted into a float.\n Please enter the minimum temperature for '{Abbr[-1]}' again."
                )
            elif repeat == 3:
                userinput = input(
                    f"WARNING: The last input couldn't be converted into a float.\n Please enter the minimum temperature for '{Abbr[-1]}' again. If you want to exit type: 'quit'."
                )
            if userinput.lower().replace(" ", "") in exitstrs or userinput == "":
                go = False
                T_min.append(np.nan)
                T_max.append(np.nan)
                S_psu_min.append(np.nan)
                S_psu_max.append(np.nan)
            else:
                try:
                    T_min.append(
                        conv.get(userinput, float(userinput.replace(",", ".")))
                    )
                    repeat = 0
                except ValueError:
                    if repeat == 2:
                        repeat = 3
                    else:
                        repeat = 2
                    logging.warning(
                        f"The input '{userinput}' for for T_min for '{Abbr[-1]}' was not valid!"
                    )

        repeat: int = 1  # 0:finished successfully, 1: first time, 2: repeated run
        while repeat > 0 and go:
            if repeat == 1:
                userinput = input(
                    f"Please enter the maximum temperature for '{Abbr[-1]}'."
                )
            elif repeat == 2:
                userinput = input(
                    f"WARNING: The last input couldn't be converted into a float.\n Please enter the maximum temperature for '{Abbr[-1]}'."
                )
            elif repeat == 3:
                userinput = input(
                    f"WARNING: The last input couldn't be converted into a float.\n Please enter the maximum temperature for '{Abbr[-1]}'. If you want to exit type: 'quit'."
                )
            if userinput.lower().replace(" ", "") in exitstrs or userinput == "":
                go = False
                T_max.append(np.nan)
                S_psu_min.append(np.nan)
                S_psu_max.append(np.nan)
            else:
                try:
                    T_max.append(
                        conv.get(userinput, float(userinput.replace(",", ".")))
                    )
                    repeat = 0
                except ValueError:
                    if repeat == 2:
                        repeat = 3
                    else:
                        repeat = 2
                    logging.warning(
                        f"The input '{userinput}' for T_max for '{Abbr[-1]}' was not valid!"
                    )

        repeat: int = 1  # 0:finished successfully, 1: first time, 2: repeated run
        while repeat > 0 and go:
            if repeat == 1:
                userinput = input(
                    f"Please enter the minumum salinity for '{Abbr[-1]}'."
                )
            elif repeat == 2:
                userinput = input(
                    f"WARNING: The last input couldn't be converted into a float.\n Please enter the minumum salinity for '{Abbr[-1]}'."
                )
            elif repeat == 3:
                userinput = input(
                    f"WARNING: The last input couldn't be converted into a float.\n Please enter the minumum salinity for '{Abbr[-1]}'. If you want to exit type: 'quit'."
                )
            if userinput.lower().replace(" ", "") in exitstrs or userinput == "":
                go = False
                S_psu_min.append(np.nan)
                S_psu_max.append(np.nan)
            else:
                try:
                    S_psu_min.append(
                        conv.get(userinput, float(userinput.replace(",", ".")))
                    )
                    repeat = 0
                except ValueError:
                    if repeat == 2:
                        repeat = 3
                    else:
                        repeat = 2
                    logging.warning(
                        f"The input '{userinput}' for S_psu_min for '{Abbr[-1]}' was not valid!"
                    )

        repeat: int = 1  # 0:finished successfully, 1: first time, 2: repeated run
        while repeat > 0 and go:
            if repeat == 1:
                userinput = input(
                    f"Please enter the maximum salinity for '{Abbr[-1]}'."
                )
            elif repeat == 2:
                userinput = input(
                    f"WARNING: The last input couldn't be converted into a float.\n Please enter the maximum salinity for '{Abbr[-1]}'."
                )
            elif repeat == 3:
                userinput = input(
                    f"WARNING: The last input couldn't be converted into a float.\n Please enter the maximum salinity for '{Abbr[-1]}'. If you want to exit type: 'quit'."
                )
            if userinput.lower().replace(" ", "") in exitstrs or userinput == "":
                go = False
                S_psu_max.append(np.nan)
            else:
                try:
                    S_psu_max.append(
                        conv.get(userinput, float(userinput.replace(",", ".")))
                    )
                    repeat = 0
                except ValueError:
                    if repeat == 2:
                        repeat = 3
                    else:
                        repeat = 2
                    logging.warning(
                        f"The input '{userinput}' for S_psu_max for '{Abbr[-1]}' was not valid!"
                    )

        if go:
            if (
                T_max[-1] < T_min[-1]
                and T_max != float("nan")
                and T_min != float("nan")
            ):
                logging.warning(
                    f"'T_max' ({T_max[-1]}) is bigger then 'T_min' ({T_min[-1]}) for '{Abbr[-1]}'."
                )
            if (
                S_psu_max[-1] < S_psu_min[-1]
                and S_psu_max != float("nan")
                and S_psu_min != float("nan")
            ):
                logging.warning(
                    f"'S_psu_max' ({S_psu_max[-1]}) is bigger then 'S_psu_min' ({S_psu_min[-1]}) for '{Abbr[-1]}'."
                )

    df: pd.DataFrame = pd.DataFrame(
        {
            "Abbr": Abbr,
            "T_max": T_max,
            "T_min": T_min,
            "S_psu_max": S_psu_max,
            "S_psu_min": S_psu_min,
        }
    )

    if output == "pd.Dataframe":
        return df
    if output == "df_func":
        T_max_str: str = _createListStr(T_max)
        T_min_str: str = _createListStr(T_min)
        S_psu_max_str: str = _createListStr(S_psu_max)
        S_psu_min_str: str = _createListStr(S_psu_min)
        out_string: str = (
            'df: pd.DataFrame = pd.DataFrame({"Abbr":',
            Abbr,
            ', "T_max":',
            T_max_str,
            ', "T_min":',
            T_min_str,
            ', "S_psu_max":',
            S_psu_max_str,
            ', "S_psu_min":',
            S_psu_min_str,
            "})",
        )
        print(out_string)
        return out_string
    if output == "None":
        return None
    print(df.head())
    df.to_csv(output, index=False)
    return None


def ctd_identify_water_masses(
    CTD: dict, water_mass_def: pd.DataFrame, stations: list = None
) -> dict[dict]:
    """Function to assign each ctd measurement tuple of T and S the corresponding water mass (AW, TAW, LW etc.).

    Args:
        CTD (dict): CTD data.
            - Created by 'read_CTD'.
        water_mass_def (pd.DataFrame): Contains the water mass abbreviations, T and S limits.
            - Needs to contain columns with the name '['Abbr','T_min','T_max','S_psu_max','S_psu_min']'.
        stations (array_like, optional): List of stations to select from CTD. Defaults to None.
            - If set to None all stations are used.

    Returns:
        dict[dict]: Dict with the ctd data, each station has new variables 'water_mass' and 'water_mass_Abbr'.
    """

    # if no stations are given, take all stations available
    if not isinstance(CTD, dict):
        raise TypeError(f"'CTD' should be a dict, not a {type(CTD).__name__}")
    if stations is None:
        stations = list(CTD.keys())
    else:
        if not pd.api.types.is_list_like(stations):
            raise TypeError(
                f"'stations' should be a array_like, not a {type(stations).__name__}."
            )
        if type(stations) != list:
            stations = [x for x in stations]
        notfound_stations: list = [
            key for key in stations if not key in list(CTD.keys())
        ]
        if len(notfound_stations) != 0:
            logging.info(
                f"The stations '{notfound_stations}' are not in the CTD data. Proceeding without them."
            )
            for i in notfound_stations:
                stations.remove(i)
            if len(stations) == 0:
                raise ValueError(f"There are no CTD stations left.")
        CTD = {key: CTD[key] for key in stations}
    if len(stations) == 0:
        raise ValueError(f"The CTD is empty.")

    if not isinstance(water_mass_def, pd.DataFrame):
        raise TypeError(
            f"'water_mass_def' should be a pandas Dataframe, not a {type(water_mass_def).__name__}."
        )
    if not "Abbr" in water_mass_def.columns:
        raise ValueError(f"'Abbr' should be a column in 'water_mass_def.columns'.")
    if not "T_min" in water_mass_def.columns:
        raise ValueError(f"'T_min' should be a column in 'water_mass_def.columns'.")
    if not "T_max" in water_mass_def.columns:
        raise ValueError(f"'T_max' should be a column in 'water_mass_def.columns'.")
    if not "S_psu_min" in water_mass_def.columns:
        raise ValueError(f"'S_psu_min' should be a column in 'water_mass_def.columns'.")
    if not "S_psu_max" in water_mass_def.columns:
        raise ValueError(f"'S_psu_max' should be a column in 'water_mass_def.columns'.")

    for s in stations:
        CTD[s]["water_mass"] = np.ones_like(CTD[s]["T"]) * np.nan
        CTD[s]["water_mass_Abbr"] = np.empty_like(CTD[s]["T"], dtype="object")
        for index, row in water_mass_def.iterrows():
            if row["Abbr"] != "ArW":
                ind = np.all(
                    np.array(
                        [
                            CTD[s]["T"] > row["T_min"],
                            CTD[s]["T"] <= row["T_max"],
                            CTD[s]["S"] > row["S_psu_min"],
                            CTD[s]["S"] <= row["S_psu_max"],
                        ]
                    ),
                    axis=0,
                )
                CTD[s]["water_mass"][ind] = index
                CTD[s]["water_mass_Abbr"][ind] = row["Abbr"]

    return CTD


############################################################################
# READING FUNCTIONS
############################################################################


def read_ADCP_CODAS(
    filepath: str, loadadditional_var: str | list[str] = None
) -> xr.Dataset:
    """Reads ADCP data from a/multiple  netCDF file(s) processed by CODAS. To be used with the *short* file!

    Args:
        filepath (str): Path to one or more '.nc' file(s).
            - For multiple files, use UNIX-style wildcards ('*' for any character(s), '?' for single character, etc.).
        loadadditional_var (str or list[str], optional): Name(s) of variabels that should be imported as well. Defaults to None.
            - Standard variabels: "u", "v", "lat", "lon", "depth", "amp", "pg", "heading", "uship", "vship"

    Returns:
        xr.Dataset: Dataset containing the adcp data. Current velocities are adjusted for the ship's motion.
    """

    if not isinstance(filepath, str):
        raise TypeError(
            f"Expected filepath as a string, but got {type(filepath).__name__}."
        )
    if not ("short" in os.path.basename(filepath).lower()):
        logging.warning(
            "Warning: This function is written to work with the CODAS *short* file. 'short' was not found in the file name."
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

    if loadadditional_var == None:
        loadadditional_var = []
    if not isinstance(loadadditional_var, str) or not pd.api.types.is_list_like(
        loadadditional_var
    ):
        if pd.api.types.is_list_like(loadadditional_var):
            for i in loadadditional_var:
                if not isinstance(i, str):
                    raise TypeError(
                        f"Objects in 'loadadditional_var' should be a string, not a {type(i).__name__}. ('{i}')"
                    )
        else:
            raise TypeError(
                f"'loadadditional_var' should be a string or a list of strings, not a {type(i).__name__}."
            )
    extract_vars: list[str] = [
        "u",
        "v",
        "lat",
        "lon",
        "depth",
        "amp",
        "pg",
        "heading",
        "uship",
        "vship",
    ]
    if isinstance(loadadditional_var, str):
        extract_vars.append(loadadditional_var)
    else:
        extract_vars.extend(loadadditional_var)

    if len(files) == 1:
        with xr.open_dataset(files[0]) as f:
            ds: xr.Dataset = f[extract_vars].load()
    elif len(files) > 1:
        with xr.open_mfdataset(files) as f:
            ds: xr.Dataset = f[extract_vars].load()

    ds = ds.set_coords(("depth", "lon", "lat"))

    ds["speed_ship"] = xr.apply_ufunc(np.sqrt, ds["uship"] ** 2.0 + ds["vship"] ** 2.0)
    ds["speed_ship"].attrs["name"] = "speed_ship"
    ds["speed_ship"].attrs["units"] = "m/s"
    ds["speed_ship"].attrs["long_name"] = "total ship speed"

    ds["u"].attrs["long_name"] = "Eastward current velocity"
    ds["v"].attrs["long_name"] = "Northward current velocity"
    ds["uship"].attrs["long_name"] = "Eastward ship speed"
    ds["vship"].attrs["long_name"] = "Northward ship speed"
    ds["pg"].attrs["long_name"] = "Percent good"
    ds["heading"].attrs["long_name"] = "Ship heading"
    ds["speed_ship"].attrs["long_name"] = "Ship speed"

    ds = ds.rename(
        {
            "heading": "Heading_ship",
            "uship": "u_ship",
            "vship": "v_ship",
            "speed_ship": "Speed_ship",
        }
    )  # renaming so it has the same naming convention

    return ds


def split_CODAS_resolution(ds: xr.Dataset) -> list[xr.Dataset]:
    """Splits the full ADCP time series into seperate datasets containing only timesteps with the same depth resolution.

    Args:
        ds (xr.Dataset): Dataset containing the full (CODAS-processed) ADCP timeseries (the return from the function read_ADCP_CODAS).

    Returns:
        list[xr.Dataset]: List of xarray datasets with different depth resolutions.
    """

    if not isinstance(ds, xr.Dataset):
        raise TypeError(f"'ds' should be a xr.Dataset, not a {type(ds).__name__}.")

    ds["depth_binsize"] = (
        ds.depth.isel(depth_cell=slice(0, 2))
        .diff(dim="depth_cell")
        .squeeze("depth_cell", drop=True)
        .drop_vars("depth")
    )

    depth_resolutions: list = sorted(list(ds.groupby("depth_binsize").groups.keys()))

    one_d_varis: list[str] = ["Heading_ship", "u_ship", "v_ship", "Speed_ship"]

    list_of_ds: list = []
    for d in depth_resolutions:
        ds_d: xr.Dataset = ds.where(ds.depth_binsize == d, np.nan)
        ds_d["depth"] = ds_d.depth.isel(time=0)
        ds_d = ds_d.swap_dims({"depth_cell": "depth"}).drop_vars("depth_binsize")
        ds_dd: xr.Dataset = ds_d[one_d_varis]
        ds_d = ds_d.where(ds_d.depth.notnull(), drop=True)
        for vari in one_d_varis:
            ds_d[vari] = ds_dd[vari]
        ds_d = ds_d.transpose("depth", "time")
        list_of_ds.append(ds_d)

    return list_of_ds


def read_WinADCP(filepath: str) -> xr.Dataset:
    """Reads data from a .mat data file processed with WinADCP.

    Args:
        filepath (str): String with path to file

    Returns:
        xr.Dataset: Dataset with time, depth as dimensions and the data.
    """

    if not isinstance(filepath, str):
        raise TypeError(
            f"Expected filepath as a string, but got {type(filepath).__name__}."
        )
    if not os.path.isfile(filepath):
        raise FileNotFoundError(f"File not found: {filepath}")
    if not filepath.endswith(".mat"):
        raise ValueError(f"Invalid file format: {filepath}. Expected a .mat file.")

    data: dict = myloadmat(filepath)

    depth: np.ndarray = np.round(
        data["RDIBin1Mid"] + (data["SerBins"] - 1) * data["RDIBinSize"]
    )

    time: list[pd.Timestamp] = [
        pd.Timestamp(year=2000 + y, month=m, day=d, hour=H, minute=M, second=s)
        for y, m, d, H, M, s in zip(
            data["SerYear"],
            data["SerMon"],
            data["SerDay"],
            data["SerHour"],
            data["SerMin"],
            data["SerSec"],
        )
    ]

    glattributes: dict[str, Any] = {
        name: data[name]
        for name in [
            "RDIFileName",
            "RDISystem",
            "RDIBinSize",
            "RDIPingsPerEns",
            "RDISecPerPing",
        ]
    }

    data_vars = dict(
        T=(
            ["time"],
            data["AnT100thDeg"] / 100.0,
            {
                "units": "degC",
                "name": "temperature",
                "long_name": "Sea water temperature",
            },
        ),
        u_raw=(
            ["time", "depth"],
            data["SerEmmpersec"] / 1000.0,
            {
                "units": "m/s",
                "name": "u_raw",
                "long_name": "Raw eastward current velocity",
            },
        ),
        v_raw=(
            ["time", "depth"],
            data["SerNmmpersec"] / 1000.0,
            {
                "units": "m/s",
                "name": "v_raw",
                "long_name": "Raw northward current velocity",
            },
        ),
        pg=(
            ["time", "depth"],
            data["SerPG4"] / 100.0,
            {"units": "percent", "name": "pg", "long_name": "Percent good"},
        ),
        u_ship=(
            ["time"],
            data["AnNVEmmpersec"] / 1000.0,
            {"units": "m/s", "name": "u_ship", "long_name": "Eastward ship speed"},
        ),
        v_ship=(
            ["time"],
            data["AnNVNmmpersec"] / 1000.0,
            {"units": "m/s", "name": "v_ship", "long_name": "Northward ship speed"},
        ),
    )
    if "SerErmmpersec" in data.keys():
        data_vars["velocity_error"] = (
            ["time", "depth"],
            data["SerErmmpersec"] / 1000.0,
            {
                "units": "m/s",
                "name": "velocity_error",
                "long_name": "Current velocity measurement error",
            },
        )
    if "AnBTEmmpersec" in data.keys():
        data_vars["u_bottomtrack"] = (
            ["time"],
            data["AnBTEmmpersec"] / 1000.0,
            {
                "units": "m/s",
                "name": "u_bottomtrack",
                "long_name": "Eastward bottomtrack velocity",
            },
        )
        data_vars["v_bottomtrack"] = (
            ["time"],
            data["AnBTNmmpersec"] / 1000.0,
            {
                "units": "m/s",
                "name": "v_bottomtrack",
                "long_name": "Northward bottomtrack velocity",
            },
        )
    if "AnBTErmmpersec" in data.keys():
        data_vars["bottomtrack_error"] = (
            ["time"],
            data["AnBTErmmpersec"] / 1000.0,
            {
                "units": "m/s",
                "name": "bottomtrack_error",
                "long_name": "Bottomtrack velocity measurement error",
            },
        )
    if "AnWMEmmpersec" in data.keys():
        data_vars["u_barotropic_raw"] = (
            ["time"],
            data["AnWMEmmpersec"] / 1000.0,
            {
                "units": "m/s",
                "name": "u_barotropic_raw",
                "long_name": "Raw eastward barotropic current velocity",
            },
        )
        data_vars["v_barotropic_raw"] = (
            ["time"],
            data["AnWMNmmpersec"] / 1000.0,
            {
                "units": "m/s",
                "name": "v_barotropic_raw",
                "long_name": "Raw northward barotropic current velocity",
            },
        )
    if "AnWMErmmpersec" in data.keys():
        data_vars["barotropic_velocity_error"] = (
            ["time"],
            data["AnWMErmmpersec"] / 1000.0,
            {
                "units": "m/s",
                "name": "barotropic_velocity_error",
                "long_name": "Barotropic current velocity measurement error",
            },
        )

    ds = xr.Dataset(
        data_vars=data_vars,
        coords=dict(time=time, depth=depth),
        # lat_start=(["time"], data["AnFLatDeg"]),
        # lat_end=(["time"], data["AnLLatDeg"]),
        # lon_start=(["time"], data["AnFLonDeg"]),
        # lon_end=(["time"], data["AnLLonDeg"])),
        attrs=glattributes,
    )

    ds["lat"] = xr.DataArray(
        0.5 * (data["AnFLatDeg"] + data["AnLLatDeg"]),
        dims=["time"],
        coords={"time": ds.time},
    )
    ds["lon"] = xr.DataArray(
        0.5 * (data["AnFLonDeg"] + data["AnLLonDeg"]),
        dims=["time"],
        coords={"time": ds.time},
    )

    ds: xr.Dataset = ds.set_coords(("lat", "lon"))

    ds["u"] = ds["u_raw"] + ds["u_ship"]
    ds["u"].attrs["name"] = "u"
    ds["u"].attrs["units"] = "m/s"
    ds["u"].attrs["long_name"] = "Eastward current velocity"

    ds["v"] = ds["v_raw"] + ds["v_ship"]
    ds["v"].attrs["name"] = "v"
    ds["v"].attrs["units"] = "m/s"
    ds["v"].attrs["long_name"] = "Northward current velocity"

    if "u_barotropic_raw" in ds.data_vars:
        ds["u_barotropic"] = ds["u_barotropic_raw"] + ds["u_ship"]
        ds["u_barotropic"].attrs["name"] = "u_barotropic"
        ds["u_barotropic"].attrs["units"] = "m/s"
        ds["u_barotropic"].attrs["long_name"] = "Eastward barotropic current velocity"

        ds["v_barotropic"] = ds["v_barotropic_raw"] + ds["v_ship"]
        ds["v_barotropic"].attrs["name"] = "v_barotropic"
        ds["v_barotropic"].attrs["units"] = "m/s"
        ds["v_barotropic"].attrs["long_name"] = "Northward barotropic current velocity"

    calc_heading = (
        lambda u, v: (((np.rad2deg(np.arctan2(-u, -v)) + 360.0) % 360.0) + 180.0)
        % 360.0
    )
    ds["Heading_ship"] = xr.apply_ufunc(calc_heading, ds["u_ship"], ds["v_ship"])
    ds["Heading_ship"].attrs["name"] = "Heading_ship"
    ds["Heading_ship"].attrs["units"] = "deg"
    ds["Heading_ship"].attrs["long_name"] = "Ship heading"

    ds["Speed_ship"] = xr.apply_ufunc(
        np.sqrt, ds["u_ship"] ** 2.0 + ds["v_ship"] ** 2.0
    )
    ds["Speed_ship"].attrs["name"] = "Speed_ship"
    ds["Speed_ship"].attrs["units"] = "m/s"
    ds["Speed_ship"].attrs["long_name"] = "Ship speed"

    ds = ds.transpose("depth", "time")

    return ds


def VMADCP_calculate_crossvel(ds: xr.Dataset) -> xr.Dataset:
    """Function to calculate the current velocity perpendicular to the ship track from the detided East and North current velocities and the ship's heading.

    Args:
        ds (xr.Dataset): Dataset containing the full VM-ADCP timeseries, after detiding!

    Returns:
        xr.Dataset: Same dataset as input, but with additional variable crossvel
    """

    if not isinstance(ds, xr.Dataset):
        raise TypeError(f"'ds' should be a xr.Dataset, not a {type(ds).__name__}.")
    if not "u_detide" in ds.data_vars:
        raise ValueError(
            f"'ds' needs to have a variable named 'u_detide' use 'detide_VMADCP()'."
        )
    if not "v_detide" in ds.data_vars:
        raise ValueError(
            f"'ds' needs to have a variable named 'v_detide' use 'detide_VMADCP()'."
        )

    calc_crossvel = lambda u, v, angle_deg: v * np.sin(
        np.deg2rad(angle_deg)
    ) - u * np.cos(np.deg2rad(angle_deg))
    ds["crossvel"] = xr.apply_ufunc(
        calc_crossvel, ds["u_detide"], ds["v_detide"], ds["Heading_ship"]
    )
    ds["crossvel"].attrs["name"] = "crossvel"
    ds["crossvel"].attrs["units"] = "m/s"
    ds["crossvel"].attrs[
        "long_name"
    ] = "Current velocity (detided) perpendicular to the ship track"

    return ds


def read_LADCP(
    filepath: str, station_dict: dict, switch_xdim: __switch_xdim__ = "station"
) -> xr.Dataset:
    """Function to read the data from the LADCP-mat-files.

    Args:
        filepath (str): String with path to the .mat datafile.
        station_dict (dict): The CTD dictionary or dictionary connecting the ship station numbers to the UNIS station numbers.
            - Can be generated from the CTD-dict with 'stations_dict = {CTD[i]["st"]: i for i in CTD.keys()}'.
        switch_xdim (str, optional): Keyword to switch between time and station (UNIS station number) as x dimension for the returned data set. Defaults to "station".

    Returns:
        xr.Dataset: xarray dataset containing the l-adcp data.
    """

    if not isinstance(filepath, str):
        raise TypeError(
            f"Expected filepath as a string, but got {type(filepath).__name__}."
        )
    if not os.path.isfile(filepath):
        raise FileNotFoundError(f"File not found: {filepath}")
    if not filepath.endswith(".mat"):
        raise ValueError(f"Invalid file format: {filepath}. Expected a .mat file.")

    if not isinstance(station_dict, dict):
        raise TypeError(
            f"'station_dict' should be a dictionary, not a {type(station_dict).__name__}."
        )
    construct_station_dict: dict = {}
    for i in station_dict.keys():
        if not isinstance(station_dict[i], dict):
            break
        else:
            construct_station_dict[station_dict[i]["st"]] = (
                i  # ships_station_num -> unis_station_number
            )
    if len(construct_station_dict) == len(station_dict):
        station_dict = construct_station_dict

    adcp: dict = myloadmat(filepath)

    variables_to_read: list[str] = ["U", "V", "U_detide", "V_detide"]
    if "E" in adcp.keys():
        variables_to_read += ["E"]

    list_of_das: list[xr.DataArray] = []
    for vari in variables_to_read:
        list_of_dfs: list[pd.DataFrame] = []
        for st in range(len(adcp["stnr"])):
            max_depth: float = np.floor((np.nanmax(adcp["Z"][:, st])))
            grid: np.ndarray = np.arange(max_depth)
            df: pd.DataFrame = pd.DataFrame(
                adcp[vari][:, st],
                index=adcp["Z"][:, st],
                columns=[station_dict[adcp["stnr"][st]]],  # setting unis_staiton_number
            )
            df = df.drop_duplicates().dropna()
            df_resampled: pd.DataFrame = (
                df.reindex(df.index.union(grid)).interpolate("values").loc[grid]
            )
            list_of_dfs.append(df_resampled)

        df_total: pd.DataFrame = pd.concat(list_of_dfs, axis=1)

        list_of_das.append(
            xr.DataArray(
                data=df_total,
                dims=["depth", "station"],
                coords={"station": df_total.columns, "depth": df_total.index},
                name=vari,
            )
        )

    ds: xr.Dataset = xr.merge(list_of_das)
    ds["ship_station"] = xr.DataArray(
        adcp["stnr"], dims=["station"], coords={"station": ds.station}
    )

    aux_variables: dict[str, str] = {"LAT": "lat", "LON": "lon", "ED": "Echodepth"}

    for vari_old, vari_new in aux_variables.items():
        ds[vari_new] = xr.DataArray(
            adcp[vari_old], dims=["station"], coords={"station": ds.station}
        )

    ds["time"] = xr.DataArray(
        pd.to_datetime(np.asarray(adcp["DT"]) - 719529.0, unit="D").round("1s"),
        dims=["station"],
        coords={"station": ds.station},
    )
    ds = ds.set_coords(["lat", "lon", "ship_station", "Echodepth"])
    if switch_xdim == "time":
        ds = ds.swap_dims({"station": "time"})

    ds = ds.rename(
        {
            "U": "u",
            "V": "v",
            "U_detide": "u_detide",
            "V_detide": "v_detide",
            "Echodepth": "bottom_depth",
        }
    )

    ds["u"] = ds["u"] / 100.0
    ds["v"] = ds["v"] / 100.0
    ds["u_detide"] = ds["u_detide"] / 100.0
    ds["v_detide"] = ds["v_detide"] / 100.0

    ds["u"].attrs["long_name"] = "Eastward current velocity"
    ds["v"].attrs["long_name"] = "Northward current velocity"
    ds["u_detide"].attrs["long_name"] = "Detided eastward current velocity"
    ds["v_detide"].attrs["long_name"] = "Detided northward current velocity"

    if "E" in ds.data_vars:
        ds = ds.rename({"E": "velocity_error"})
        ds["velocity_error"] = ds["velocity_error"] / 100.0
        ds["velocity_error"].attrs["long_name"] = "Current velocity measurement error"

    return ds


def read_CTD(
    inpath: str,
    cruise_name: str = "cruise",
    outpath: str = None,
    stations: npt.ArrayLike = None,
    salt_corr: tuple[num.Number, num.Number] = (1.0, 0.0),
    oxy_corr: tuple[num.Number, num.Number] = (1.0, 0.0),
    use_system_time: bool = False,
) -> dict:
    """This function reads in the CTD data from cnv files in `inpath`
    for the stations `stations` and returns a list of dicts containing
    the data. Conductivity correction (if any) can be specified in `corr`

    Args:
        inpath (str):
            - Path to folder where the where the .cnv files are stored.
            - Path to the .cnv file(s) with UNIX-wildecards ('*' for any character(s), '?' for single character, etc.).
            - Path to a .npy file with the data.
        cruise_name (str, optional): Name of the cruise. Defaults to "cruise".
            - Used to create the name of the output file.
        outpath (str, optional): Path to a folder to store the output. Defaults to None.
            - If set to None, the output is not saved.
            - Saves as 'cruise_name'_CTD.npy.
        stations (array_like, optional): List of stations to read in. Defaults to None.
            - If set to None, all stations in 'inpath' are read in.
        salt_corr (tuple[numeric], optional): Tuple with 2 values containing (slope, intersect) of salinity. Defaults to (1.0, 0.0).
            - Uses a linear correction model.
        oxy_corr (tuple[num.Number, num.Number], optional): Tuple with 2 values containing (slope, intersect) of oxygen. Defaults to (1.0, 0.0).
            - Uses a linear correction model.
        use_system_time (bool, optional): Switch to use the system upload time stamp insted of the NMEA one. Defaults to False.
            - False uses NMEA time stamp.
            - True uses system time stamp.

    Returns:
        dict: Dict of dicts containing the data for all the relevant station data.
    """

    if not isinstance(inpath, str):
        raise TypeError(
            f"Expected inpath as a string, but got {type(inpath).__name__}."
        )
    files: list[str] = sorted(glob.glob(inpath))
    # checks if the input is a folder or a .npy file
    if len(files) == 1:
        # first, check if the infile is a npy file. In that case, just read the
        # npy file and return the dict. No correction can be applied.
        if inpath[-4::] == ".npy":
            CTD_dict: dict = np.load(inpath, allow_pickle=True).item()
            if stations is not None:
                # just use existing stations
                if not pd.api.types.is_list_like(stations):
                    raise TypeError(
                        f"'stations' should be a array_like, not a {type(stations).__name__}."
                    )
                if type(stations) != list:
                    stations = [x for x in stations]
                notfound_stations: list = [
                    key for key in stations if not key in list(CTD_dict.keys())
                ]
                if len(notfound_stations) != 0:
                    logging.info(
                        f"The stations '{notfound_stations}' are not in the CTD data. Proceeding without them."
                    )
                    for i in notfound_stations:
                        stations.remove(i)
                    if len(stations) == 0:
                        raise ValueError(f"There are no CTD stations left.")
                CTD_dict = {key: CTD_dict[key] for key in stations}

            return CTD_dict
        elif os.path.isdir(inpath):
            inpath = os.path.join(inpath, "*.cnv")
            files = sorted(glob.glob(inpath))
        elif inpath.endswith(".cnv"):
            pass
        else:
            raise ValueError(
                f"Invalid input: '{inpath}'. Expected valid file name with .npy extension, file name(s) with .cnv extension or folder name"
            )
    if len(files) == 0:
        raise ValueError(f"No such file(s):'{inpath}'")
    for i in files:
        if i.endswith(".npy"):
            raise ValueError(
                f"Invalid input: '{inpath}'. It's not possible to read in multiple .npy files."
            )
        if not i.endswith(".cnv"):
            raise ValueError(
                f"Invalid input: '{inpath}'. All files should end with .cnv"
            )

    if not isinstance(cruise_name, str):
        raise TypeError(
            f"'cruise_name' should be a string, not a {type(cruise_name).__name__}."
        )

    if not isinstance(outpath, str) and outpath != None:
        raise TypeError(
            f"'outpath' should be a string, not a {type(outpath).__name__}."
        )

    if outpath != None:
        if not os.path.isdir(outpath):
            raise ValueError(f"Invalid input: '{outpath}'. Expected valid folder name.")

    if not pd.api.types.is_array_like(stations) and stations != None:
        raise TypeError(
            f"'stations' should be a array_like, not a {type(stations).__name__}."
        )

    if not isinstance(salt_corr, tuple):
        raise TypeError(
            f"'salt_corr' should be a tuple, not a {type(salt_corr).__name__}."
        )
    if not len(salt_corr) == 2:
        raise ValueError(
            f"'salt_corr' should be a tuple with 2 values, not {len(salt_corr)}."
        )
    if not all([isinstance(i, num.Number) for i in salt_corr]):
        raise TypeError(f"'salt_corr' should contain only floats, not {salt_corr}.")

    if not isinstance(oxy_corr, tuple):
        raise TypeError(
            f"'oxy_corr' should be a tuple, not a {type(oxy_corr).__name__}."
        )
    if not len(oxy_corr) == 2:
        raise ValueError(
            f"'oxy_corr' should be a tuple with 2 values, not {len(oxy_corr)}."
        )
    if not all([isinstance(i, num.Number) for i in oxy_corr]):
        raise TypeError(f"'oxy_corr' should contain only floats, not {oxy_corr}.")

    if not isinstance(use_system_time, bool):
        raise TypeError(
            f"'use_system_time' should be a boolean, not a {type(use_system_time).__name__}."
        )

    # If a folder is given, read single cnv files.
    # create a dict that converts the variable names in the cnv files to
    # the variable names used by us:
    var_names: dict[str, str] = {
        "DEPTH": "D [m]",
        "PRES": "P [dbar]",
        "prdM": "P [dbar]",
        "TEMP": "T [degC]",
        "tv290C": "T [degC]",
        "CNDC": "C [S/m]",
        "c0mS/m": "C [S/m]",
        "c0mS/cm": "C [S/cm]",
        "PSAL": "S []",
        "sigma_t": "SIGTH [kg/m^3]",
        "avgsvCM": "Speed_sound_avg [m/s]",
        "sbeox0PS": "OX_sat [%]",
        "seaTurbMtr": "TURB [FTU]",
        "par/sat/log": "PAR [mumol photons/m^2s]",
        "oxygen_ml_L": "OX [ml/l]",
        "potemperature": "T_pot [degC]",
        "oxsolML/L": "OX_sol [ml/l]",
    }

    # If stations are provided, select the ones that exist
    if stations is not None:
        use_files: list[str] = [i for i in files for j in stations if str(j) in i]
        unused_stations: list[str] = [
            i for i in stations if not any(str(i) in j for j in use_files)
        ]
        if len(use_files) == 0:
            raise ValueError(
                f"None of the stations you provided exist in the data files: {unused_stations}."
            )
        if len(unused_stations) != 0:
            logging.warning(
                f"The following stations '{unused_stations}' were not found in the data files."
            )
        files = use_files

    files = sorted(files)

    # Read in the data, file by file
    CTD_dict = {}
    used_unis_stations: dict[str, int] = {}
    num_files: int = len(files)
    for i, file in enumerate(files):
        # to not have a progress bar, when there are just a few files
        if num_files > 60:
            uf.progress_bar(i, num_files - 1)
        # get all the fields, construct a dict with the fields
        profile = fCNV(file)
        p: dict[str, Any] = {
            var_names[name]: profile[name]
            for name in profile.keys()
            if name in var_names
        }

        # get the interesting header fields and append it to the dict
        p.update(profile.attrs)

        # get the UNIS station number
        found_unis_station = False
        with open(file, encoding="ISO-8859-1") as f:
            while not found_unis_station:
                line = f.readline()
                if ("unis station" in line.lower()) or ("unis-station" in line.lower()):
                    found_unis_station = True
                    if ":" in line:
                        unis_station: str = ((line.split(":"))[-1]).strip()
                    else:
                        unis_station = ((line.split(" "))[-1]).strip()
        if not found_unis_station:
            unis_station = "unknown"

        # deal with multiple times the same unis station
        if unis_station in used_unis_stations.keys():
            used_unis_stations[unis_station] += 1
            unis_station = f"{unis_station}_{used_unis_stations[unis_station]}"
        else:
            used_unis_stations[unis_station] = 0

        # if lat and lon not in profile attributes
        if "LATITUDE" not in p.keys():
            p["LATITUDE"] = -999.0
            p["LONGITUDE"] = -999.0
            with open(file, encoding="ISO-8859-1") as f:
                while (p["LATITUDE"] < -990.0) and (p["LONGITUDE"] < -990.0):
                    line: str = f.readline()
                    if "lat" in line.lower():
                        if ":" in line:
                            p["LATITUDE"] = float(((line.split(":"))[-1]).strip())
                        else:
                            p["LATITUDE"] = float(((line.split(" "))[-1]).strip())
                    if "lon" in line.lower():
                        if ":" in line:
                            p["LONGITUDE"] = float(((line.split(":"))[-1]).strip())
                        else:
                            p["LONGITUDE"] = float(((line.split(" "))[-1]).strip())

        # if NMEA time is wrong, replace with system upload time (needs to be manually switched on)
        if use_system_time:
            found_system_time = False
            with open(file, encoding="ISO-8859-1") as f:
                while not found_system_time:
                    line = f.readline()
                    if "system upload time" in line.lower():
                        found_system_time = True
                        p["datetime"] = datetime.datetime.strptime(
                            ((line.split("="))[-1]).strip(), "%b %d %Y %H:%M:%S"
                        )
            if not found_system_time:
                p["datetime"] = datetime.datetime(1970, 1, 1, 0, 0, 0)

        # if time is present: convert to dnum
        try:
            p["dnum"] = date2num(p["datetime"])
        except:
            pass
        # rename the most important ones to the same convention used in MATLAB,
        # add other important ones

        p["LAT"] = p.pop("LATITUDE")
        p["LON"] = p.pop("LONGITUDE")
        p["z [m]"] = gsw.z_from_p(p["P [dbar]"], p["LAT"])
        p["BottomDepth [m]"] = np.round(np.nanmax(np.abs(p["z [m]"])) + 8)
        if np.nanmin(p["C [S/m]"]) > 10.0:
            p["C [S/m]"] /= 10.0
        p["C [S/m]"][p["C [S/m]"] < 1] = np.nan
        p["T [degC]"][p["T [degC]"] < -2] = np.nan
        p["S []"] = salt_corr[0] * p["S []"] + salt_corr[1]  # apply correction
        p["S []"][p["S []"] < 20] = np.nan
        p["C [S/m]"][p["S []"] < 20] = np.nan
        p["SA [g/kg]"] = gsw.SA_from_SP(p["S []"], p["P [dbar]"], p["LON"], p["LAT"])
        p["CT [degC]"] = gsw.CT_from_t(p["SA [g/kg]"], p["T [degC]"], p["P [dbar]"])
        p["SIGTH [kg/m^3]"] = gsw.sigma0(p["SA [g/kg]"], p["CT [degC]"])
        if p["filename"].split(".")[0].split("_")[0][-4::].isdigit():
            p["st"] = int(p["filename"].split(".")[0].split("_")[0][-4::])
        p["unis_st"] = unis_station
        if "OX [ml/l]" in p:
            p["OX [ml/l]"] = oxy_corr[0] * p["OX [ml/l]"] + oxy_corr[1]
        CTD_dict[p["unis_st"]] = p

    # check if a station was duplicated
    dub_stations: list[str] = [
        i for i in used_unis_stations.keys() if used_unis_stations[i] > 0
    ]
    for i in dub_stations:
        CTD_dict[i + "_0"] = CTD_dict[i]
        CTD_dict.pop(i)
    if len(dub_stations) > 0:
        logging.info(
            f"The following stations were duplicated (naming convention: first station (timewise): UnisNum_0, second: UnisNum_1,...):"
        )
        for i in dub_stations:
            logging.info(f"{i} was found {used_unis_stations[i]+1} times.")

    # save data if outpath was given
    if outpath is not None:
        outpath = os.path.join(outpath, cruise_name + "_CTD")
        np.save(outpath, CTD_dict)

    return CTD_dict


def read_CTD_from_mat(matfile: str) -> dict:
    """Reads CTD data from a matfile.

    Args:
        matfile (str): Path to the .mat file.
            - This should contain a struct with the name CTD.
            - This is the common output style of the cruise matlab scripts.

    Returns:
        dict: Dictionary with the CTD Data.
    """

    if not isinstance(matfile, str):
        raise TypeError(
            f"'matfile' should be a string, not a {type(matfile).__name__}."
        )
    if not matfile.endswith(".mat"):
        raise ValueError(f"Invalid file format: {matfile}. Expected a .mat file.")
    if not os.path.isfile(matfile):
        raise FileNotFoundError(f"File not found: {matfile}.")

    # read the raw data using scipy.io.loadmat
    raw_data = loadmat(matfile, squeeze_me=True, struct_as_record=False)["CTD"]
    # convert to dictionary
    CTD: dict = {}
    for record in raw_data:
        station = record.__dict__["st"]
        CTD[station] = record.__dict__
        CTD[station].pop("_fieldnames", None)
        # correct dnum parameter, because MATLAB and PYTHON
        # datenumbers are different
        CTD[station]["dnum"] = datestr2num(CTD[station]["date"])

    if "note" in CTD[next(iter(CTD))]:
        logging.info("Note: This CTD data is already calibrated.")

    return CTD


def read_MSS(files: str, excel_file: str = None) -> tuple[dict, dict, dict]:
    """Function to read MSS data from .mat-files can also read the excel_file to add latitude and longditue.
    Calculates z from p and adds it to the data (if no latitude is given, uses 60N).

    Args:
        files (str):
            - Path to the folder where the .mat file(s) are stored.
            - Path to .mat file(s) with UNIX-wildcards ('*' for any character(s), '?' for single character, etc.).
        excel_file (str, optional): Path to the .xlsx file. Defaults to None.
            - This should contain a table with the following columns:
                - Station name
                - Latitude/ N
                - Longitude/ E
                - File name CASTXXX.MRD
            - The file name CASTXXX.MRD should be the same as the file name of the .mat file.
            - The latitude and longitude are used to add the coordinates to the data.

    Returns:
        tuple[dict,dict,dict]:
            - Dictionary with the CTD data.
            - Dictionary with the MIX data.
            - Dictionary with the DATA data.
    """

    if excel_file == None:
        pass
    elif not isinstance(excel_file, str):
        raise TypeError(
            f"'excel_file' should be a string, not a {type(excel_file).__name__}."
        )
    elif not excel_file.endswith(".xlsx"):
        raise ValueError(f"Invalid file format: {excel_file}. Expected a .xlsx file.")
    elif not os.path.isfile(excel_file) and excel_file != None:
        raise FileNotFoundError(f"File not found: {excel_file}.")

    if not isinstance(files, str):
        raise TypeError(f"'files' should be a string, not a {type(files).__name__}.")
    if os.path.isdir(files):
        files = glob.glob(os.path.join(files, "*.mat"))
    else:
        files = glob.glob(files)

    if len(files) == 0:
        raise FileNotFoundError(f"No files where found.")

    for file in files:
        if not file.endswith(".mat"):
            raise ValueError(f"Invalid file format: {file}. Expected a .mat file.")

    # first, handle the excel file
    if excel_file:
        try:
            exc: pd.DataFrame = pd.read_excel(excel_file)
            exc.columns = np.arange(len(exc.columns))
            st: pd.Series | pd.DataFrame = exc[
                [a for a in exc.columns if "Station name" in exc[a].to_numpy()][0]
            ]
            st_ind: int = st[st == "Station name"].index[0]
            lat_deg: pd.Series | pd.DataFrame = exc[
                [a for a in exc.columns if "Latitude/ N" in exc[a].to_numpy()][0]
            ]
            lat_deg_ind: int = lat_deg[lat_deg == "Latitude/ N"].index[0]
            lat_min: pd.Series | pd.DataFrame = exc[
                [a for a in exc.columns if "Latitude/ N" in exc[a].to_numpy()][0] + 1
            ]
            lat_min_ind: int = lat_min[lat_min == "min"].index[0]
            lon_deg: pd.Series | pd.DataFrame = exc[
                [a for a in exc.columns if "Longitude/ E" in exc[a].to_numpy()][0]
            ]
            lon_deg_ind: int = lon_deg[lon_deg == "Longitude/ E"].index[0]
            lon_min: pd.Series | pd.DataFrame = exc[
                [a for a in exc.columns if "Longitude/ E" in exc[a].to_numpy()][0] + 1
            ]
            lon_min_ind: int = lon_min[lon_min == "min"].index[0]
            file_nr: pd.Series | pd.DataFrame = exc[
                [
                    a
                    for a in exc.columns
                    if "File name CASTXXX.MRD" in exc[a].to_numpy()
                ][0]
            ]
            file_nr_ind: int = file_nr[file_nr == "File name CASTXXX.MRD"].index[0]

            st = st[st_ind + 2 :].reset_index(drop=True)
            file_nr = file_nr[file_nr_ind + 2 :].reset_index(drop=True)
            lat_deg = lat_deg[lat_deg_ind + 2 :].reset_index(drop=True)
            lat_min = lat_min[lat_min_ind + 1 :].reset_index(drop=True)
            lon_deg = lon_deg[lon_deg_ind + 2 :].reset_index(drop=True)
            lon_min = lon_min[lon_min_ind + 1 :].reset_index(drop=True)

            if (
                len(
                    {
                        len(st),
                        len(file_nr),
                        len(lat_deg),
                        len(lat_min),
                        len(lon_deg),
                        len(lon_min),
                    }
                )
                != 1
            ):
                raise IndexError
        except IndexError:
            logging.error(
                f"Error: Couldn't use the excel file. Failed to read the tabel format."
            )
            excel_file = None

    out_data: dict[str, dict] = {"CTD": {}, "MIX": {}, "DATA": {}}
    len_files: int = len(files)
    for num, file in enumerate(files):
        # to not have a progress bar, when there are just a few files
        if len_files > 20:
            uf.progress_bar(num + 1, len_files)

        st_name = int(file.split(".mat")[0][-4:])
        raw_data: dict = myloadmat(file)
        data: dict[str, dict] = {k: raw_data[k] for k in ["CTD", "MIX", "DATA"]}

        for name in ["CTD", "MIX"]:
            for var in ["LON", "LAT", "fname", "date"]:
                data[name][var] = raw_data["STA"][var]

            if excel_file:
                matches: np.array = np.where(file_nr == st_name)[0]
                if len(matches) != 0:
                    index: int = matches[0]
                    data[name]["LON"] = lon_deg[index] + float(lon_min[index]) / 60
                    data[name]["LAT"] = lat_deg[index] + float(lat_min[index]) / 60
                    data[name]["Station name"] = st[index]
                else:
                    logging.warning(
                        f"Warning: No match for {st_name} in the excel file."
                    )

            try:
                data[name]["z"] = gsw.z_from_p(data[name]["P"], data[name]["LAT"])
            except:  # just use 60N as lat if lat is not provided
                data[name]["z"] = gsw.z_from_p(data[name]["P"], 60)

            # Something weird in the data...
            data[name]["z"][np.isnan(data[name]["SIGTH"])] = np.nan
            data[name]["BottomDepth"] = np.nanmax(-data[name]["z"])
            data[name]["datetime"] = pd.to_datetime(
                data[name]["date"], format="%d-%b-%Y %H:%M:%S"
            )

        for name in ["CTD", "MIX", "DATA"]:
            out_data[name][st_name] = data[name]

    return out_data["CTD"], out_data["MIX"], out_data["DATA"]


def read_mooring(filepath: str, normal_dict: bool = True) -> dict:
    """Read mooring data prepared in a either a .npy or .mat file.
    The data is converted into a dictionary with the depth as key and the
    data as value. The data is a pandas dataframe with the time as index
    and the variables as columns.

    Args:
        filepath (str): Path to the .npy or .mat file.
        normal_dict (bool, optional): If the data should be sorted by depths (sensor) or a raw output. Defaults to True.
            - If True, the data is sorted by depth (sensor).
    Returns:
        dict: Dictionary with mooring data, format depends on normal_dict.
    """

    if not isinstance(filepath, str):
        raise TypeError(
            f"'filepath' should be a string, not a {type(filepath).__name__}."
        )
    if not os.path.isfile(filepath):
        raise FileNotFoundError(f"File not found: {filepath}.")

    if filepath.endswith(".mat"):
        raw_data: dict = myloadmat(filepath)
        variable_name = list(raw_data.keys())[-1]
        raw_data = raw_data[variable_name]

        for key in raw_data.keys():
            if key[:4] == "date":
                raw_data[key] = mat2py_time(np.asarray(raw_data[key]))
    elif filepath.endswith(".npy"):
        raw_data = np.load(filepath, allow_pickle=True).item()
    else:
        raise ValueError(
            f"Invalid file format: '{filepath}'. Expected a .mat or .npy file."
        )
    if not normal_dict:
        return raw_data

    else:
        # get it into a dict[str,pd.DataFrame] form
        sensors: list[str] = [
            i.split("=")[0].strip(" ") for i in raw_data["sensor"]
        ]  # saves the sensor serialnumber
        vars: list[str] = [i.split("=")[0].strip(" ") for i in raw_data["name"]]
        vars_unit: list[str] = [i for i in raw_data["unit"]]
        if not "date" in vars:
            logging.warning(
                f"Warning: Did not find 'date' in the data. Giving back the data without changing the format."
            )
            return raw_data
        else:
            vars_unit.pop(vars.index("date")), vars.remove("date")  # gets rid of time
        found_md: bool = True

        if not "md" in vars:
            logging.warning(
                f"Warning: Did not find 'md' in the data. Trying to use the mean depth."
            )
            found_md = False
        else:
            vars_unit.pop(vars.index("md")), vars.remove("md")

        # starting constructing the dict
        norm_dict: dict = {}
        sort: bool = True
        # sensors with multiple depths
        # {depthname,snum: [depthvar0, depthvar1, ...]}
        multidepth: dict[str, list[str]] = {}
        for snum in sensors:
            # checking for sensors with multiple depths
            for var_0 in vars:
                if var_0 + snum in raw_data.keys() and len(
                    raw_data[var_0 + snum]
                ) != len(raw_data["date" + snum]):
                    multidepth[var_0 + "," + snum] = [
                        var_1
                        for var_1 in vars
                        if var_0 != var_1
                        and var_1 + snum in raw_data.keys()
                        and (len(raw_data["date" + snum]), len(raw_data[var_0 + snum]))
                        == raw_data[var_1 + snum].shape
                    ]

            df: pd.DataFrame = pd.DataFrame(
                {
                    f"{var} [{var_unit}]": raw_data[var + snum]
                    for var, var_unit in zip(vars, vars_unit)
                    if var + snum in raw_data.keys()
                    if len(raw_data[var + snum]) == len(raw_data["date" + snum])
                    if np.ndim(raw_data[var + snum]) == 1
                },
                index=raw_data["date" + snum],
            )

            df = uf.std_names(df)

            df.columns = df.columns.to_series().replace(
                {
                    "P ": "p ",
                    "O ": "OX ",
                    "Osat": "OX_sat",
                    "U ": "u ",
                    "V ": "v ",
                    "W ": "w ",
                },
                regex=True,
            )
            if found_md and "md" + snum in raw_data.keys():
                norm_dict[np.round(raw_data["md" + snum])] = df
                if df.empty:
                    logging.warning(
                        f"Warning: No data at depth {np.round(raw_data['md' + snum])}."
                    )
            elif "p" in df.columns:
                norm_dict[np.round(df.p)] = df
                if df.empty:
                    logging.warning(f"Warning: No data at depth {np.round(df.p)}.")
            else:
                norm_dict[sensors.index(snum)] = df
                if df.empty:
                    logging.warning(f"Warning: No data at {sensors.index(snum)}.")
                sort = False

        for depths_snum, sensors in multidepth.items():
            snum: str = depths_snum.split(",")[1]
            depthname: str = depths_snum.split(",")[0] + snum
            for i, depth in enumerate(raw_data[depthname]):
                df = pd.DataFrame(
                    {
                        f"{var_name} [{vars_unit[vars.index(var_name)]}]": raw_data[
                            var_name + snum
                        ][:, i]
                        for var_name in sensors
                    },
                    index=raw_data["date" + snum],
                )
                df = uf.std_names(df)
                df.columns = df.columns.to_series().replace(
                    {
                        "P ": "p ",
                        "O ": "OX ",
                        "Osat": "OX_sat",
                        "U ": "u ",
                        "V ": "v ",
                        "W ": "w ",
                    },
                    regex=True,
                )
                # if data already exists at that depth
                if np.round(float(depth)) in norm_dict.keys():
                    # if variables already exist, rename them (_x)
                    existing_df: pd.DataFrame = norm_dict[np.round(float(depth))]
                    existing_vars: list[str] = [
                        i
                        for i in df.columns
                        if i in existing_df.columns
                        or i.split(" ")[0] + "_0 " + i.split(" ")[1]
                        in existing_df.columns
                    ]
                    new_vars: list[str] = []
                    for i in existing_vars:
                        n = 1
                        new_name: str = (
                            i.split(" ")[0] + "_" + str(n) + " " + i.split(" ")[1]
                        )
                        while new_name in existing_df.columns:
                            n += 1
                            new_name = (
                                i.split(" ")[0] + "_" + str(n) + " " + i.split(" ")[1]
                            )
                        logging.info(
                            "The variable {0} at a depth of {1} is from the sensor number {2}.".format(
                                new_name.split(" ")[0], np.round(float(depth)), snum
                            )
                        )
                        new_vars.append(new_name)

                    for i in existing_vars:
                        if i in existing_df.columns:
                            existing_df.rename(
                                {i: i.split(" ")[0] + "_0 " + i.split(" ")[1]},
                                axis=1,
                                inplace=True,
                            )
                    df.rename(
                        {old: new for old, new in zip(existing_vars, new_vars)},
                        axis=1,
                        inplace=True,
                    )
                    norm_dict[np.round(float(depth))] = pd.merge(
                        existing_df, df, left_index=True, right_index=True, how="outer"
                    )
                else:
                    norm_dict[np.round(float(depth))] = df

        if sort:
            norm_dict = dict(sorted(norm_dict.items()))

        return norm_dict


def read_Seaguard(filepath: str, header_len: int = 4) -> pd.DataFrame:
    """Reads data from one data file from a Seaguard.

    Args:
        filepath (str): Path to the .txt file.
        header_len (int, optional): Number of header lines tat have to be skipped. Defaults to 4.

    Returns:
        pd.DataFrame: Dataframe with time as index and the individual variables as columns.
    """

    if not isinstance(filepath, str):
        raise TypeError(
            f"'filepath' should be a string, not a {type(filepath).__name__}."
        )
    if not filepath.endswith(".txt"):
        raise ValueError(f"Invalid file format: {filepath}. Expected a .txt file.")
    if not os.path.isfile(filepath):
        raise FileNotFoundError(f"File not found: {filepath}.")

    if not isinstance(header_len, int):
        raise TypeError(
            f"'header_len' should be a int, not a {type(header_len).__name__}."
        )

    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="Could not infer format, so each element will be parsed individually, falling back to `dateutil`. To ensure parsing is consistent and as-expected, please specify a format.",
            category=UserWarning,
        )

        df: pd.DataFrame = pd.read_csv(
            filepath,
            sep="\t",
            header=header_len,
            parse_dates=["Time tag (Gmt)"],
            dayfirst=True,
        )

    df.rename({"Time tag (Gmt)": "TIMESTAMP"}, axis=1, inplace=True)
    df = df.set_index("TIMESTAMP")
    df.sort_index(axis=0, inplace=True)

    df = uf.std_names(df)

    if "p [kPa]" in df.columns:
        df["p [dbar]"] = df["p [kPa]"] / 10.0

    return df


def read_Minilog(filepath: str) -> pd.DataFrame:
    """Reads data from one data file from a VEMCO Minilog temperature sensor.

    Args:
        filepath (str): Path to the .csv file.

    Returns:
        pd.DataFrame: Dataframe with time as index and temperature as column.
    """

    if not isinstance(filepath, str):
        raise TypeError(
            f"'filepath' should be a string, not a {type(filepath).__name__}."
        )
    if not filepath.endswith(".csv"):
        raise ValueError(f"Invalid file format: {filepath}. Expected a .csv file.")
    if not os.path.isfile(filepath):
        raise FileNotFoundError(f"File not found: {filepath}.")

    with open(
        filepath,
        "r",
        encoding="ISO-8859-1",
    ) as f:
        for i in range(7):
            f.readline()
        col_names: list[str] = f.readline().strip().split(",")

    if ("date" in col_names[0].lower()) and ("time" in col_names[0].lower()):
        df: pd.DataFrame = pd.read_csv(
            filepath,
            sep=",",
            skiprows=7,
            parse_dates=[col_names[0]],
            encoding="ISO-8859-1",
        )
        df.rename({f"{col_names[0]}": "TIMESTAMP"}, axis=1, inplace=True)
        df["TIMESTAMP"] = pd.to_datetime(df["TIMESTAMP"])
    else:
        df = pd.read_csv(
            filepath,
            sep=",",
            skiprows=7,
            encoding="ISO-8859-1",
        )
        df["TIMESTAMP"] = pd.to_datetime(
            df[col_names[0]].astype(str) + " " + df[col_names[1]].astype(str)
        )
        df.drop([col_names[0], col_names[1]], axis=1, inplace=True)

    df = df.set_index("TIMESTAMP")
    df.sort_index(axis=0, inplace=True)
    df.rename({"Temperature (°C)": "T [degC]"}, axis=1, inplace=True)

    return df


def read_SBE37(filepath: str) -> pd.DataFrame:
    """Reads data from one data file from a SBE37 Microcat sensor.

    Args:
        filepath (str): Path to the .cnv file.

    Returns:
        pd.DataFrame: Dataframe with time as index and the individual variables as columns.
    """

    if not isinstance(filepath, str):
        raise TypeError(
            f"'filepath' should be a string, not a {type(filepath).__name__}."
        )
    if not filepath.endswith(".cnv"):
        raise ValueError(f"Invalid file format: {filepath}. Expected a .cnv file.")
    if not os.path.isfile(filepath):
        raise FileNotFoundError(f"File not found: {filepath}.")

    var_names: dict[str, str] = {
        "cond0S/m": "C",
        "sigma-�00": "SIGTH",
        "prdM": "p",
        "potemperature": "T_pot",
        "tv290C": "T",
        "timeS": "Time",
        "PSAL": "S",
    }

    data = fCNV(filepath)

    d: dict[str, Any] = {
        var_names[name]: data[name] for name in data.keys() if name in var_names
    }

    d.update(data.attrs)

    d["TIMESTAMP"] = pd.to_datetime(
        d["Time"], unit="s", origin=pd.Timestamp(d["start_time"].split("[")[0].strip())
    )

    df: pd.DataFrame = pd.DataFrame(
        0.0,
        index=d["TIMESTAMP"],
        columns=list(
            set(
                [
                    field
                    for field in d
                    if (
                        (np.size(d[field]) > 1) and (field not in ["Time", "TIMESTAMP"])
                    )
                ]
            )
        ),
    )
    for k in df.columns:
        df[k] = d[k]
    df.sort_index(axis=0, inplace=True)

    df = uf.std_names(df, add_units=True, module="o")

    return df


def read_SBE26(filepath: str) -> pd.DataFrame:
    """Reads data from one data file from a SBE26 sensor.

    Args:
        filepath (str): Path to the .tid file.

    Returns:
        pd.DataFrame: Dataframe with time as index and the individual variables as columns.
    """

    if not isinstance(filepath, str):
        raise TypeError(
            f"'filepath' should be a string, not a {type(filepath).__name__}."
        )
    if not filepath.endswith(".tid"):
        raise ValueError(f"Invalid file format: {filepath}. Expected a .tid file.")
    if not os.path.isfile(filepath):
        raise FileNotFoundError(f"File not found: {filepath}.")

    df: pd.DataFrame = pd.read_csv(
        filepath, sep="\s+", header=None, names=["RECORD", "date", "time", "P", "T"]
    )
    df["TIMESTAMP"] = pd.to_datetime(
        df["date"] + " " + df["time"], format="%m/%d/%Y %H:%M:%S"
    )
    df.set_index("TIMESTAMP", inplace=True)
    df.drop(["date", "time"], axis=1, inplace=True)
    if "P" in df.columns:
        df.rename({"P": "p"}, axis=1, inplace=True)

    df = uf.std_names(df, add_units=True, module="o")

    return df


def read_RBR(filepath: str) -> pd.DataFrame:
    """Reads data from a .rsk data file from a RBR logger (concerto, solo, ...).

    Args:
        filepath (str): Path to the .rsk file.

    Returns:
        pd.DataFrame: Dataframe with time as index and the individual variables as columns.
    """

    if not isinstance(filepath, str):
        raise TypeError(
            f"'filepath' should be a string, not a {type(filepath).__name__}."
        )
    if not filepath.endswith(".rsk"):
        raise ValueError(f"Invalid file format: {filepath}. Expected a .rsk file.")
    if not os.path.isfile(filepath):
        raise FileNotFoundError(f"File not found: {filepath}.")

    with RSK(filepath) as rsk:
        rsk.readdata()
        rsk.deriveseapressure()
        variables = list(rsk.channelNames)
        time: pd.DatetimeIndex = pd.to_datetime(rsk.data["timestamp"])

        if "conductivity" in variables:
            rsk.derivesalinity()
            rsk.derivesigma()
            # variables.append("salinity")
            # variables.append("density")
        variables = list(rsk.channelNames)

        data: np.NDArray = rsk.data[variables]

        df: pd.DataFrame = pd.DataFrame(data, index=time, columns=variables)

        df = uf.std_names(df, add_units=True)

        df.sort_index(axis=0, inplace=True)

        # convert units, because RBR uses
        # mS/cm
        if "C [S/m]" in df.columns and np.mean(df["C [S/m]"]) >= 4:
            df["C [S/m]"] = df["C [S/m]"] * 0.1

    return df


def read_Thermosalinograph(
    filepath: str, use_system_time: bool = False
) -> pd.DataFrame:
    """Reads data from one data file from the Helmer Hanssen thermosalinograph.

    Args:
        filepath (str): Path to one or more .cnv file(s).
            - For multiple files, use UNIX-style wildcards ('*' for any character(s), '?' for single character, etc.)
        use_system_time (bool, optional): Switch to use the system upload time stamp insted of the NMEA one. Defaults to False.
            - False uses NMEA time stamp.
            - True uses system time stamp.

    Returns:
        pd.DataFrame: Dataframe with time as index and the individual variables as columns.
    """

    if not isinstance(filepath, str):
        raise TypeError(
            f"'filepath' should be a string, not a {type(filepath).__name__}."
        )
    list_of_files: list[str] = sorted(glob.glob(filepath))
    for i in list_of_files:
        if not i.endswith(".cnv"):
            raise ValueError(
                f"Invalid file format: '{filepath}'. Expected a .cnv file."
            )
        if not os.path.isfile(i):
            raise FileNotFoundError(f"File not found: {filepath}.")

    if not isinstance(use_system_time, bool):
        raise TypeError(
            f"'use_system_time' should be a boolean, not a {type(use_system_time).__name__}."
        )
    var_names: dict[str, str] = {
        "CNDC": "C",
        "sigma-�00": "SIGTH",
        "prM": "P",
        "potemperature": "T_pot",
        "TEMP": "T",
        "timeS": "Time",
        "PSAL": "S",
        "LATITUDE": "lat",
        "LONGITUDE": "lon",
    }

    list_of_df: list = []
    num_files: int = len(list_of_files)
    for i, file in enumerate(list_of_files):
        data = fCNV(file)
        if num_files > 2:
            uf.progress_bar(i, num_files - 1)
        d: dict[str, Any] = {
            var_names[name]: data[name] for name in data.keys() if name in var_names
        }

        d.update(data.attrs)

        if use_system_time:
            found_system_time = False
            with open(file, encoding="ISO-8859-1") as f:
                while not found_system_time:
                    line: str = f.readline()
                    if "system upload time" in line.lower():
                        found_system_time = True
                        d["start_time"] = line.split("=")[-1].strip()
        else:
            d["start_time"] = d["start_time"].split("[")[0].strip()

        d["TIMESTAMP"] = pd.to_datetime(
            d["Time"], unit="s", origin=pd.Timestamp(d["start_time"])
        )

        df = pd.DataFrame(
            0.0,
            index=d["TIMESTAMP"],
            columns=list(
                set(
                    [
                        field
                        for field in d
                        if (
                            (np.size(d[field]) > 1)
                            and (field not in ["Time", "TIMESTAMP"])
                        )
                    ]
                )
            ),
        )
        for k in df.columns:
            df[k] = d[k]
        df.sort_index(axis=0, inplace=True)

        list_of_df.append(df)

    df_total: pd.DataFrame = pd.concat(list_of_df)
    df_total.sort_index(axis=0, inplace=True)

    df_total = uf.std_names(df_total, add_units=True, module="o")

    return df_total


############################################################################
# TIDE FUNCTIONS
############################################################################


def download_tidal_model(
    model: str = "Arc2kmTM", outpath: str | pathlib.Path = pathlib.Path.cwd()
) -> None:
    """Function to download a tidal model later used to calculate
    e.g. tidal currents at a certain location with the pyTMD
    package. This only needs to be done once.

    Args:
        model (str, optional): String specifying the tidal model to download. Defaults to "Arc2kmTM".
            - Options are: "AODTM-5", "AOTIM-5", "AOTIM-5-2018", "Arc2kmTM", "Gr1kmTM".
        outpath (str or pathlib.Path, optional): Path where a new folder with the tidal data will be created. Defaults to current directory.

    Returns:
        None
    """

    if not isinstance(model, str):
        raise ValueError(f"'model' should be a string, not a {type(model).__name__}.")
    if not isinstance(outpath, (str, pathlib.Path)):
        raise ValueError(
            f"'outpath' should be a string, not a {type(outpath).__name__}."
        )

    if pyTMD.utilities.check_connection("https://arcticdata.io"):
        logging.info("starting download...")

        # digital object identifier (doi) for each Arctic tide model
        DOI: dict[str, str] = {}
        DOI["AODTM-5"] = "10.18739/A2901ZG3N"
        DOI["AOTIM-5"] = "10.18739/A2S17SS80"
        DOI["AOTIM-5-2018"] = "10.18739/A21R6N14K"
        DOI["Arc2kmTM"] = "10.18739/A2D21RK6K"
        DOI["Gr1kmTM"] = "10.18739/A2B853K18"
        # local subdirectory for each Arctic tide model
        LOCAL: dict[str, str] = {}
        LOCAL["AODTM-5"] = "aodtm5_tmd"
        LOCAL["AOTIM-5"] = "aotim5_tmd"
        LOCAL["AOTIM-5-2018"] = "Arc5km2018"
        LOCAL["Arc2kmTM"] = "Arc2kmTM"
        LOCAL["Gr1kmTM"] = "Gr1kmTM"

        # recursively create directories if non-existent
        DIRECTORY: matplotlib.Path = pathlib.Path(outpath).expanduser().absolute()
        local_dir: matplotlib.Path = DIRECTORY.joinpath(LOCAL[model])
        local_dir.mkdir(0o775, parents=True, exist_ok=True)

        # build host url for model
        resource_map_doi: str = f"resource_map_doi:{DOI[model]}"
        HOST: list[str] = [
            "https://arcticdata.io",
            "metacat",
            "d1",
            "mn",
            "v2",
            "packages",
            pyTMD.utilities.quote_plus(posixpath.join("application", "bagit-097")),
            pyTMD.utilities.quote_plus(resource_map_doi),
        ]
        # download zipfile from host
        zfile = zipfile.ZipFile(pyTMD.utilities.from_http(HOST, timeout=360))
        # find model files within zip file
        rx: re.Pattern[str] = re.compile(
            r"(grid|h[0]?|UV[0]?|Model|xy)_(.*?)", re.VERBOSE
        )
        members: list = [m for m in zfile.filelist if rx.search(m.filename)]
        # extract each member
        for m in members:
            # strip directories from member filename
            m.filename = posixpath.basename(m.filename)
            local_file = local_dir.joinpath(m.filename)
            # extract file
            zfile.extract(m, path=local_dir)
            # change permissions mode
            local_file.chmod(mode=0o775)
            # close the zipfile object
            zfile.close()

        logging.info("Done downloading!")

    return None


def detide_VMADCP(ds, path_tidal_models, tidal_model="Arc2kmTM", method="spline"):
    """
    Function to correct the VM-ADCP data for the tides (substract the tidal currents from the measurements).

    Parameters
    ----------
    ds : xarray dataset
        Data from VM-ADCP, read and transformed with the respective functions (see example notebook).
    path_tidal_models : str
            Path to the folder with all the tidal model data (don't include the name of the actual tidal model here!)
    tidal_model : str
        Name of the tidal model to be used (also name of the folder where these respective tidal model data are stored)
    method : str
        Spatial interpolation method (from tidal model grid to actual locations of the ship). One of 'bilinear', 'spline', 'linear' and 'nearest'

    Returns
    -------
    ds : same xarray dataset as the input, but with additional variables for the tidal currents and the de-tided measurements

    """

    time = (
        (ds.time.to_pandas() - pd.Timestamp(1970, 1, 1, 0, 0, 0)).dt.total_seconds()
    ).values

    tide_uv = pyTMD.compute.tide_currents(
        ds.lon.values,
        ds.lat.values,
        time,
        DIRECTORY=path_tidal_models,
        MODEL=tidal_model,
        EPSG=4326,
        EPOCH=(1970, 1, 1, 0, 0, 0),
        TYPE="drift",
        TIME="UTC",
        METHOD=method,
        EXTRAPOLATE=True,
        FILL_VALUE=np.nan,
    )

    ds["u_tide"] = xr.DataArray(
        tide_uv["u"] / 100.0, dims=["time"], coords={"station": ds.time}, name="u_tide"
    )
    ds["v_tide"] = xr.DataArray(
        tide_uv["v"] / 100.0, dims=["time"], coords={"station": ds.time}, name="v_tide"
    )

    ds["u_detide"] = ds["u"] - ds["u_tide"]
    ds["v_detide"] = ds["v"] - ds["v_tide"]

    ds["u_detide"].attrs["units"] = "m/s"
    ds["u_detide"].attrs["name"] = "u_detide"
    ds["u_detide"].attrs["long_name"] = "Detided eastward current velocity"
    ds["v_detide"].attrs["units"] = "m/s"
    ds["v_detide"].attrs["name"] = "v_detide"
    ds["v_detide"].attrs["long_name"] = "Detided northward current velocity"

    ds["u_tide"].attrs["units"] = "m/s"
    ds["u_tide"].attrs["name"] = "u_tide"
    ds["u_tide"].attrs["long_name"] = "Eastward tidal current velocity"
    ds["v_tide"].attrs["units"] = "m/s"
    ds["v_tide"].attrs["name"] = "v_tide"
    ds["v_tide"].attrs["long_name"] = "Northward tidal current velocity"

    return ds


def get_tidal_uvh(
    latitude,
    longitude,
    time,
    path_tidal_models,
    tidal_model="Arc2kmTM",
    method="spline",
):
    """
    Function to calculate time series of tidal currents u and v as well as the surface elevation change h for a given pair of lat and lon (only one position at a time!), based on the specified tidal model.

    Parameters
    ----------
    latitude : float
        Latitude of e.g. the mooring
    longitude : float
        Longitude of e.g. the mooring
    time : pandas DatetimeIndex
        Timestamps of the time series. Needs to be a pandas DatetimeIndex. Use df.index of e.g. the raw SeaGuard data, if you have read the data with the read_Seaguard-function.
    path_tidal_models : str
            Path to the folder with all the tidal model data (don't include the name of the actual tidal model here!)
    tidal_model : str
        Name of the tidal model to be used (also name of the folder where these respective tidal model data are stored)
    method : str
        Spatial interpolation method (from tidal model grid to actual locations of the ship). One of 'bilinear', 'spline', 'linear' and 'nearest'

    Returns
    -------
    df : pandas Dataframe with time as the index, and three columns u, v, and h with the tidal current velocities and the surface height anomalies.
    """

    time_model = ((time - pd.Timestamp(1970, 1, 1, 0, 0, 0)).total_seconds()).values

    tide_uv = pyTMD.compute.tide_currents(
        longitude,
        latitude,
        time_model,
        DIRECTORY=path_tidal_models,
        MODEL=tidal_model,
        EPSG=4326,
        EPOCH=(1970, 1, 1, 0, 0, 0),
        TYPE="time series",
        TIME="UTC",
        METHOD=method,
        EXTRAPOLATE=True,
        FILL_VALUE=np.nan,
    )
    tide_h = pyTMD.compute.tide_elevations(
        longitude,
        latitude,
        time_model,
        DIRECTORY=path_tidal_models,
        MODEL=tidal_model,
        EPSG=4326,
        EPOCH=(1970, 1, 1, 0, 0, 0),
        TYPE="time series",
        TIME="UTC",
        METHOD=method,
        EXTRAPOLATE=True,
        FILL_VALUE=np.nan,
    )

    df = pd.DataFrame(
        {
            "u [m/s]": tide_uv["u"].squeeze() / 100.0,
            "v [m/s]": tide_uv["v"].squeeze() / 100.0,
            "h [m]": tide_h.squeeze(),
        },
        index=time,
    )

    return df


def calculate_tidal_spectrum(data, bandwidth=8):
    """
    Function to calculate a spectrum for a given time series (e.g. pressure measurements from a SeaGuard). The raw periodogram is filtered using a multitapering approach.

    Parameters
    ----------
    data : pd.Series
        time Series of data
    bandwidth : int, optional
        bandwidth for the multitaper smoothing. Should be 2,4,8,16,32 etc. (the higher the number the stronger the smoothing) Default is 8.

    Returns
    -------
    s_multitap : pd.Series
        Series with the spectral data, the index is specifying the frequency.
    """

    timeseries = data.interpolate(method="linear").values
    resolution = (data.index[1] - data.index[0]).seconds / 3600.0
    delta = resolution * (1.0 / 24.0)  # in days

    N = len(timeseries)

    # Periodogram
    freq, _ = signal.periodogram(timeseries, fs=1.0 / delta, return_onesided=False)

    # Multitapering
    Sk_complex, weights, eigenvalues = spectrum.mtm.pmtm(
        timeseries,
        NW=bandwidth,
        NFFT=N,
        k=int(2 * bandwidth - 1),
        method="adapt",
        show=False,
    )
    Sk = np.abs(Sk_complex) ** 2.0
    Sk = Sk.T
    multitap = np.mean(Sk * weights, axis=1) * delta

    return pd.Series(multitap[freq >= 0.0], index=freq[freq >= 0.0])


def tidal_harmonic_analysis(data, constituents=["M2"], remove_mean=False):
    """
    Function to perform a tidal harmonic analysis on a given time series, e.g. pressure or u/v current measurements from a SeaGuard.

    Parameters
    ----------
    data : pd.Series
        time Series of time series data.
    constituents : list, optional
        List with constituents to include in the analysis. Default is ['M2'].
    remove_mean : bool, optional
        Switch to enable substracting the mean of the time series. Default is False.

    Returns
    -------
    amp : array
        Array with the amplitudes of the constituents specified in the function call (in the respective order.)
    pha : array
        Array with the phases of the constituents specified in the function call (in the respective order.)
    detid : pd.Series
        Pandas Series with the residual (detided) time series.
    tidal_ts : list
        List of pd.Series, each of the series is the pure tidal time series of the respective constituent (in the same order as amp, pha and the constituents in the function call).
    """

    if remove_mean:
        if "Z0" not in constituents:
            constituents = ["Z0"] + constituents
    else:
        if "Z0" in constituents:
            constituents.remove("Z0")

    time_seconds = np.array([(t - data.index[0]).total_seconds() for t in data.index])

    tide = uptide.Tides(constituents)
    tide.set_initial_time(pd.Timestamp(data.index[0]).to_pydatetime())
    amp, pha = uptide.harmonic_analysis(tide, data.values, time_seconds)

    detid = data - pd.Series(
        tide.from_amplitude_phase(amp, pha, time_seconds), index=data.index
    )

    if "Z0" in constituents:
        amp = list(amp)
        pha = list(pha)
        i = constituents.index("Z0")
        del amp[i]
        del pha[i]
        constituents.remove("Z0")
        amp = np.asarray(amp)
        pha = (np.asarray(pha),)
    tide = uptide.Tides(constituents)
    tide.set_initial_time(pd.Timestamp(data.index[0]).to_pydatetime())

    tidal_ts = []
    for a, p in zip(amp, pha):
        tidal_ts.append(
            pd.Series(
                tide.from_amplitude_phase([a], [p], time_seconds), index=data.index
            )
        )

    return amp, pha, detid, tidal_ts


############################################################################
# PLOTTING FUNCTIONS
############################################################################


def contour_section(
    X,
    Y,
    Z,
    Z2=None,
    ax=None,
    station_pos=None,
    cmap="jet",
    Z2_contours=None,
    clabel="",
    bottom_depth=None,
    clevels=20,
    station_text="",
    interp_opt=1,
    tlocator=None,
    cbar=True,
):
    """
    Plots a filled contour plot of *Z*, with contour lines of *Z2* on top to
    the axes *ax*. It also displays the position of stations, if given in
    *station_pos*, adds labels to the contours of Z2, given in
    *Z2_contours*. If no labels are given, it assumes Z2 is density (sigma0)
    and adds its own labels. It adds bottom topography if given in *bottom_depth*.

    Parameters
    ----------
    X : (N,K) array_like
        X-values.
    Y : (N,K) array_like
        Y-values.
    Z : (N,K) array_like
        the filled contour field.
    Z2 : (N,K) array_like, optional
        the contour field on top. The default is None.
    ax : plot axes, optional
        axes object to plot on. The default is the current axes.
    station_pos : (S,) array_like, optional
        the station positions. The default is None (all stations are plotted).
    cmap : str or array_like, optional
        the colormap for the filled contours. The default is 'jet'.
    Z2_contours : array_like, optional
        the contour label positions for `Z2`str. The default is None.
    clabel : str, optional
        label to put on the colorbar. The default is ''.
    bottom_depth : (S,) array_like, optional
        list with bottom depth. The default is None.
    clevels : array_like or number, optional
        list of color levels, or number of levels to use for `Z`.
        The default is 20.
    station_text : str, optional
        Name to label the station locations. Can be the Section Name for
        instance. The default is ''.
    interp_opt: int, optional
        Indicator which is used to decide whether to use pcolormesh or contourf
    tlocator: matplotlib.ticker locators, optional
        special locator for the colorbar. For example logarithmic values,
        for that use matplotlib.ticker.LogLocator(). Default is None.

    Returns
    -------
    ax : plot axes
        The axes of the plot.
    """
    # open new figure and get current axes, if none is provided
    if ax is None:
        ax = plt.gca()

    # get the labels for the Z2 contours
    if Z2 is not None and Z2_contours is None:
        Z2_contours = np.concatenate([list(range(21, 26)), np.arange(25.5, 29, 0.2)])
        Z2_contours = [i for i in Z2_contours if np.nanmin(Z2) < i < np.nanmax(Z2)]

    # get the Y-axis limits
    y_limits = (0, np.nanmax(Y))
    if bottom_depth is not None:
        y_limits = (0, np.nanmax(bottom_depth))

    if interp_opt == 0:  # only z-interpolation: use pcolormesh
        norm = None
        if type(clevels) == int:
            if tlocator == "logarithmic":
                norm = matplotlib.colors.LogNorm(np.nanmin(Z), np.nanmax(Z))
            cmap = plt.cm.get_cmap(cmap, clevels)
        else:
            norm = matplotlib.colors.BoundaryNorm(
                clevels, ncolors=len(clevels) - 1, clip=False
            )
            if tlocator == "logarithmic":
                norm = matplotlib.colors.LogNorm(np.min(clevels), np.max(clevels))
            cmap = plt.cm.get_cmap(cmap, len(clevels))

        cT = ax.pcolormesh(X, Y, Z, cmap=cmap, shading="auto", norm=norm)  # draw Z
        plt.xlim(np.nanmin(X), np.nanmax(X))
    else:  # full interpolation: use contours
        locator = None
        if tlocator == "logarithmic":
            locator = matplotlib.ticker.LogLocator()
        cT = ax.contourf(
            X, Y, Z, cmap=cmap, levels=clevels, extend="both", locator=locator
        )  # draw Z

    if Z2 is not None:
        cSIG = ax.contour(
            X, Y, Z2, levels=Z2_contours, colors="k", linewidths=[1], alpha=0.6
        )  # draw Z2
        clabels = plt.clabel(
            cSIG, cSIG.levels, fontsize=8, fmt="%1.1f"
        )  # add contour labels
        [
            txt.set_bbox(dict(facecolor="white", edgecolor="none", pad=0, alpha=0.6))
            for txt in clabels
        ]
    else:
        cSIG = None

    if cbar:
        plt.colorbar(cT, ax=ax, label=clabel, pad=0.01)  # add colorbar

    ax.set_ylim(y_limits)
    ax.invert_yaxis()

    # add bathymetry
    if bottom_depth is not None:
        # make sure bottom_depth is an np.array
        bottom_depth = np.asarray(bottom_depth)

        ax.fill_between(
            station_pos,
            bottom_depth * 0 + y_limits[1] + 10,
            bottom_depth,
            zorder=999,
            color="gray",
        )

    ax.xaxis.set_ticks_position("both")
    ax.yaxis.set_ticks_position("both")

    # add station ticks
    if station_pos is not None:
        for i, pos in enumerate(station_pos):
            ax.text(pos, 0, "v", ha="center", fontweight="bold")
            if len(station_text) == len(station_pos):
                ax.annotate(
                    str(station_text[i]),
                    (pos, 0),
                    xytext=(0, 10),
                    textcoords="offset points",
                    ha="center",
                )

    return ax, cT, cSIG


def plot_CTD_section(
    CTD,
    stations,
    section_name="",
    clevels_T=20,
    clevels_S=20,
    x_type="distance",
    interp_opt=1,
    bottom=False,
    z_fine=False,
):
    """
    This function plots a CTD section of Temperature and Salinity,
    given CTD data either directly or via a file.

    Parameters
    ----------
    CTD : str or dict
        Either a dict of dicts containing the CTD data, which can be made with
              the function read_CTD. Or a str with a file where the dict is stored
    stations : array_like
        stations to plot (station numbers have to be found inside the CTD data!).
    section_name : str, optional
        name of the Section, will appear in the plot title. The default is ''.
    clevels_T : array-like or number, optional
        The levels of the filled contourf for the temperature plot. Either a number of levels,
        or the specific levels. The defauls is 20.
    x_type : str, optional
        Wheter to use 'distance' or 'time' as the x-axis. The default is 'distance'.
    interp_opt: int, optional
        Integer which interpolation method to use for gridding
                     0: no interpolation,
                     1: linear interpolation, fine grid (default),
                     2: linear interpolation, coarse grid. The default is 1.
    z_fine: Whether to use a fine z grid. If True, will be 10 cm, otherwise 1 m

    Returns
    -------
    axT: matplotlib.pyplot.axes
        The axes for the temperature subplot
    axS: matplotlib.pyplot.axes
        The axes for the Salinity subplot
    Ct_T:
        The ...
    """
    # Check if the function has data to work with
    assert type(CTD) in [dict, str], (
        "Parameter *CTD*: You must provide either\n"
        " a) a data dict or \n b) a npy file string with the data !"
    )

    # read in the data (only needed if no CTD-dict, but a file was given)
    if type(CTD) is str:
        print("reading file...")
        CTD = np.load(CTD, allow_pickle=True).item()

    # Check if all stations given are found in the data
    assert min([np.isin(st, list(CTD.keys())) for st in stations]), (
        "Not all "
        "of the provided stations were found in the CTD data! \n"
        "The following stations were not found in the data: "
        + "".join([str(st) + " " for st in stations if ~np.isin(st, list(CTD.keys()))])
    )
    # Check if x_type is either distance or time
    assert x_type in ["distance", "time"], "x_type must be eigher distance or " "time!"

    # select only the given stations in the data
    CTD = {key: CTD[key] for key in stations}

    # extract Bottom Depth
    if type(bottom) == bool:
        BDEPTH = np.asarray([d["BottomDepth"] for d in CTD.values()])
    else:
        BDEPTH = bottom

    # put the fields (the vector data) on a regular, common pressure and X grid
    # by interpolating.
    fCTD, Z, X, station_locs = CTD_to_grid(
        CTD, x_type=x_type, interp_opt=interp_opt, z_fine=z_fine
    )

    # plot the figure
    fig, [axT, axS] = plt.subplots(2, 1, figsize=(8, 9), sharex=True)

    # Temperature
    _, Ct_T, C_T = contour_section(
        X,
        Z,
        fCTD["T"],
        fCTD["SIGTH"],
        ax=axT,
        station_pos=station_locs,
        cmap=cmocean.cm.thermal,
        clabel="Temperature [˚C]",
        bottom_depth=BDEPTH,
        clevels=clevels_T,
        station_text=stations,
        interp_opt=interp_opt,
    )
    # Salinity
    _, Ct_S, C_S = contour_section(
        X,
        Z,
        fCTD["S"],
        fCTD["SIGTH"],
        ax=axS,
        station_pos=station_locs,
        cmap=cmocean.cm.haline,
        clabel="Salinity []",
        bottom_depth=BDEPTH,
        clevels=clevels_S,
        interp_opt=interp_opt,
    )
    # Add x and y labels
    axT.set_ylabel("Depth [m]")
    axS.set_ylabel("Depth [m]")
    if x_type == "distance":
        axS.set_xlabel("Distance [km]")
    else:
        axS.set_xlabel("Time [h]")

    # add title
    fig.suptitle(section_name, fontweight="bold")

    # tight_layout
    fig.tight_layout(h_pad=0.1, rect=[0, 0, 1, 0.95])

    return axT, axS, Ct_T, Ct_S, C_T, C_S


def plot_CTD_single_section(
    CTD,
    stations,
    section_name="",
    x_type="distance",
    parameter="T",
    parameter_contourlines="SIGTH",
    clabel="Temperature [˚C]",
    cmap=cmocean.cm.thermal,
    clevels=20,
    contourlevels=5,
    interp_opt=1,
    bottom=False,
    tlocator=None,
    z_fine=False,
    cbar=True,
):
    """
    This function plots a CTD section of a chosen variable,
    given CTD data either directly (through `CTD`) or via a file (through)
    `infile`.
    Parameters
    ----------
    CTD : str or dict
        Either a dict of dicts containing the CTD data, which can be made with
              the function read_CTD. Or a str with a file where the dict is stored
    stations : array_like
        stations to plot (station numbers have to be found inside the CTD data!).
    section_name : str, optional
        name of the Section, will appear in the plot title. The default is ''.
    x_type : str, optional
        Wheter to use 'distance' or 'time' as the x-axis. The default is 'distance'.
    parameter : str, optional
        Which parameter to plot as filled contours. Check what parameters are available
        in `CTD`. The default is 'T'.
    parameter_contourlines : str, optional
        Which parameter to plot as contourlines. Check what parameters are available
        in `CTD`. The default is 'SIGTH'.
    clabel : str, optional
        The label on the colorbar axis. The default is 'Temperature [˚C]'.
    cmap : array-like or str, optional
        The colormap to be used. The default is cmocean.cm.thermal.
    clevels : array-like or number, optional
        The levels of the filled contourf. Either a number of levels,
        or the specific levels. The defauls is 20.
    contourlevels : array-like or number, optional
        The levels of the contourlines. Either a number of levels,
        or the specific levels. The defauls is 5.
    bottom : array-like or False, optional
        The bottom topography, either an array with values extracted from a bathymetry file, or False (default).
        If False, the bottom depth from the CTD profiles will be used.
    interp_opt: int, optional
        Integer which interpolation method to use for gridding
                     0: no interpolation,
                     1: linear interpolation, fine grid (default),
                     2: linear interpolation, coarse grid. The default is 1.
    tlocator: matplotlib.ticker locators, optional
        special locator for the colorbar. For example logarithmic values,
        for that use matplotlib.ticker.LogLocator(). Default is None.
    z_fine: Whether to use a fine z grid. If True, will be 10 cm, otherwise 1 m
    cbar: switch to enable/disable the colorbar

    Returns
    -------
    None.
    """
    # Check if the function has data to work with
    assert type(CTD) in [dict, str], (
        "Parameter *CTD*: You must provide either\n"
        " a) a data dict or \n b) a npy file string with the data !"
    )

    # read in the data (only needed if no CTD-dict, but a file was given)
    if type(CTD) is str:
        print("reading file...")
        CTD = np.load(CTD, allow_pickle=True).item()

    # Check if all stations given are found in the data
    assert min([np.isin(st, list(CTD.keys())) for st in stations]), (
        "Not all "
        "of the provided stations were found in the CTD data! \n"
        "The following stations were not found in the data: "
        + "".join([str(st) + " " for st in stations if ~np.isin(st, list(CTD.keys()))])
    )
    # Check if x_type is either distance or time
    assert x_type in ["distance", "time"], "x_type must be eigher distance or " "time!"

    # select only the given stations in the data
    CTD = {key: CTD[key] for key in stations}

    # extract Bottom Depth
    if type(bottom) == bool:
        BDEPTH = np.asarray([d["BottomDepth"] for d in CTD.values()])
    else:
        BDEPTH = bottom

    # put the fields (the vector data) on a regular, common pressure and X grid
    # by interpolating.
    fCTD, Z, X, station_locs = CTD_to_grid(
        CTD, x_type=x_type, interp_opt=interp_opt, z_fine=z_fine
    )

    # plot the figure
    fig, ax = plt.subplots(1, 1, figsize=(8, 5))

    # plot the cross section
    _, Ct, C = contour_section(
        X,
        Z,
        fCTD[parameter],
        fCTD[parameter_contourlines],
        ax=ax,
        station_pos=station_locs,
        cmap=cmap,
        clabel=clabel,
        bottom_depth=BDEPTH,
        station_text=stations,
        clevels=clevels,
        Z2_contours=contourlevels,
        interp_opt=interp_opt,
        tlocator=tlocator,
        cbar=cbar,
    )

    # Add x and y labels
    ax.set_ylabel("Depth [m]")
    if x_type == "distance":
        ax.set_xlabel("Distance [km]")
    else:
        ax.set_xlabel("Time [h]")

    # add title
    fig.suptitle(section_name, fontweight="bold")

    # tight_layout
    fig.tight_layout(h_pad=0.1, rect=[0, 0, 1, 0.95])
    return ax, Ct, C


def plot_xarray_sections(
    list_das,
    list_cmaps,
    list_clevels=None,
    da_contours=None,
    contourlevels=5,
    interp=False,
    switch_cbar=True,
    add_station_ticks=True,
):
    """
    Function to plot a variable number of variables from a section. Data can be from CTD or ADCP, but has to be provided as xarray datasets (see example notebook!)

    Parameters
    ----------
    list_das : list
        Each element of the list must be an xarray dataarray containing T, S, current data.
    list_cmaps : list
        Colormaps to be used for each subplot, order corresponding to the first list with the data.
    list_clevels : list, optional
        List with levels to use for the contourf plots, order corresponding to the first list with the data. Can bei either a integer (then it specifies the number of levels) or the explicit levels. The default is 20 levels.
    da_contours : xarray data array, optional
        Dataarray with data to plot as contour lines on top of the countourf. Typically used for density lines. The default is None (no contour lines).
    contourlevels : int or array-like, optional
        Sale as the clevels for the contourf plots, but for the contour lines. The default is 5.
    interp : bool, optional
        Switch to enable interpolation of the data onto a finer grid along the distance axis. The default is False.
    switch_cbar : bool, optional
        Switch to enable adding a colorbar to each contourf plot. The default is True.
    add_station_ticks : bool, optional
        Switch to add ticks for the locations of the CTD stations along the section. The default is True.

    Returns
    -------
    fig, ax
        The handles for the figure and the axes, to be used for further adjustments.

    """

    if list_clevels == None:
        list_clevels = len(list_das) * [20]

    N_subplots = len(list_das)

    fig, axes = plt.subplots(
        N_subplots, 1, sharey=True, sharex=True, figsize=(12, N_subplots * 4)
    )
    if N_subplots == 1:
        axes = [axes]
    pics = []
    for i, da in enumerate(list_das):
        if interp:
            X = da.distance.to_numpy()
            Z = da.depth.to_numpy()
            # original grids
            X_orig, Z_orig = [f.ravel() for f in np.meshgrid(X, Z)]
            # new grids
            X_int = np.linspace(np.min(X), np.max(X), len(X) * 20)  # create fine X grid
            Z_int = Z[:]
            temp_array = da.to_numpy().ravel()
            mask = np.where(~np.isnan(temp_array))  # NaN mask
            # grid in X and Z
            data_to_plot = griddata(
                (X_orig[mask], Z_orig[mask]),  # old grid
                temp_array[mask],  # data
                tuple(np.meshgrid(X_int, Z_int)),
            )  # new grid
            if da.name == "water_mass":
                data_to_plot = np.round(data_to_plot)
            pic = axes[i].contourf(
                X_int,
                Z_int,
                data_to_plot,
                cmap=list_cmaps[i],
                levels=list_clevels[i],
                extend="both",
            )
            pics.append(pic)
            if switch_cbar:
                cbar = plt.colorbar(pic, ax=axes[i])
                cbar.ax.set_ylabel(da.attrs["long_name"])
        else:
            pic = da.plot.pcolormesh(
                x="distance",
                y="depth",
                ax=axes[i],
                shading="nearest",
                cmap=list_cmaps[i],
                levels=list_clevels[i],
                add_colorbar=switch_cbar,
                infer_intervals=False,
                robust=True,
                extend="both",
            )
            pics.append(pic)

        if da_contours is not None:
            if interp:
                X = da_contours.distance.to_numpy()
                Z = da_contours.depth.to_numpy()
                # original grids
                X_orig, Z_orig = [f.ravel() for f in np.meshgrid(X, Z)]
                # new grids
                X_int = np.linspace(
                    np.min(X), np.max(X), len(X) * 20
                )  # create fine X grid
                Z_int = Z[:]
                temp_array = da_contours.to_numpy().ravel()
                mask = np.where(~np.isnan(temp_array))  # NaN mask
                # grid in X and Z
                data_to_plot = griddata(
                    (X_orig[mask], Z_orig[mask]),  # old grid
                    temp_array[mask],  # data
                    tuple(np.meshgrid(X_int, Z_int)),
                )  # new grid
                contourlines = axes[i].contour(
                    X_int,
                    Z_int,
                    data_to_plot,
                    levels=contourlevels,
                    colors="k",
                    linewidths=[1],
                    alpha=0.6,
                )
            else:
                contourlines = da_contours.plot.contour(
                    x="distance",
                    y="depth",
                    ax=axes[i],
                    levels=contourlevels,
                    colors="k",
                    linewidths=[1],
                    alpha=0.6,
                )
            clabels = plt.clabel(
                contourlines, contourlines.levels, fontsize=8, fmt="%1.1f"
            )
            [
                txt.set_bbox(
                    dict(facecolor="white", edgecolor="none", pad=0, alpha=0.6)
                )
                for txt in clabels
            ]

        axes[i].set_xlabel("")
        axes[i].set_ylabel("Depth [m]")

    # extract bathymetry
    bottom = None
    found_bottom = False
    i = 0
    while (found_bottom == False) & (i < len(list_das)):
        if "bottom_depth" in list_das[i].coords:
            bottom = list_das[i]["bottom_depth"].to_numpy()
            bottom_x = list_das[i]["distance"].to_numpy()
            found_bottom = True
        i += 1
    if (found_bottom == False) & (da_contours is not None):
        if "station" in da_contours.coords:
            bottom = da_contours.coords["bottom_depth"].to_numpy()
            bottom_x = da_contours.coords["distance"].to_numpy()
            found_bottom = True

    # get the axis limits
    if bottom is not None:
        y_limits = (0, np.nanmax(bottom))
    else:
        da_max_depths = [da.depth.max().values for da in list_das]
        y_limits = (0, np.nanmax(da_max_depths))

    da_max_distances = [da.distance.max().values for da in list_das]
    if da_contours is not None:
        da_max_distances = da_max_distances + [da_contours.distance.max().values]
    x_limits = (0, np.nanmin(da_max_distances))

    if bottom is not None:
        for a in axes:
            a.fill_between(
                bottom_x,
                bottom * 0 + y_limits[1] + 10,
                bottom,
                zorder=999,
                color="gray",
            )
    else:
        print("No bottom data found!")

    # add station ticks
    if add_station_ticks:
        found_stations = False
        i == 0
        while (found_stations == False) & (i < len(list_das)):
            if "station" in list_das[i].coords:
                stations = list_das[i]["station"].to_numpy()
                distances = list_das[i]["distance"].to_numpy()
                found_stations = True
            i += 1
        if (found_stations == False) & (da_contours is not None):
            if "station" in da_contours.coords:
                stations = da_contours["station"].to_numpy()
                distances = da_contours["distance"].to_numpy()
                found_stations = True

        if found_stations:
            for s, d in zip(stations, distances):
                for a in axes:
                    a.text(d, 0, "v", ha="center", fontweight="bold")
                axes[0].annotate(
                    str(s),
                    (d, 0),
                    xytext=(0, 10),
                    textcoords="offset points",
                    ha="center",
                )
        else:
            print("Station ticks not possible when only VM-ADCP data is provided!")

    for a in axes:
        a.set_xlim(x_limits)
        a.set_ylim(y_limits)
        a.invert_yaxis()
        a.yaxis.set_ticks_position("both")

    axes[-1].set_xlabel("Distance [km]")

    return fig, axes, pics


def plot_CTD_station(CTD, station, axes=None, add=False, linestyle="-"):
    """
    Plots the temperature and salinity profile of a single station.
    Parameters
    ----------
    CTD : str or dict
        Either a dict of dicts containing the CTD data, which can be made with
              the function read_CTD. Or a str with a file where the dict is stored
    station : number
        Number which station to plot (must be in the CTD data!).

    ax: (2,) array-like
        List of two axes, the first one being the axes for temperature,
        and the second one for Salinity
    add : bool, optional, depracated
        Switch whether to add the plot to a figure (True), or to create a
        new figure for the plot (False). The default is True. This parameter
        is depracated, which means that it doesn't have any effect anymore.
    Returns
    -------
    None.
    """
    # Check if the function has data to work with
    assert type(CTD) in [dict, str], (
        "Parameter *CTD*: You must provide either\n"
        " a) a data dict or \n b) a npy file string with the data !"
    )

    # read in the data (only needed if no CTD-dict, but a file was given)
    if type(CTD) is str:
        print("reading file...")
        CTD = np.load(CTD, allow_pickle=True).item()

    # Check if all stations given are found in the data
    assert np.isin(station, list(CTD.keys())), (
        "The station was not found in "
        "the CTD data! \n The following stations are in the data: "
        + "".join([str(st) + " " for st in CTD.keys()])
    )

    # end of checks.

    # select station
    CTD = CTD[station]

    if axes == None:
        ax = plt.gca()
        ax2 = ax.twiny()
        ax.invert_yaxis()
    else:
        assert len(axes) == 2, "You need to provide a list of two axes"
        ax = axes[0]
        ax2 = axes[1]

    # plot
    ax.plot(CTD["CT"], -CTD["z"], "r", linestyle=linestyle)
    ax.set_xlabel("Conservative temperature [˚C]", color="r")
    ax.set_ylabel("Depth [m]")
    ax.spines["bottom"].set_color("r")
    ax.tick_params(axis="x", colors="r")

    ax2.plot(CTD["SA"], -CTD["z"], "b", linestyle=linestyle)
    ax2.set_xlabel("Absolute salinity [g / kg]", color="b")
    ax2.tick_params(axis="x", colors="b")
    plt.tight_layout()

    return ax, ax2


def plot_CTD_map(
    CTD,
    stations=None,
    topography=None,
    extent=None,
    depth_contours=[10, 50, 100, 150, 200, 300, 400, 500, 1000, 2000, 3000, 4000, 5000],
    st_labels="",
    adjust_text=False,
):
    """
    Function which plots a very basic map of selected CTD stations.
    Parameters
    ----------
    CTD : dict
        Dictionary containing the CTD data.
    stations : array_like, optional
        The positions to put on the map. The default is all stations.
    topography : str or array-like, optional
        Either a file or an array with topography data.
        If topography is given in a file, three filetypes are supported:
            - .nc, in that case the file should contain the variables
              'lat', 'lon', and 'z'
            - .mat, in that case the file should contain the variables
              'lat', 'lon', and 'D'
            - .npy, in that case the file should contain an array with
              lat, lon and elevation as columns (and total size 3 x lon x lat)
        If topography is given as an array, it should be an array with
              lat, lon and elevation as columns (and total size 3 x lon x lat)
        The default is None, then no bathymetry will be plotted (only coasts).
    extent : (4,) array_like, optional
        List of map extent. Must be given as [lon0,lon1,lat0,lat1].
        The default is None.
    depth_contours : array_like, optional
        A list containing contour levels for the bathymetry. The default is
        [10,50,100,150,200,300,400,500,1000,2000,3000,4000,5000].
    adjust_text : bool, optional
        Whether to adjust the station names so they don't overlap. Default is
        True.
    Returns
    -------
    None.
    """

    assert type(st_labels) in [str, list, tuple], (
        "st_labels must either be" "a string, a tuple or a list."
    )
    # if no stations are provided, just plot all stations
    if stations is None:
        stations = CTD.keys()

    # select only stations
    CTD = {key: CTD[key] for key in stations}
    lat = [value["LAT"] for value in CTD.values()]
    lon = [value["LON"] for value in CTD.values()]
    std_lat, std_lon = np.std(lat), np.std(lon)
    lon_range = [min(lon) - std_lon, max(lon) + std_lon]
    lat_range = [min(lat) - std_lat, max(lat) + std_lat]

    ax = plt.axes(projection=ccrs.PlateCarree())
    if extent is None:
        extent = [lon_range[0], lon_range[1], lat_range[0], lat_range[1]]
    ax.set_extent(extent)

    if topography is not None:
        if type(topography) is str:
            ext = topography.split(".")[-1]
            if ext == "mat":
                topo = loadmat(topography)
                topo_lat, topo_lon, topo_z = topo["lat"], topo["lon"], topo["D"]
            elif ext == "npy":
                topo = np.load(topography)
                topo_lat, topo_lon, topo_z = topo[0], topo[1], topo[2]
            elif ext == "nc":
                topo = Dataset(topography)
                topo_lat, topo_lon, topo_z = (
                    topo.variables["lat"][:],
                    topo.variables["lon"][:],
                    topo.variables["z"][:],
                )
                if len(topo_lon.shape) == 1:
                    topo_lon, topo_lat = np.meshgrid(topo_lon, topo_lat)
            else:
                assert False, "Unknown topography file extension!"
        else:  # assume topography is array with 3 columns (lat,lon,z)
            topo_lat, topo_lon, topo_z = topography[0], topography[1], topography[2]

        topo_z[topo_z < -1] = -1  # discard elevation above sea level

        BC = ax.contour(
            topo_lon,
            topo_lat,
            topo_z,
            colors="lightblue",
            levels=depth_contours,
            linewidths=0.3,
            transform=ccrs.PlateCarree(),
        )
        clabels = ax.clabel(BC, depth_contours, fontsize=4, fmt="%i")
        print(clabels)
        if clabels is not None:
            for txt in clabels:
                txt.set_bbox(dict(facecolor="none", edgecolor="none", pad=0, alpha=0.0))
        ax.contour(topo_lon, topo_lat, topo_z, levels=[0], colors="k", linewidths=0.5)
        ax.contourf(
            topo_lon, topo_lat, topo_z, levels=[-1, 1], colors=["lightgray", "white"]
        )
    else:  # if no topography is provided
        ax.add_feature(
            cartopy.feature.GSHHSFeature(
                scale="auto", facecolor="lightgray", linewidth=0.5
            )
        )

    # add the points, and add labels
    if type(st_labels) == str:
        st_texts = [st_labels + str(s) for s in stations]
    else:
        st_texts = st_labels

    ax.plot(lon, lat, "xr", transform=ccrs.PlateCarree())
    texts = []
    for i, station in enumerate(stations):
        if extent[0] < lon[i] < extent[1] and extent[2] < lat[i] < extent[3]:
            texts.append(
                ax.text(
                    lon[i],
                    lat[i],
                    st_texts[i],
                    horizontalalignment="center",
                    verticalalignment="bottom",
                )
            )

    gl = ax.gridlines(
        crs=ccrs.PlateCarree(),
        draw_labels=True,
        linewidth=1,
        color="gray",
        alpha=0.5,
        linestyle="--",
    )
    gl.xformatter = LONGITUDE_FORMATTER
    gl.yformatter = LATITUDE_FORMATTER

    # make sure aspect ration of the axes is not too extreme
    ax.set_aspect("auto")
    if adjust_text:
        adj_txt(
            texts,
            expand_text=(1.2, 1.6),
            arrowprops=dict(arrowstyle="-", color="black"),
            ax=ax,
        )
    plt.gcf().canvas.draw()
    plt.tight_layout()


def plot_empty_map(
    extent,
    topography=None,
    depth_contours=[10, 50, 100, 150, 200, 300, 400, 500, 1000, 2000, 3000, 4000, 5000],
):
    """
    Function which plots a very basic map of selected CTD stations.
    Parameters
    ----------
    extent : (4,) array_like
        List of map extent. Must be given as [lon0,lon1,lat0,lat1].
    topography : str or array-like, optional
        Either a file or an array with topography data.
        If topography is given in a file, three filetypes are supported:
            - .nc, in that case the file should contain the variables
              'lat', 'lon', and 'z'
            - .mat, in that case the file should contain the variables
              'lat', 'lon', and 'D'
            - .npy, in that case the file should contain an array with
              lat, lon and elevation as columns (and total size 3 x lon x lat)
        If topography is given as an array, it should be an array with
              lat, lon and elevation as columns (and total size 3 x lon x lat)
        The default is None, then no bathymetry
        will be plotted.
    depth_contours : array_like, optional
        A list containing contour levels for the bathymetry. The default is
        [10,50,100,150,200,300,400,500,1000,2000,3000,4000,5000].
    Returns
    -------
    None.
    """

    fig, ax = plt.subplots(1, 1, subplot_kw={"projection": ccrs.PlateCarree()})
    ax.set_extent(extent)
    if topography is not None:
        if type(topography) is str:
            ext = topography.split(".")[-1]
            if ext == "mat":
                topo = loadmat(topography)
                topo_lat, topo_lon, topo_z = topo["lat"], topo["lon"], topo["D"]
            elif ext == "npy":
                topo = np.load(topography)
                topo_lat, topo_lon, topo_z = topo[0], topo[1], topo[2]
            elif ext == "nc":
                topo = Dataset(topography)
                topo_lat, topo_lon, topo_z = (
                    topo.variables["lat"][:],
                    topo.variables["lon"][:],
                    topo.variables["z"][:],
                )
                if len(topo_lon.shape) == 1:
                    topo_lon, topo_lat = np.meshgrid(topo_lon, topo_lat)
            else:
                assert False, "Unknown topography file extension!"
        else:  # assume topography is array with 3 columns (lat,lon,z)
            topo_lat, topo_lon, topo_z = topography[0], topography[1], topography[2]

        topo_z[topo_z < -1] = -1  # discard elevation above sea level
        BC = ax.contour(
            topo_lon,
            topo_lat,
            topo_z,
            colors="lightblue",
            levels=depth_contours,
            linewidths=0.3,
            transform=ccrs.PlateCarree(),
        )
        clabels = ax.clabel(BC, depth_contours, fontsize=4, fmt="%i")
        print(clabels)
        if clabels is not None:
            for txt in clabels:
                txt.set_bbox(dict(facecolor="none", edgecolor="none", pad=0, alpha=0.0))
        ax.contour(topo_lon, topo_lat, topo_z, levels=[0.1], colors="k", linewidths=0.5)
        ax.contourf(
            topo_lon, topo_lat, topo_z, levels=[-1, 1], colors=["lightgray", "white"]
        )
    else:  # if no topography is provided
        ax.add_feature(
            cartopy.feature.GSHHSFeature(
                scale="auto", facecolor="lightgray", linewidth=0.5
            )
        )

    gl = ax.gridlines(
        crs=ccrs.PlateCarree(),
        draw_labels=True,
        linewidth=1,
        color="gray",
        alpha=0.5,
        linestyle="--",
    )
    gl.xformatter = LONGITUDE_FORMATTER
    gl.yformatter = LATITUDE_FORMATTER

    # make sure aspect ration of the axes is not too extreme
    ax.set_aspect("auto")
    plt.gcf().canvas.draw()
    plt.tight_layout()

    return fig, ax


def plot_CTD_ts(CTD, stations=None, pref=0):
    """
    Plots a TS diagram of selected stations from a CTD dataset.
    Parameters
    ----------
    CTD : dict
        Dictionary containing the CTD data.
    stations : array-like, optional
        The desired stations. The default is all stations in CTD.
    pref : TYPE, optional
        Which reference pressure to use. The following options exist:\n
        0:    0 dbar\n
        1: 1000 dbar\n
        2: 2000 dbar\n
        3: 3000 dbar\n
        4: 4000 dbar\n
        The default is 0.
    Returns
    -------
    None.
    """
    # select only input stations
    if stations is not None:
        CTD = {key: CTD[key] for key in stations}

    max_S = max([np.nanmax(value["SA"]) for value in CTD.values()]) + 0.1
    min_S = min([np.nanmin(value["SA"]) for value in CTD.values()]) - 0.1

    max_T = max([np.nanmax(value["CT"]) for value in CTD.values()]) + 0.5
    min_T = min([np.nanmin(value["CT"]) for value in CTD.values()]) - 0.5

    create_empty_ts((min_T, max_T), (min_S, max_S), p_ref=pref)

    # Plot the data in the empty TS-diagram
    for station in CTD.values():
        plt.plot(
            station["SA"],
            station["CT"],
            linestyle="none",
            marker=".",
            label=station["unis_st"],
        )

    if len(CTD.keys()) > 1:
        plt.legend(ncol=2, framealpha=1, columnspacing=0.7, handletextpad=0.4)

    return


def create_empty_ts(T_extent, S_extent, p_ref=0):
    """
    Creates an empty TS-diagram to plot data into.
    Parameters
    ----------
    T_extent : (2,) array_like
        The minimum and maximum conservative temperature.
    S_extent : (2,) array_like
        The minimum and maximum absolute salinity.
    p_ref : int, optional
        Which reference pressure to use. The following options exist:\n
        0:    0 dbar\n
        1: 1000 dbar\n
        2: 2000 dbar\n
        3: 3000 dbar\n
        4: 4000 dbar\n
        The default is 0.
    Returns
    -------
    None.
    """

    sigma_functions = [gsw.sigma0, gsw.sigma1, gsw.sigma2, gsw.sigma3, gsw.sigma4]
    T = np.linspace(T_extent[0], T_extent[1], 100)
    S = np.linspace(S_extent[0], S_extent[1], 100)

    T, S = np.meshgrid(T, S)

    SIGMA = sigma_functions[p_ref](S, T)

    cs = plt.contour(S, T, SIGMA, colors="k", linestyles="--")
    plt.clabel(cs, fmt="%1.1f")

    plt.ylabel("Conservative Temperature [°C]")
    plt.xlabel("Absolute Salinity [g kg$^{-1}$]")
    plt.title("$\Theta$ - $S_A$ Diagram")
    if p_ref > 0:
        plt.title("Density: $\sigma_{" + str(p_ref) + "}$", loc="left", fontsize=10)

    return


def check_VM_ADCP_map(ds):
    """
    Small function to produce an interactive map of the VM-ADCP measurements. This can be used to easily determine the times to use in a section plot (see example notebook!).

    Parameters
    ----------
    ds : xarray dataset
        Either from Codas or WinADCP.

    Returns
    -------
    None.

    """

    df = ds[["time", "lat", "lon"]].to_pandas()
    df["time"] = df.index

    fig = px.scatter_mapbox(df, lon="lon", lat="lat", hover_data="time")
    fig.update_layout(mapbox_style="open-street-map"),
    pplot(fig)

    return


def plot_tidal_spectrum(data, constituents=["M2"]):
    """
    Function to plot a tidal spectrum and add indicators for a set of tidal frequencies. A complete list of all available tidal constituents can be printed with 'print(list(uptide.tidal.omega.keys()))'.

    Parameters
    ----------
    data : pd.Series
        time Series of spectral data (output of the function 'calculate_tidal_spectrum')
    constituents : list, optional
        List with constituents to add to the plot. Default is ['M2'].

    Returns
    -------
        fig, ax : Handles of the created figure.
    """

    tidal_freqs = np.array([uptide.tidal.omega[c] for c in constituents])

    fig, ax = plt.subplots(1, 1, figsize=(10, 5))
    data.plot(ax=ax, color="b", zorder=10)
    ax.grid()
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xticks((24.0 * 3600.0) / ((2.0 * np.pi) / tidal_freqs))
    ax.set_xticklabels(constituents)
    ax.minorticks_off()
    ax.set_xlim(right=data.index[-1])
    ax.set_ylabel("power spectral density")

    return fig, ax


def plot_map_tidal_ellipses(
    amp_major,
    amp_minor,
    inclin,
    theta,
    constituents,
    lat_center=78.122,
    lon_center=14.26,
    map_extent=[11.0, 16.0, 78.0, 78.3],
    topography=None,
):
    """
    Function to plot tidal ellipses on a map.
    Parameters
    ----------
    amp_major : array
        Amplitudes along the major axis, one element for each specified tidal constituent (see below).
    amp_minor : array
        Amplitudes along the minor axis, one element for each specified tidal constituent (see below).
    inclin : array
        Inclination of the ellipses, one element for each specified tidal constituent (see below).
    theta : array
        Phase of the maximum current, one element for each specified tidal constituent (see below).
    constituents : list
        List with the names of the constituent, for the legend.
    lat_center : float, optional
        Center position for the ellipses. Typically the position of the mooring that measured the data. Default is the approximate position of IS-E.
    lon_center : float, optional
        Center position for the ellipses. Typically the position of the mooring that measured the data. Default is the approximate position of IS-E.
    map_extent : list, optional
        List with order lon_min, lon_max, lat_min, lat_max. Specifies the area limits to plot on the map.
    topography : str or array-like, optional
        Either a file or an array with topography data.
        If topography is given in a file, three filetypes are supported:
            - .nc, in that case the file should contain the variables
              'lat', 'lon', and 'z'
            - .mat, in that case the file should contain the variables
              'lat', 'lon', and 'D'
            - .npy, in that case the file should contain an array with
              lat, lon and elevation as columns (and total size 3 x lon x lat)
        If topography is given as an array, it should be an array with
              lat, lon and elevation as columns (and total size 3 x lon x lat)
        The default is None, then no bathymetry
        will be plotted.

    Returns
    -------
        fig, ax_map, ellipse_inset : Handles of the created figure.
    """

    phi = np.linspace(0, 2 * np.pi, 1000)

    fig, ax_map = plot_empty_map(extent=map_extent, topography=topography)

    inset_size = 0.3

    """x, y = ax_map.projection.transform_point(lon_center, lat_center, ccrs.PlateCarree())
    data2axes = (ax_map.transAxes + ax_map.transData.inverted()).inverted()
    xp, yp = data2axes.transform((x, y))
    ip = InsetPosition(ax_map, [xp - inset_size / 2, yp - inset_size / 2, inset_size, inset_size])
    ellipse_inset = fig.add_axes((0, 0, 1, 1))
    ellipse_inset.set_axes_locator(ip)"""
    ellipse_inset = inset_axes(
        ax_map,
        width=inset_size,
        height=inset_size,
        loc="center",
        bbox_to_anchor=(lon_center, lat_center),
        bbox_transform=ax_map.transData,
    )  # should work, not sure, didn't test
    ellipse_inset.axis("off")
    ellipse_inset.set_facecolor("none")
    ellipse_inset.tick_params(labelleft=False, labelbottom=False)
    ellipse_inset.grid(False)
    ellipse_inset.set_aspect(1.0)

    for i, (a, b, t, g) in enumerate(zip(amp_major, amp_minor, inclin, theta)):
        E = np.array([a * np.cos(phi), b * np.sin(phi)])
        R_rot = np.squeeze(np.array([[np.cos(t), -np.sin(t)], [np.sin(t), np.cos(t)]]))
        E_rot = np.zeros((2, E.shape[1]))
        for j in range(E.shape[1]):
            E_rot[:, j] = np.dot(R_rot, E[:, j])

        ellipse_inset.plot(E_rot[0, :], E_rot[1, :], c=f"C{i}", label=constituents[i])

        ind = np.where(abs(t - g) == np.nanmin(abs(t - g)))[0][0]

        ellipse_inset.annotate(
            "",
            xy=(E_rot[0, ind], E_rot[1, ind]),
            xycoords="data",
            xytext=(0.0, 0.0),
            textcoords="data",
            arrowprops=dict(arrowstyle="-|>", connectionstyle="arc3", color=f"C{i}"),
        )

        ellipse_inset.legend(
            ncol=len(constituents), bbox_to_anchor=(1.65, 3.85), loc="upper right"
        )

    return fig, ax_map, ellipse_inset


############################################################################
# PORTASAL
############################################################################


def play_tone(
    frequency: float, duration: float = 0.5, samplerate: float = 44100
) -> None:
    """Play a tone at a given frequency and duration using sounddevice.

    Args:
        frequency (float): Frequency of the tone in Hz.
        duration (float, optional): Duration of the tone in seconds. Defaults to 0.5.
        samplerate (float, optional): Sampling rate in Hz. Defaults to 44100.

    Returns:
        None
    """
    t: NDArray = np.linspace(0, duration, int(samplerate * duration), endpoint=False)
    waveform: NDArray = np.sin(2 * np.pi * frequency * t)
    sd.play(waveform, samplerate)
    sd.wait()


def portasal(
    number: int,
    n_flushing: int = 3,
    t_flushing: float = 23,
    n_measure: int = 3,
    t_measure: float = 24,
) -> None:
    """Function to run in the backrground to get audio feedback for the different
    steps of the protocol. The function will play a sound at the beginning of each
    flushing and measurement. The sound will be played at different frequencies.
    You can stop by write "ende" in the console.

    Args:
        number (_type_): Number of bottles to be sampled.
        n_flushing (int, optional): Number of flushing before the sampling starts. Defaults to 3.
        t_flushing (int, optional): The time it takes from flushing to fill up the tubes. Defaults to 23.
            - In seconds.
        n_measure (int, optional): Number of measurements done per bootle. Defaults to 3.
        t_measure (int, optional): The time it takes to do one measurment. Defaults to 24.
            - In seconds.

    Returns:
        None
    """
    run: int = 0
    f_flushing: float = 200
    f_measure: float = 400
    f_stdby: float = 600
    f_final: float = 800
    while number > run:
        if input("Press Enter if you start the sampling process.").lower() == "ende":
            break
        for i in range(n_flushing):
            if i == 0:
                time.sleep(t_flushing * 1.3)
            else:
                time.sleep(t_flushing)
            play_tone(f_flushing)
            print(datetime.now().strftime("%H:%M:%S"), f"{i+1}. flush", sep="\t")
        for i in range(n_measure):
            time.sleep(t_flushing)
            play_tone(f_measure)
            print(
                datetime.now().strftime("%H:%M:%S"), f"Start measurment {i}.", sep="\t"
            )
            if i == 0:
                if input("Press Enter if you start the measurments.") == "ende":
                    break
            time.sleep(t_measure)
            if i != n_measure - 1:
                play_tone(f_stdby)
                print(
                    datetime.now().strftime("%H:%M:%S"),
                    f"End of measurment {i}",
                    sep="\t",
                )
            else:
                play_tone(f_final)
        print(
            datetime.now().strftime("%H:%M:%S"),
            f"Finished the {run}. bottle.",
            sep="\t",
        )
        run += 1
    return None


# to be removed:
if None:
    data_path = "/Users/pselle/Library/CloudStorage/OneDrive-UniversitetssenteretpåSvalbardAS/Svalbard/Groupwork/AGF-214/Mooring/DATA"
    dict_mooring_0 = {
        33: read_Seaguard(f"{data_path}/SeaGuard/RCM_2375_20231002_1800/2375.txt"),
        34: read_SBE37(f"{data_path}/SBE37/37-SM_03723000_2024_09_25.cnv"),
        42: read_Minilog(
            f"{data_path}/Minilog_II_T/Minilog-II-T_358949_20240925_1.csv"
        ),
        52: read_RBR(f"{data_path}/RBRconcerto/206125_20240925_2028.rsk"),
        67: read_RBR(f"{data_path}/RBRsolo/205993_20240925_2018.rsk"),
        77: read_RBR(f"{data_path}/RBRconcerto/206124_20240925_2031.rsk"),
        87: read_Minilog(
            f"{data_path}/Minilog_II_T/Minilog-II-T_358953_20240925_1.csv"
        ),
        97: read_Seaguard(f"{data_path}/SeaGuard/RCM_2370_20231002_1800/2370.txt"),
        98: read_SBE37(f"{data_path}/SBE37/37-SM_03722999_2024_09_25.cnv"),
        107: read_Minilog(
            f"{data_path}/Minilog_II_T/Minilog-II-T_358948_20240925_1.csv"
        ),
        127: read_RBR(f"{data_path}/RBRconcerto/206127_20240925_2023.rsk"),
        156: read_Minilog(
            f"{data_path}/Minilog_II_T/Minilog-II-T_358945_20240925_1.csv"
        ),
        160: read_SBE37(f"{data_path}/SBE37/37-SM_03723003_2024_09_25.cnv"),
        161: read_Seaguard(f"{data_path}/SeaGuard/RCM_2282_20231002_1800/2282.txt"),
        # Depth on paper : read pd.Dataframe from Data,
    }

    ds_mooring_0 = mooring_into_xarray(dict_mooring_0)
