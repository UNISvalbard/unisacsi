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
from _collections_abc import dict_keys

import matplotlib.axes
from numpy._typing._array_like import NDArray
import warnings

warnings.filterwarnings(
    "ignore",
    message="pkg_resources is deprecated as an API. See https://setuptools.pypa.io/en/latest/pkg_resources.html.",
    category=UserWarning,
)
from . import universal_func as uf
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

from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import utide
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
from typing import Literal, Any, get_args, overload
import logging
import sounddevice as sd
from collections import Counter
from itertools import chain
import copy

import ephem as eph


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

    if not (pd.api.types.is_list_like(u) or isinstance(u, num.Real)):
        raise TypeError(
            f"'u' should be numeric or array_like, not a {type(u).__name__}."
        )
    if not (pd.api.types.is_list_like(v) or isinstance(v, num.Real)):
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

    if not (pd.api.types.is_list_like(angle) or isinstance(angle, num.Real)):
        raise TypeError(
            f"'u' should be numeric or array_like, not a {type(angle).__name__}."
        )
    if not (pd.api.types.is_list_like(speed) or isinstance(speed, num.Real)):
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
    maxdepth: float = np.nanmax([np.nanmax(-CTD[i]["z [m]"]) for i in stations])
    mindepth: float = np.nanmin([np.nanmin(-CTD[i]["z [m]"]) for i in stations])
    if z_fine:
        Z: np.ndarray = np.linspace(
            mindepth, maxdepth, int((maxdepth - mindepth) * 10) + 1
        )
    else:
        Z = np.linspace(mindepth, maxdepth, int(maxdepth - mindepth) + 1)

    # construct the X-vector, either distance or time
    if x_type == "distance":
        LAT: np.ndarray = np.asarray([d["lat"] for d in CTD.values()])
        LON: np.ndarray = np.asarray([d["lon"] for d in CTD.values()])
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
                        interp1d(-value["z [m]"], value[field], bounds_error=False)(Z)
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

    def _all_convertible_to_float(arr) -> bool:
        try:
            data = arr.compressed()
            np.array(data, dtype=float)
            return True
        except (ValueError, TypeError):
            return False

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
    maxdepth: np.ndarray = np.nanmax([np.nanmax(-CTD[i]["z [m]"]) for i in stations])
    mindepth: np.ndarray = np.nanmin([np.nanmin(-CTD[i]["z [m]"]) for i in stations])

    Z: np.ndarray = np.linspace(mindepth, maxdepth, int(maxdepth - mindepth) + 1)

    # collect station numbers and other metadata
    ship_station: np.ndarray = np.array([d["st"] for d in CTD.values()])
    station: np.ndarray = np.array([d["unis_st"] for d in CTD.values()])
    lat: np.ndarray = np.array([d["lat"] for d in CTD.values()])
    lon: np.ndarray = np.array([d["lon"] for d in CTD.values()])
    bdepth: np.ndarray = np.array([d["BottomDepth [m]"] for d in CTD.values()])

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
    interset: set[str] = (
        set()
    )  # every var, where values where assignet to the nearest grid point
    for field in fields:
        # grid over Z
        temp_array: list = []
        for value in CTD.values():
            if field in value:
                if _all_convertible_to_float(value[field]):
                    temp_array.append(
                        interp1d(-value["z [m]"], value[field], bounds_error=False)(Z)
                    )
                else:
                    nearest_indices = np.abs(Z[:, None] + value["z [m]"]).argmin(axis=0)
                    Z_labels: NDArray = np.full_like(Z, fill_value=np.nan, dtype=object)
                    for label, idx in zip(value[field], nearest_indices):
                        Z_labels[idx] = label  # can override!!
                    temp_array.append(Z_labels)
                    interset.add(field)
            else:
                temp_array.append(interp1d(Z, Z * np.nan, bounds_error=False)(Z))
        temp_array = np.array(temp_array).transpose()

        fCTD[field] = temp_array

        if field == "water_mass":
            fCTD["water_mass"] = np.round(fCTD["water_mass"])

    if len(interset):
        logging.info(
            f"The following variables were assigned to the nearest grid point: {[name for name in interset]}\nIf two values are close to a grid point, the lower one is selected."
        )

    list_da: list = []
    for vari in fCTD.keys():
        unit_match: uf.Match[str] | None = re.match(r"^(.*?)\s*\[(.*?)\]$", vari)
        vari_name: str = unit_match.group(1) if unit_match else vari
        unit: str = unit_match.group(2) if unit_match else ""
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
                name=vari_name,
                attrs={"units": unit, "orignal_name": vari},
            )
        )

    ds: xr.Dataset = xr.merge(list_da)

    ds = ds.sortby("time")
    ds = ds.interp(depth=np.arange(np.ceil(ds.depth[0]), np.floor(ds.depth[-1]) + 1.0))

    if switch_xdim == "station":
        ds = ds.swap_dims({"time": "station"})

    ds["SA"].attrs["long_name"] = "Absolute Salinity"
    ds["S"].attrs["long_name"] = "Salinity"
    ds["CT"].attrs["long_name"] = "Conservative Temperature"
    ds["T"].attrs["long_name"] = "Temperature"
    ds["C"].attrs["long_name"] = "Conductivity"
    ds["P"].attrs["long_name"] = "Pressure"
    ds["SIGTH"].attrs["long_name"] = "Density (sigma-theta)"
    ds["OX"].attrs["long_name"] = "Oxygen"

    ds.attrs = {"source": "CTD_to_xarray"}

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
        if len(ds.station.values) != len(np.unique(ds.station.values)):
            logging.warning(
                "There are duplicate station numbers in the dataset. Using the first occurrence of each station."
            )
            ds = ds.drop_duplicates("station", keep="first")
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


def mooring_to_xarray(
    dict_of_instr: dict[pd.DataFrame],
    transfer_vars: list[str] = ["T", "S", "SIGTH", "U", "V", "OX", "P"],
) -> xr.Dataset:
    """Function to store mooring data from a mooring in an xarray dataset.
    The returned dataset can be regridded onto a regular time/depth grid using the xarray methods interpolate_na and interp.

    Args:
        dict_of_instr (dict[pd.DataFrame]): Dictionary with the dataframes returned from the respective read functions for the different instruments, keys: depth levels.
        transfer_vars (list[str], optional): Variables to read into the dataset. Defaults to ["T", "S", "SIGTH", "U", "V", "OX", "P"].
            - If there are different units for one Variable, uses most offen used unit.

    Returns:
        xr.Dataset: Dataset with two dimensions depth and time, and the variables from transfer_vars (Units are striped).
    """

    if not isinstance(dict_of_instr, dict):
        raise TypeError(
            f"'dict_of_instr' should be a dict, not a {type(dict_of_instr).__name__}."
        )
    for k, v in dict_of_instr.items():
        if not isinstance(v, pd.DataFrame):
            raise TypeError(f"'{k}' in dict_of_instr should be a pd.DataFrame.")

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
    for d, df in dict_of_instr.items():
        varis_instr: list[str] = [
            v
            for v in df.columns
            if (m := re.match(r"^(.*?)\s*(?:\[[^\]]*\])?$", v))
            and m.group(1) in transfer_vars
        ]
        dict_vars[d] = varis_instr

    # check for vars with diff units
    all_vars: list[str] = list(chain.from_iterable(dict_vars.values()))
    var_to_full: dict[str, list[str]] = {}
    for v in all_vars:
        if short := re.match(r"^(.*?)\s*(?:\[[^\]]*\])?$", v):
            var_to_full.setdefault(short.group(1), []).append(v)
        else:
            logging.warning(f"Warning: Couldn't make sense of '{v}'. Skipping it.")

    final_vars: dict[str, str] = {}
    for var, reps in var_to_full.items():
        if len(reps) > 1:
            counts = Counter([v for d in dict_vars.values() for v in d if v in reps])
            final_vars[var] = counts.most_common(1)[0][0]
        else:
            final_vars[var] = reps[0]

    list_da: list[xr.DataArray] = []
    for short_var, full_var in final_vars.items():
        list_df: list[pd.DataFrame] = []
        for d, df_instr in dict_of_instr.items():
            if full_var in list(df_instr.keys()):
                list_df.append(df_instr[full_var].rename(d))
        if len(list_df) == 0:  # to continue if no data is available
            logging.warning(f"No data for '{short_var}' in the mooring data.")
            df_vari: pd.DataFrame = pd.DataFrame(index=pd.DatetimeIndex([]))
        else:
            df_vari = pd.concat(list_df, axis=1)
        df_vari = df_vari.resample("20min").mean()

        unit_match = re.match(r"^(.*?)\s*\[([^\]]+)\]$", full_var)
        unit: str = unit_match.group(2) if unit_match else ""

        list_da.append(
            xr.DataArray(
                data=df_vari,
                dims=["time", "depth"],
                coords={
                    "depth": np.array(list(df_vari.columns), dtype=float),
                    "time": df_vari.index.values,
                },
                name=short_var,
                attrs={"units": unit, "orignal_name": full_var},
            )
        )

    ds: xr.Dataset = xr.merge(list_da)
    ds.attrs = {"source": "mooring_to_xarry"}

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
            f"'output' should be 'pd.DataFrame', 'csv', 'df_func', 'None' or a path, not {output}."
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
    CTD: dict, water_mass_def: pd.DataFrame, stations: npt.ArrayLike = None
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
        CTD[s]["water_mass"] = np.ones_like(CTD[s]["T [degC]"]) * np.nan
        CTD[s]["water_mass_Abbr"] = np.empty_like(CTD[s]["T [degC]"], dtype="object")
        for index, row in water_mass_def.iterrows():
            if row["Abbr"] != "ArW":
                ind = np.all(
                    np.array(
                        [
                            CTD[s]["T [degC]"] > row["T_min"],
                            CTD[s]["T [degC]"] <= row["T_max"],
                            CTD[s]["S []"] > row["S_psu_min"],
                            CTD[s]["S []"] <= row["S_psu_max"],
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
        pd.Timestamp(
            year=2000 + int(y),
            month=int(m),
            day=int(d),
            hour=int(H),
            minute=int(M),
            second=int(s),
        )
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

    if not pd.api.types.is_list_like(stations) and stations != None:
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

        p["lat"] = p.pop("LATITUDE")
        p["lon"] = p.pop("LONGITUDE")
        p["z [m]"] = gsw.z_from_p(p["P [dbar]"], p["lat"])
        p["BottomDepth [m]"] = np.round(np.nanmax(np.abs(p["z [m]"])) + 8)
        if np.nanmin(p["C [S/m]"]) > 10.0:
            p["C [S/m]"] /= 10.0
        p["C [S/m]"][p["C [S/m]"] < 1] = np.nan
        p["T [degC]"][p["T [degC]"] < -2] = np.nan
        p["S []"] = salt_corr[0] * p["S []"] + salt_corr[1]  # apply correction
        p["S []"][p["S []"] < 20] = np.nan
        p["C [S/m]"][p["S []"] < 20] = np.nan
        p["SA [g/kg]"] = gsw.SA_from_SP(p["S []"], p["P [dbar]"], p["lon"], p["lat"])
        p["CT [degC]"] = gsw.CT_from_t(p["SA [g/kg]"], p["T [degC]"], p["P [dbar]"])
        p["SIGTH [kg/m^3]"] = gsw.sigma0(p["SA [g/kg]"], p["CT [degC]"])
        if p["filename"].split(".")[0].split("_")[0][-4::].isdigit():
            p["st"] = int(p["filename"].split(".")[0].split("_")[0][-4::])
        p["unis_st"] = unis_station.split("_")[0]
        if "OX [ml/l]" in p:
            p["OX [ml/l]"] = oxy_corr[0] * p["OX [ml/l]"] + oxy_corr[1]
        CTD_dict[unis_station] = p

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
    Standard variable names and convention are used (e.g. p [dbar], S []).

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
    Standard variable names and convention are used (e.g. p [dbar], S []).

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
        df["p [dbar]"] = (df["p [kPa]"] / 10.0) - 10.0

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
    Standard variable names and convention are used (e.g. p [dbar], S []).

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
    Standard variable names and convention are used (e.g. p [dbar], S []).

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
        filepath, sep=r"\s+", header=None, names=["RECORD", "date", "time", "P", "T"]
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
    Standard variable names and convention are used (e.g. p [dbar], S []).

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
    Standard variable names and convention are used (e.g. p [dbar], S []).

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


def read_RCM7(
    filepath: str,
    column_rename: dict[str, str] = {
        "U": "[cm/s]",
        "V": "v [cm/s]",
        "F": "Speed [cm/s]",
        "A": "Dir [deg]",
    },
) -> pd.DataFrame:
    """Reads data from one data file from a RCM7 current meter.
    Standard variable names and convention are used (e.g. p [dbar], S []).

    Args:
        filepath (str): Path to the .lst file.
        column_rename (dict, optional): Dictionary to rename the columns. Defaults to {"U": "[cm/s]", "V": "v [cm/s]", "F": "Speed [cm/s]", "A": "Dir [deg]"}.

    Returns:
        pd.DataFrame: Dataframe with time as index and the individual variables as columns.
    """
    if not isinstance(filepath, str):
        raise TypeError(
            f"'filepath' should be a string, not a {type(filepath).__name__}."
        )
    if not filepath.endswith(".lst"):
        raise ValueError(f"Invalid file format: {filepath}. Expected a .lst file.")
    if not os.path.isfile(filepath):
        raise FileNotFoundError(f"File not found: {filepath}.")

    df = pd.read_csv(filepath, header=1, sep=r"\s+", skip_blank_lines=True)
    if all(col in df.columns for col in ["%Y", "MM", "DD", "hh", "mm"]):
        df["TIMESTAMP"] = pd.to_datetime(
            df["%Y"].astype(str)
            + "-"
            + df["MM"].astype(str).str.zfill(2)
            + "-"
            + df["DD"].astype(str).str.zfill(2)
            + " "
            + df["hh"].astype(str).str.zfill(2)
            + ":"
            + df["mm"].astype(str).str.zfill(2),
            format="%y-%m-%d %H:%M",
        )
        df.set_index("TIMESTAMP", inplace=True)
        df.drop(columns=["%Y", "MM", "DD", "hh", "mm"], inplace=True)
    df.sort_index(inplace=True)
    df.rename(
        columns=column_rename,
        inplace=True,
    )

    df = uf.std_names(df, add_units=True, module="o")

    return df


############################################################################
# TIDE FUNCTIONS
############################################################################


__tidal_models__ = Literal[
    "Arc2kmTM", "AODTM-5", "AOTIM-5", "AOTIM-5-2018", "Arc2kmTM", "Gr1kmTM"
]


def download_tidal_model(
    model: str | __tidal_models__ = "Arc2kmTM",
    outpath: str | pathlib.Path = pathlib.Path.cwd(),
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
    if model not in get_args(__tidal_models__):
        raise ValueError(
            f"'model' should be 'Arc2kmTM', 'AODTM-5', 'AOTIM-5', 'AOTIM-5-2018', 'Arc2kmTM' or 'Gr1kmTM', not {model}."
        )
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


__inter_methods__ = Literal["bilinear", "spline", "linear", "nearest"]


def detide_VMADCP(
    ds: xr.Dataset,
    path_tidal_models: str,
    tidal_model: str | __tidal_models__ = "Arc2kmTM",
    method: str | __inter_methods__ = "spline",
) -> xr.Dataset:
    """Function to correct the VM-ADCP data for the tides
    (substract the tidal currents from the measurements).

    Args:
        ds (xr.Dataset): Data from VM-ADCP.
            - Read and transformed with the respective functions (see example notebook).
        path_tidal_models (str): Path to folder with tidal model data.
            - Don't include the name of the actual tidal model!
        tidal_model (str, optional): Name of the tidal model to be used (also name of the folder where these respective tidal model data are stored). Defaults to "Arc2kmTM".
            - Options are: "AODTM-5", "AOTIM-5", "AOTIM-5-2018", "Arc2kmTM", "Gr1kmTM".
        method (str, optional): Spatial interpolation method. Defaults to "spline".
            - From tidal model grid to actual locations of the ship.
            - Options: 'bilinear', 'spline', 'linear' and 'nearest'.

    Returns:
        xr.Dataset: same xarray dataset as the input, but with additional variables for the tidal currents and the de-tided measurements
    """
    if not isinstance(ds, xr.Dataset):
        raise TypeError(f"'ds' should be an xarray dataset, not a {type(ds).__name__}.")
    if not isinstance(path_tidal_models, str):
        raise TypeError(
            f"'path_tidal_models' should be a string, not a {type(path_tidal_models).__name__}."
        )
    if not os.path.isdir(path_tidal_models):
        raise FileNotFoundError(f"Path to tidal models not found: {path_tidal_models}.")
    if not isinstance(tidal_model, str):
        raise TypeError(
            f"'tidal_model' should be a string, not a {type(tidal_model).__name__}."
        )
    if tidal_model not in get_args(__tidal_models__):
        raise ValueError(
            f"'tidal_model' should be 'Arc2kmTM', 'AODTM-5', 'AOTIM-5', 'AOTIM-5-2018', 'Arc2kmTM' or 'Gr1kmTM', not {tidal_model}."
        )
    if not isinstance(method, str):
        raise TypeError(f"'method' should be a string, not a {type(method).__name__}.")
    if method not in get_args(__inter_methods__):
        raise ValueError(
            f"'method' should be 'bilinear', 'spline', 'linear' or 'nearest', not {method}."
        )

    time: NDArray = (
        (ds.time.to_pandas() - pd.Timestamp(1970, 1, 1, 0, 0, 0)).dt.total_seconds()
    ).values

    tide_uv: dict = pyTMD.compute.tide_currents(
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
    latitude: float,
    longitude: float,
    time: pd.DatetimeIndex,
    path_tidal_models: str,
    tidal_model: str | __tidal_models__ = "Arc2kmTM",
    method: str | __inter_methods__ = "spline",
) -> pd.DataFrame:
    """Function to calculate time series of tidal currents u and v
    as well as the surface elevation change h for a given pair of lat and lon
    (only one position at a time!), based on the specified tidal model.

    Args:
        latitude (float): Latitude of e.g. the mooring.
        longitude (float): Longitude of e.g. the mooring.
        time (pd.DatetimeIndex): Timestamps of the time series.
            - Needs to be a pd.DatetimeIndex.
            - Use df.index of e.g. the raw SeaGuard data, if you have read the data with the read_Seaguard-function.
        path_tidal_models (str): Path to folder with tidal model data.
            - Don't include the name of the actual tidal model!
        tidal_model (str, optional): Name of the tidal model to be used (also name of the folder where these respective tidal model data are stored). Defaults to "Arc2kmTM".
            - Options are: "AODTM-5", "AOTIM-5", "AOTIM-5-2018", "Arc2kmTM", "Gr1kmTM".
        method (str, optional): Spatial interpolation method. Defaults to "spline".
            - From tidal model grid to actual locations of the ship.
            - Options: 'bilinear', 'spline', 'linear' and 'nearest'.

    Returns:
        pd.DataFrame: Dataframe with time as the index and three columns u, v and h with the tidal current velocities and the surface height anomalies.
    """
    if not isinstance(latitude, (float, int)):
        raise TypeError(
            f"'latitude' should be a float, not a {type(latitude).__name__}."
        )
    if not isinstance(longitude, (float, int)):
        raise TypeError(
            f"'longitude' should be a float, not a {type(longitude).__name__}."
        )
    if not isinstance(time, pd.DatetimeIndex):
        raise TypeError(
            f"'time' should be a pandas DatetimeIndex, not a {type(time).__name__}."
        )
    if not isinstance(path_tidal_models, str):
        raise TypeError(
            f"'path_tidal_models' should be a string, not a {type(path_tidal_models).__name__}."
        )
    if not os.path.isdir(path_tidal_models):
        raise FileNotFoundError(f"Path to tidal models not found: {path_tidal_models}.")
    if not isinstance(tidal_model, str):
        raise TypeError(
            f"'tidal_model' should be a string, not a {type(tidal_model).__name__}."
        )
    if tidal_model not in get_args(__tidal_models__):
        raise ValueError(
            f"'tidal_model' should be 'Arc2kmTM', 'AODTM-5', 'AOTIM-5', 'AOTIM-5-2018', 'Arc2kmTM' or 'Gr1kmTM', not {tidal_model}."
        )
    if not isinstance(method, str):
        raise TypeError(f"'method' should be a string, not a {type(method).__name__}.")
    if method not in get_args(__inter_methods__):
        raise ValueError(
            f"'method' should be 'bilinear', 'spline', 'linear' or 'nearest', not {method}."
        )

    time_model: NDArray = (
        (time - pd.Timestamp(1970, 1, 1, 0, 0, 0)).total_seconds()
    ).values

    tide_uv: dict = pyTMD.compute.tide_currents(
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
    tide_h: NDArray = pyTMD.compute.tide_elevations(
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


class tide:
    @overload
    def __init__(
        t: npt.ArrayLike,
        u: npt.ArrayLike,
        v: npt.ArrayLike,
        lat: float,
        constituents: str | list[str] = "auto",
        add_vari: list[str] = None,
        **args,
    ): ...

    @overload
    def __init__(
        t: npt.ArrayLike,
        p: npt.ArrayLike,
        lat: float,
        constituents: str | list[str] = "auto",
        add_vari: list[str] = None,
        **args,
    ): ...

    @overload
    def __init__(df_const: pd.DataFrame, single_val: dict, orignal: dict): ...

    def __init__(
        self,
        t=None,
        p=None,
        u=None,
        v=None,
        lat=None,
        constituents="auto",
        add_vari=None,
        df_const=None,
        single_val=None,
        orignal=None,
        **args,
    ) -> None:
        """Class to simplefy the work with tides.

        Can be called in 3 different ways:

        1. **using current (u/v) data**
            Analyzes horizontal current velocity components (u and v).

            Args:
                t (array_like): Time index of the data.
                u (array_like): Eastward (u) current data.
                v (array_like): Northward (v) current data.
                lat (float): Latitude in degrees.
                constituents (str | list[str], optional): Tidal constituents to analyze. Defaults to "auto".
                add_vari (list[str], optional): Additional single variables to include. Defaults to None.
                    - Check the `utide.solve` documentation for available options.
                **args: Additional keyword arguments passed to `utide.solve

        2. **Using pressure (p) data**:
            Analyzes pressure time series data.

            Args:
                t (array_like): Time index of the data.
                p (array_like): Pressure data.
                lat (float): Latitude in degrees.
                constituents (str | list[str], optional): Tidal constituents to analyze. Defaults to "auto".
                add_vari (list[str], optional): Additional single variables to include. Defaults to None.
                    - Check the `utide.solve` documentation for available options.
                **args: Additional keyword arguments passed to `utide.solve`.

        3. **From precomputed data**:
            Initialize the object directly from precomputed results.

            Args:
                df_const (pd.DataFrame): DataFrame containing tidal constituent information.
                single_val (dict): Dictionary of single-value metrics (e.g., total variability, mean values).
                orignal (dict): Original output dictionary from `utide.solve`.
        """
        t_ex: bool = False
        p_ex: bool = False
        uv_ex: bool = False
        lat_ex: bool = False
        df_const_ex: bool = False
        single_val_ex: bool = False
        orignal_ex: bool = False
        if t is not None:
            t_ex: bool = True
            if not pd.api.types.is_list_like(t):
                raise TypeError(f"'t' should be array_like, not a {type(t).__name__}.")
        if p is not None:
            p_ex: bool = True
            if not pd.api.types.is_list_like(p):
                raise TypeError(f"'p' should be array_like, not a {type(p).__name__}")
            if len(p) != len(t):
                raise ValueError(
                    f"'p' should have the same length as 't', not ({len(p)},{len(t)})."
                )
        if u is not None and v is not None:
            uv_ex: bool = True
            if not pd.api.types.is_list_like(u):
                raise TypeError(f"'u' should be array_like, not a {type(u).__name__}")
            if len(u) != len(t):
                raise ValueError(
                    f"'u' should have the same length as 't', not ({len(u)},{len(t)})."
                )
            if not pd.api.types.is_list_like(u):
                raise TypeError(f"'u' should be array_like, not a {type(u).__name__}")
            if len(v) != len(t):
                raise ValueError(
                    f"'v' should have the same length as 't', not ({len(v)},{len(t)})."
                )
        if lat is not None:
            lat_ex: bool = True
            if not isinstance(lat, (float, int)):
                raise TypeError(
                    f"'lat' should be a float or int, not a {type(lat).__name__}."
                )
        if constituents != "auto" and not isinstance(constituents, list):
            raise TypeError(
                f"'constituents' should be a list of strings or 'auto', not a {type(constituents).__name__}."
            )
        if isinstance(constituents, list):
            for i in constituents:
                if not isinstance(i, str):
                    raise TypeError(
                        f"Each element in 'constituents' should be a string, not a {type(i).__name__}."
                    )
        if df_const is not None:
            df_const_ex: bool = True
            if not isinstance(df_const, pd.DataFrame):
                raise TypeError(
                    f"'df_const' should be a pandas DataFrame, not a {type(df_const).__name__}."
                )
        if single_val is not None:
            single_val_ex: bool = True
            if not isinstance(single_val, dict):
                raise TypeError(
                    f"'single_val' should be a dictionary, not a {type(single_val).__name__}."
                )
        if orignal is not None:
            orignal_ex: bool = True
            if not isinstance(orignal, dict):
                raise TypeError(
                    f"'orignal' should be a dictionary, not a {type(orignal).__name__}."
                )

        found: bool = False
        if t_ex and p_ex and lat_ex:
            found = True
            args["manual"] = (t, p)
            self.p: npt.ArrayLike = p
            tha: dict = tidal_harmonic_analysis(
                df=None, lat=lat, constituents=constituents, add_vari=add_vari, **args
            )
            self.lat: float = lat
        elif t_ex and uv_ex and lat_ex:
            if found:
                logging.warning(
                    f"Warning: There was p and u/v data detected. Continuing with u/v data."
                )
            found = True
            args["manual"] = (t, u, v)
            self.u: npt.ArrayLike = u
            self.v: npt.ArrayLike = v
            tha = tidal_harmonic_analysis(
                df=None, lat=lat, constituents=constituents, add_vari=add_vari, **args
            )
        if t_ex and lat_ex and found:
            self.constituents: pd.DataFrame = tha.constituents
            self.single_values: dict = tha.single_values
            self.utide: dict = tha.utide
            self.t: pd.DatetimeIndex = t
        if df_const_ex and single_val_ex and orignal_ex:
            if found:
                logging.warning(
                    f"Warning: Data and results were passed. Results were newly generated and the passed ones ignored."
                )
            else:
                self.constituents: pd.DataFrame = df_const
                self.single_values: dict = single_val
                self.utide: dict = orignal
                found = True
        if not found:
            raise ValueError(
                f"No complete data set (with latitude) or results were passed!"
            )

        return None

    def __str__(self) -> str:
        keys: dict_keys[str, Any] = vars(self).keys()
        has_t: bool = "t" in keys
        has_u: bool = "u" in keys
        has_v: bool = "v" in keys
        has_p: bool = "p" in keys
        has_ha: bool = (
            len([i for i in ["constituents", "single_values", "utide"] if i in keys])
            == 3
        )
        has_spec: bool = "spectrum" in keys
        has_recon: bool = "recon" in keys

        outstr: list[str] = []

        if has_t and has_u and has_v and has_p:
            outstr.append(
                f"The object contains velocity and pressure data in the timespan {min(self.t)} to {max(self.t)}."
            )
        elif has_t and has_p:
            outstr.append(
                f"The object contains pressure data in the timespan {min(self.t)} to {max(self.t)}."
            )
        elif has_t and has_u and has_v:
            outstr.append(
                f"The object contains velocity data in the timespan {min(self.t)} to {max(self.t)}."
            )
        else:
            outstr.append("The object does not contain complete data.")

        if has_ha and has_spec and has_recon:
            outstr.append(
                "A tidal harmonic analysis, spectrum and reconstruction was done."
            )
        elif has_ha and has_spec:
            outstr.append("A tidal harmonic analysis and spectrum was done.")
        elif has_ha and has_recon:
            outstr.append("A tidal harmonic analysis and reconstruction was done.")
        elif has_recon and has_spec:
            outstr.append("A spectrum and a reconstruction was done.")
        elif has_ha:
            outstr.append("A tidal harmonic analysis was done.")
        elif has_spec:
            outstr.append("A tidal spectrum was done.")
        elif has_recon:
            outstr.append("A tidal reconstruciton was done.")

        return "\n".join(outstr)

    def __repr__(self) -> str:
        cls_name: str = self.__class__.__name__
        keys: list = list(vars(self).keys())
        return f"<{cls_name}(keys={keys})>"

    def __len__(self) -> int:
        if "t" in vars(self).keys():
            return len(self.t)
        else:
            return 0

    # be careful, but should work
    def __bool__(self) -> bool:
        return True

    def __eq__(self, other) -> bool:
        if not isinstance(other, tide):
            return False
        return self.__dict__.keys() == other.__dict__.keys() and all(
            (
                np.array_equal(self.__dict__[k], other.__dict__[k])
                if isinstance(self.__dict__[k], np.ndarray)
                else self.__dict__[k] == other.__dict__[k]
            )
            for k in self.__dict__
        )

    def __dir__(self) -> list[str]:
        return list(super().__dir__()) + list(vars(self).keys())

    def __getitem__(self, key) -> Any:
        return getattr(self, key)

    def __setitem__(self, key, value) -> None:
        return setattr(self, key, value)

    def __contains__(self, key) -> bool:
        return key in vars(self)

    def add_data(
        self,
        t: npt.ArrayLike = None,
        p: Any = None,
        u: npt.ArrayLike = None,
        v: npt.ArrayLike = None,
    ) -> None:
        """Method to add data to the tide object after initialization.
        If the object was initialized with data, this will overwrite the existing data.
        If set to None, the existing data will not be changed.

        Args:
            t (array_like): Time index of the data. Defaults to None.
            p (array_like): Pressure data. Defaults to None.
            u (array_like): Eastward (u) current data. Defaults to None.
            v (array_like): Northward (v) current data. Defaults to None.

        Returns:
            None

        """
        if t is not None:
            if not pd.api.types.is_list_like(t):
                raise TypeError(f"'t' should be array_like, not a {type(t).__name__}.")
            self.t = t
        if p is not None:
            if not pd.api.types.is_list_like(p):
                raise TypeError(f"'p' should be array_like, not a {type(p).__name__}.")
            self.p = p
        if u is not None:
            if not pd.api.types.is_list_like(u):
                raise TypeError(f"'u' should be array_like, not a {type(u).__name__}.")
            self.u = u
        if v is not None:
            if not pd.api.types.is_list_like(v):
                raise TypeError(f"'v' should be array_like, not a {type(v).__name__}.")
            self.v = v
        return None

    def get_spectrum(self) -> pd.Series:
        """Returns the spectrum of the tide object.

        Returns:
            pd.Series: The spectrum of the tide object.
        """
        if not hasattr(self, "spectrum"):
            raise AttributeError(
                "The tide object does not have a spectrum. Please calculate it first using 'calc_spectrum'."
            )
        return self.spectrum

    def get_constituents(self) -> pd.DataFrame:
        """Returns the tidal constituents DataFrame of the tide object.

        Returns:
            pd.DataFrame: The tidal constituents DataFrame.
        """
        if not hasattr(self, "constituents"):
            raise AttributeError(
                "The tide object does not have tidal constituents. Please perform a tidal harmonic analysis first."
            )
        return self.constituents

    def get_single_values(self) -> dict:
        """Returns the single values dictionary of the tide object.

        Returns:
            dict: The single values dictionary.
        """
        if not hasattr(self, "single_values"):
            raise AttributeError(
                "The tide object does not have single values. Please perform a tidal harmonic analysis first."
            )
        return self.single_values

    def calc_spectrum(self, p: pd.Series = None, bandwidth: int = 8) -> pd.Series:
        """Function to calculate a spectrum for a the tide time series.

        Args:
            p (pd.Series, optional): Time series of data. Defaults to None.
                - Just needed if the object was initialized without.
                - Will overwrite original.
            bandwidth (int, optional): Bandwith fo the multitaper smoothing. Defaults to 8.
                - Should be 2,4,8,16,32 etc. (the higher the number the stronger the smoothing).

        Returns:
            pd.Series: Series with the spectral data, the index is specifying the frequency per day.
        """
        if p is None:
            if "p" in vars(self).keys():
                p = pd.Series(self.p, index=self.t)
            else:
                raise ValueError(
                    "No pressure data was passed to the object. Please pass 'p' as a parameter or set 'self.p' before calling this method."
                )
        else:
            if not isinstance(p, pd.Series):
                raise TypeError(
                    f"'p' should be a pandas Series, not a {type(p).__name__}."
                )
        if not isinstance(bandwidth, int):
            raise TypeError(
                f"'bandwidth' should be an int, not a {type(bandwidth).__name__}."
            )
        self.spectrum: pd.Series = calculate_tidal_spectrum(p, bandwidth=bandwidth)

        return self.spectrum

    def reconstruct(
        self,
        t: npt.ArrayLike = None,
        constituents: list[str] = None,
        exclude_constituents: list[str] = None,
        **args,
    ) -> pd.DataFrame:
        """Reconstructs the tidal signal for a given time index using the utide package.

        Args:
            t (array_like, optional): Time index for reconstruction. Defaults to None.
                - If None, uses the time index from the object.
            constituents (list[str], optional): List of standard letter abbreviations of tidal constituents to reconstruct. Defaults to None.
                - If None, uses all constituents from the utide tidalharmonicanalysis.
            exclude_constituents (list[str], optional): List of constituents to exclude from the reconstruction. Defaults to None.
                - If provided, constituents will be excluded from the reconstruction.
            **args: Additional keyword arguments passed to `utide.reconstruct`.

        Returns:
            pd.DataFrame: DataFrame with the reconstructed tidal signal.
        """
        if t is None:
            if "t" in vars(self).keys():
                t = self.t
            else:
                raise ValueError(
                    "No time index was passed to the object. Please pass 't' as a parameter or set 'self.t' before calling this method."
                )
        if not pd.api.types.is_list_like(t):
            raise TypeError(f"'t' should be a array_like, not a {type(t).__name__}.")

        if not hasattr(self, "constituents"):
            raise AttributeError(
                "The tide object does not have tidal constituents. Please perform a tidal harmonic analysis first."
            )
        if constituents is not None:
            if not isinstance(constituents, list):
                raise TypeError(
                    f"'constituents' should be a list of strings, not a {type(constituents).__name__}."
                )
            for i in constituents:
                if not isinstance(i, str):
                    raise TypeError(
                        f"Each element in 'constituents' should be a string, not a {type(i).__name__}."
                    )
        if exclude_constituents is not None:
            if constituents is not None:
                logging.warning(
                    "Warning: Both 'constituents' and 'exclude_constituents' were passed. 'exclude_constituents' will be ignored."
                )
            else:
                if not isinstance(exclude_constituents, list):
                    raise TypeError(
                        f"'exclude_constituents' should be a list of strings, not a {type(exclude_constituents).__name__}."
                    )
                for i in exclude_constituents:
                    if not isinstance(i, str):
                        raise TypeError(
                            f"Each element in 'exclude_constituents' should be a string, not a {type(i).__name__}."
                        )
                constituents = [
                    c for c in self.constituents.index if c not in exclude_constituents
                ]
        if constituents is None:
            constituents = list(self.constituents.index)

        temp_data: tide = tidal_reconstruct(self, t=t, constit=constituents, **args)

        self.recon: pd.DataFrame = temp_data.recon
        self.recon_utide: dict = temp_data.recon_utide

        return self.recon

    def plot_spectrum(
        self, constituents: list[str] = None, exclude_constituents: list[str] = None
    ) -> tuple[plt.Figure, matplotlib.axes.Axes]:
        """Plots the tidal spectrum of the tide object.

        Args:
            constituents (list[str], optional): List of tidal constituents to plot. Defaults to None.
                - If None, plots all constituents from the spectrum.
            exclude_constituents (list[str], optional): List of constituents to exclude from the plot. Defaults to None.
                - If provided, constituents will be excluded from the plot.

        Returns:
            tuple[plt.Figure, matplotlib.axes.Axes]:
                - The figure of the plot.
                - The axes of the plot.
        """
        if not hasattr(self, "spectrum"):
            self.calc_spectrum()
        if constituents is not None:
            if not isinstance(constituents, list):
                raise TypeError(
                    f"'constituents' should be a list of strings, not a {type(constituents).__name__}."
                )
            for i in constituents:
                if not isinstance(i, str):
                    raise TypeError(
                        f"Each element in 'constituents' should be a string, not a {type(i).__name__}."
                    )
        if exclude_constituents is not None:
            if constituents is not None:
                logging.warning(
                    "Warning: 'constituents' and 'exclude_constituents' were both passed. 'exclude_constituents' will be ignored."
                )
            elif not isinstance(exclude_constituents, list):
                raise TypeError(
                    f"'exclude_constituents' should be a list of strings, not a {type(exclude_constituents).__name__}."
                )
            else:
                for i in exclude_constituents:
                    if not isinstance(i, str):
                        raise TypeError(
                            f"Each element in 'exclude_constituents' should be a string, not a {type(i).__name__}."
                        )
            constituents = [
                c for c in self.constituents.index if c not in exclude_constituents
            ]
        if constituents is None and exclude_constituents is None:
            constituents = self.constituents.index

        fig, ax = plot_tidal_spectrum(self.spectrum, constituents=constituents)

        return fig, ax

    def plot_map_tidal_ellipses(
        self,
        constituents: list[str] = None,
        exclude_constituents: list[str] = None,
        lat_center: num.Number = 78.122,
        lon_center: num.Number = 14.26,
        map_extent: list = [11.0, 16.0, 78.0, 78.3],
        topography: str = None,
    ) -> tuple[plt.Figure, matplotlib.axes.Axes, Any]:
        """Plots tidal ellipses for selected constituents on a map.

        Args:
            constituents (list[str], optional): List of tidal constituents to plot. Defaults to None (all).
            exclude_constituents (list[str], optional): List of constituents to exclude. Defaults to None.
            lat_center (float, optional): Center latitude for ellipses. Defaults to 78.122.
            lon_center (float, optional): Center longitude for ellipses. Defaults to 14.26.
            map_extent (list, optional): [lon_min, lon_max, lat_min, lat_max]. Defaults to [11.0, 16.0, 78.0, 78.3].
            topography (str, optional): Path or array for bathymetry. Defaults to None.

        Returns:
            tuple[plt.Figure, matplotlib.axes.Axes, Any]:
                - Figure object.
                - Map axes.
                - Ellipse inset axes.
        """
        if not hasattr(self, "constituents"):
            raise AttributeError(
                "The tide object does not have tidal constituents. Please perform a tidal harmonic analysis first."
            )
        if not hasattr(self, "u") or not hasattr(self, "v"):
            raise AttributeError("The tide object does not have tidal current data.")

        if constituents is not None:
            if not isinstance(constituents, list):
                raise TypeError(
                    f"'constituents' should be a list of strings, not a {type(constituents).__name__}."
                )
            for i in constituents:
                if not isinstance(i, str):
                    raise TypeError(
                        f"Each element in 'constituents' should be a string, not a {type(i).__name__}."
                    )
        if exclude_constituents is not None:
            if constituents is not None:
                logging.warning(
                    "Warning: Both 'constituents' and 'exclude_constituents' were passed. 'exclude_constituents' will be ignored."
                )
            else:
                if not isinstance(exclude_constituents, list):
                    raise TypeError(
                        f"'exclude_constituents' should be a list of strings, not a {type(exclude_constituents).__name__}."
                    )
                for i in exclude_constituents:
                    if not isinstance(i, str):
                        raise TypeError(
                            f"Each element in 'exclude_constituents' should be a string, not a {type(i).__name__}."
                        )
                constituents = [
                    c for c in self.constituents.index if c not in exclude_constituents
                ]
        if constituents is None:
            constituents = list(self.constituents.index)

        amp_major = [self.constituents["amp_major"][c] for c in constituents]
        amp_minor = [self.constituents["amp_minor"][c] for c in constituents]
        inclin = [self.constituents["inclination [deg]"][c] for c in constituents]

        if not isinstance(lat_center, num.Number):
            raise TypeError(
                f"'lat_center' should be a number, not a {type(lat_center).__name__}."
            )
        if not isinstance(lon_center, num.Number):
            raise TypeError(
                f"'lon_center' should be a number, not a {type(lon_center).__name__}."
            )

        fig, ax, ax_ellipse = plot_map_tidal_ellipses(
            amp_major,
            amp_minor,
            inclin,
            constituents,
            lat_center,
            lon_center,
            map_extent,
            topography,
        )
        return fig, ax, ax_ellipse

    def plot_tidal_ellipses(
        self,
        constituents: list[str] = None,
        exclude_constituents: list[str] = None,
        multiple_plots: bool = False,
        n_row_col: tuple[int, int] = None,
    ) -> tuple[plt.Figure, tuple[matplotlib.axes.Axes] | matplotlib.axes.Axes]:
        """
        Plots tidal ellipses for selected constituents.

        Args:
            constituents (list[str], optional): List of tidal constituents to plot. Defaults to all.
            exclude_constituents (list[str], optional): List of constituents to exclude. Defaults to None.
            multiple_plots (bool, optional): If True, plot each constituent in a separate subplot. Defaults to False.
            n_row_col (tuple[int, int], optional): Tuple specifying (nrows, ncols) for subplots. If None, tries to guess. Defaults to None.

        Returns:
            tuple[plt.Figure, tuple[matplotlib.axes.Axes] | matplotlib.axes.Axes]:
                - Figure object.
                - Axes object(s).
        """
        if not hasattr(self, "constituents"):
            raise AttributeError(
                "The tide object does not have tidal constituents. Please perform a tidal harmonic analysis first."
            )
        if constituents is not None:
            if not isinstance(constituents, list):
                raise TypeError(
                    f"'constituents' should be a list of strings, not a {type(constituents).__name__}."
                )
            for i in constituents:
                if not isinstance(i, str):
                    raise TypeError(
                        f"Each element in 'constituents' should be a string, not a {type(i).__name__}."
                    )
        if exclude_constituents is not None:
            if constituents is not None:
                logging.warning(
                    "Warning: Both 'constituents' and 'exclude_constituents' were passed. 'exclude_constituents' will be ignored."
                )
            else:
                if not isinstance(exclude_constituents, list):
                    raise TypeError(
                        f"'exclude_constituents' should be a list of strings, not a {type(exclude_constituents).__name__}."
                    )
                for i in exclude_constituents:
                    if not isinstance(i, str):
                        raise TypeError(
                            f"Each element in 'exclude_constituents' should be a string, not a {type(i).__name__}."
                        )
                constituents = [
                    c for c in self.constituents.index if c not in exclude_constituents
                ]
        if constituents is None:
            constituents = list(self.constituents.index)

        amp_major = [self.constituents["amp_major"][c] for c in constituents]
        amp_minor = [self.constituents["amp_minor"][c] for c in constituents]
        inclin = [self.constituents["inclination [deg]"][c] for c in constituents]

        return plot_tidal_ellipses(
            amp_major,
            amp_minor,
            inclin,
            constituents,
            muliple_plots=multiple_plots,
            n_row_col=n_row_col,
        )


def calculate_tidal_spectrum(data: pd.Series, bandwidth: int = 8) -> pd.Series:
    """Function to calculate a spectrum for a given time series
    (e.g. pressure measurements from a SeaGuard).
    The raw periodogram is filtered using a multitapering approach.

    Args:
        data (pd.Series): Time series of data.
        bandwidth (int, optional): Bandwith fo the multitaper smoothing. Defaults to 8.
            - Should be 2,4,8,16,32 etc. (the higher the number the stronger the smoothing).

    Returns:
        pd.Series: Series with the spectral data, the index is specifying the frequency per day.
    """
    if not isinstance(data, pd.Series):
        raise TypeError(
            f"'data' should be a pandas Series, not a {type(data).__name__}."
        )
    if not isinstance(bandwidth, int):
        raise TypeError(
            f"'bandwidth' should be a int, not a {type(bandwidth).__name__}."
        )

    timeseries: NDArray = data.interpolate(method="linear").values
    resolution: float = (data.index[1] - data.index[0]).seconds / 3600.0
    delta: float = resolution * (1.0 / 24.0)  # in days

    N: int = len(timeseries)

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


def tidal_harmonic_analysis(
    df: pd.DataFrame,
    lat: float,
    constituents: str | list[str] = "auto",
    add_vari: list[str] = None,
    **args,
) -> tide | tuple[dict, dict]:
    """Performs tidal harmonic analysis on a time series, e.g., pressure and/or u/v current measurements from a SeaGuard.

        The function automatically detects u/v components and matches them as pairs. If multiple u/v pairs exist, they are matched by their `u_*` suffix.
    If the same variable appears with different units more then twice, only the last occurrence will be retained.


        Args:
            df (pd.DataFrame): Dataframe containing the data with a pd.DatetimeIndex.
            lat (float): Latitude in degrees.
            constituents (str or list[str], optional): List of tidal constituents by their standard letter abbreviations or 'auto'. Defaults to "auto".
                - If 'auto', the constituent list is selected based on the data's time span.
            add_vari (list[str], optional): Additional variables to include in the output dictionary for each constituent. Defaults to None.

        Additional parameters:
            - Other keyword arguments are passed to `utide.solve`. Use `help(utide.solve)` for more details.
            - Allows for manual data input:
                - u/v: manual = (time,u,v)
                - p: manual = (time,p)

        Returns:
            tide or tuple[dict, dict]: A class tide if only one variable (u/v or pressure) is analyzed, or a tuple of dictionaries if both are analyzed.
                - If more than one variable (or pair) is analyzed, the results are nested under their suffixes.
    """
    if "manual" in args.keys():
        manual = args.pop("manual")
    else:
        manual = None

    if not manual and not isinstance(df, pd.DataFrame):
        raise TypeError(
            f"'data' should be a pandas DataFrame, not a {type(df).__name__}."
        )

    if not manual and not isinstance(df.index, pd.DatetimeIndex):
        raise TypeError(
            f"The index of 'data' should be a pandas DatetimeIndex, not a {type(df.index).__name__}."
        )

    if not isinstance(lat, (float, int)):
        raise TypeError(f"'lat' should be a float, not a {type(lat).__name__}.")
    args["lat"] = lat

    if not isinstance(constituents, (str, list)):
        raise TypeError(
            f"'constituents' should be a string or a list, not a {type(constituents).__name__}."
        )
    if isinstance(constituents, str):
        if constituents != "auto":
            raise ValueError(
                f"'constituents' should be 'auto' or a list, not {constituents}."
            )
    if isinstance(constituents, list):
        if len(constituents) == 0:
            raise ValueError(
                f"'constituents' should be 'auto' or a list with at least one element, not {constituents}."
            )
        for i in constituents:
            if not isinstance(i, str):
                raise TypeError(
                    f"Elements of 'constituents' should be strings, not {type(i).__name__}."
                )
    args["constit"] = constituents

    if add_vari == None:
        add_vari = []
    if not isinstance(add_vari, list):
        raise TypeError(f"'add_var' should be a list, not a {type(add_vari).__name__}.")

    def _search(name: str, var: str) -> tuple[str, str | Any, str | Any] | None:
        if rematch := re.match(rf"{var}(?:_(\w+))?(?:\s*\[\s*([^\]]+)\s*\])?", name):
            return name, rematch.group(1), rematch.group(2)
        else:
            return None

    finddict: dict[str, list] = {"u": [], "v": [], "uv": [], "p": []}
    if not manual:
        # find if u and v or p is in the data
        for name in df.columns:
            if uindata := _search(name, "u"):
                finddict["u"].append(uindata)
            elif vindata := _search(name, "v"):
                finddict["v"].append(vindata)
            elif pindata := _search(name, "p"):
                finddict["p"].append((pindata[0], pindata[1]))

        # match u and v, check for same unit
        for u_item in finddict["u"]:
            for v_item in finddict["v"]:
                if u_item[1] == v_item[1]:
                    if u_item[2] == v_item[2]:
                        finddict["uv"].append((u_item[0], v_item[0], u_item[1]))
                        finddict["u"].remove(u_item)
                        finddict["v"].remove(v_item)
                        break
                    else:
                        logging.warning(
                            f"Warning: A match of u and v was found for: {u_item[0]} and {v_item[0]}, but they have different units."
                        )

    if len(finddict["u"]) != 0:
        logging.warning(
            "Warning: No matches where found for u: {}".format(
                [colname[0] for colname in finddict["u"]]
            )
        )
    if len(finddict["v"]) != 0:
        logging.warning(
            "Warning: No matches where found for v: {}".format(
                [colname[0] for colname in finddict["v"]]
            )
        )
    if len(finddict["uv"]):
        logging.info(
            "The following u and v where matched: {}".format(
                [pair[0:2] for pair in finddict["uv"]]
            )
        )
    if len(finddict["p"]):
        logging.info(
            "The following p where found: {}".format(
                [name[0] for name in finddict["p"]]
            )
        )

    if manual != None and len(manual) == 2:
        df = pd.DataFrame({"p": manual[1]}, index=pd.DatetimeIndex(manual[0]))
        finddict["p"] = [("p", None)]
    elif manual != None and len(manual) == 3:
        df = pd.DataFrame(
            {"u": manual[1], "v": manual[2]}, index=pd.DatetimeIndex(manual[0])
        )
        finddict["uv"] = [("u", "v", None)]
    elif manual != None:
        raise ValueError(
            "It was not possible to read the manual data. Please follow the format described in the docstring."
        )

    uvdata: dict | None = None
    pdata: dict | None = None
    if len(finddict["uv"]):
        uvdata = {}
        for pair in finddict["uv"]:
            pairdata = utide.solve(t=df.index, u=df[pair[0]], v=df[pair[1]], **args)
            add_vari_in: list = []
            add_vari_out: list = []
            for add_var in add_vari:
                if add_var in pairdata.keys():
                    add_vari_in.append(add_var)
                else:
                    add_vari_out.append(add_var)
            if len(add_vari_out):
                logging.warning(
                    f"Warning: {add_vari_out} couldn't be found in the data for {pname[2]}."
                )
            df_data: pd.DataFrame = pd.DataFrame(
                {
                    "amp_major": pairdata["Lsmaj"],
                    "amp_major_ci": pairdata["Lsmaj_ci"],
                    "amp_minor": pairdata["Lsmin"],
                    "amp_minor_ci": pairdata["Lsmin_ci"],
                    "inclination [deg]": pairdata["theta"],
                    "inclination_ci [deg]": pairdata["theta_ci"],
                    "phase [deg]": pairdata["g"],
                    "phase_ci [deg]": pairdata["g_ci"],
                    "snr": pairdata["SNR"],
                    "rotation": [
                        "CW" if val > 0 else "CCW" for val in pairdata["Lsmin"]
                    ],
                    "PE [%]": pairdata["PE"],
                },
                index=pairdata["name"],
            )

            for add_var in add_vari_in:
                if add_var in pairdata.keys:
                    df_data[add_var] = pairdata[add_var]
            df_data = df_data.sort_values(by=["snr", "amp_major"], ascending=False)
            total_variability: float = np.sum(df_data["amp_major"] ** 2)
            df_data["percent_variability"] = (
                df_data["amp_major"] ** 2 / total_variability
            )
            dict_data: dict = {
                "total_variability": total_variability,
                "umean": pairdata["umean"],
                "vmean": pairdata["vmean"],
            }

            if "trend" in args.keys():
                if args["trend"]:
                    dict_data["uslope"] = pairdata["uslope"]
                    dict_data["vslope"] = pairdata["vslope"]
            else:
                dict_data["uslope"] = pairdata["uslope"]
                dict_data["vslope"] = pairdata["vslope"]
            if not pair[2]:
                dic_name = "No_suffix"
            else:
                dic_name = pair[2]
            if dic_name in uvdata.keys():
                uvdata[dic_name + "_1"] = tide(
                    df_const=df_data,
                    single_val=dict_data,
                    orignal=pairdata,
                )
                uvdata[dic_name + "_0"] = uvdata.pop(dic_name)
            else:
                uvdata[dic_name] = tide(
                    df_const=df_data,
                    single_val=dict_data,
                    orignal=pairdata,
                )
        if len(uvdata) == 1:
            uvdata = uvdata[list(uvdata)[0]]
    if len(finddict["p"]):
        pdata: dict = {}
        for pname in finddict["p"]:
            namedata = utide.solve(t=df.index, u=df[pname[0]], **args)
            add_vari_in: list = []
            add_vari_out: list = []
            for add_var in add_vari:
                if add_var in namedata.keys():
                    add_vari_in.append(add_var)
                else:
                    add_vari_out.append(add_var)
            if len(add_vari_out):
                logging.warning(
                    f"Warning: {add_vari_out} couldn't be found in the data for {pname[2]}."
                )
            df_data: pd.DataFrame = pd.DataFrame(
                {
                    "amplitude": namedata["A"],
                    "amplitude_ci": namedata["A_ci"],
                    "phase [deg]": namedata["g"],
                    "phase_ci [deg]": namedata["g_ci"],
                    "snr": namedata["SNR"],
                    "PE [%]": namedata["PE"],
                },
                index=namedata["name"],
            )
            for add_var in add_vari_in:
                df_data[add_var] = namedata[add_var]

            df_data = df_data.sort_values(by=["snr", "amplitude"], ascending=False)
            total_variability: float = np.sum(df_data["amplitude"] ** 2)
            df_data["percent_variability"] = (
                df_data["amplitude"] ** 2 / total_variability
            )

            dict_data: dict = {
                "total_variability": total_variability,
                "mean": namedata["mean"],
            }
            if "trend" in args.keys():
                if args["trend"]:
                    dict_data["slope"] = namedata["slope"]
            else:
                dict_data["slope"] = namedata["slope"]

            if not pname[1]:
                dic_name = "No_suffix"
            else:
                dic_name = pname[1]
            if dic_name in pdata.keys():
                pdata[dic_name + "_1"] = tide(
                    df_const=df_data,
                    single_val=dict_data,
                    orignal=namedata,
                )
                pdata[dic_name + "_0"] = pdata.pop(dic_name)
            else:
                pdata[dic_name] = tide(
                    df_const=df_data,
                    single_val=dict_data,
                    orignal=namedata,
                )
        if len(pdata) == 1:
            pdata = pdata[list(pdata)[0]]

    if not (uvdata or pdata):
        raise ValueError(
            f"No data found in 'data'. Make sure it contains either 'u' and 'v' or 'p'. It doesn't matter if there is a sufix (_asd) or a unit ([...]) behind."
        )
    if uvdata and pdata:
        return uvdata, pdata
    if pdata:
        return pdata
    if uvdata:
        return uvdata


def tidal_reconstruct(
    data: tide, t: npt.ArrayLike = None, min_SNR: int = 2, min_PE: int = 0, **args
) -> tide:
    """Reconstructs the tidal signal from a tide object.

    Args:
        data (tide): Tide object containing the tidal constituents.
        t (array_like, optional): Time index for the reconstruction. Defaults to None.
            - If None, uses the time index from the tide object.
        min_SNR (int, optional): Minimum signal-to-noise ratio to consider a constituent. Defaults to 2.
        min_PE (int, optional): Minimum percentage of energy to consider a constituent. Defaults to 0.
        **args: Additional arguments passed to `utide.reconstruct`.

    Returns:
        tide: Tide object with the reconstructed tidal signal.
    """

    if not isinstance(data, tide):
        raise TypeError(f"'data' should be a tide object, not a {type(data).__name__}.")
    if not "utide" in vars(data).keys():
        raise ValueError(
            f"Run a tidal harmonic analysis on 'data', by eiter running tidal_harmonic_analysis() or create a tide object with the necessary data."
        )
    temp_data: tide = copy.deepcopy(data)

    if t is not None:
        if not isinstance(t, pd.DatetimeIndex):
            raise TypeError(
                f"'t' should be a pandas DatetimeIndex, not a {type(t).__name__}."
            )
        temp_data.t = t

    if not isinstance(min_SNR, int):
        raise TypeError(f"'min_SNR' should be an int, not a {type(min_SNR).__name__}.")

    if not isinstance(min_PE, int):
        raise TypeError(f"'min_PE' should be an int, not a {type(min_PE).__name__}.")

    temp_data.recon_utide = utide.reconstruct(
        temp_data.t, temp_data.utide, min_SNR=min_SNR, min_PE=min_PE, **args
    )

    if "h" in temp_data.recon_utide.keys():
        temp_df = pd.DataFrame(
            {
                "p": temp_data.recon_utide.h,
                "p_atide": temp_data.recon_utide.h - temp_data.single_values["mean"],
                "p_detide": temp_data.p
                - temp_data.recon_utide.h
                + temp_data.single_values["mean"],
            },
            index=temp_data.recon_utide.t_in,
        )
        temp_data.recon = temp_df
    elif "u" in temp_data.recon_utide.keys() and "v" in temp_data.recon_utide.keys():
        temp_df = pd.DataFrame(
            {
                "u": temp_data.recon_utide.u,
                "v": temp_data.recon_utide.v,
                "u_atide": temp_data.recon_utide.u - temp_data.single_values["umean"],
                "v_atide": temp_data.recon_utide.v - temp_data.single_values["vmean"],
                "u_detide": temp_data.u
                - temp_data.recon_utide.u
                + temp_data.single_values["umean"],
                "v_detide": temp_data.v
                - temp_data.recon_utide.v
                + temp_data.single_values["vmean"],
            },
            index=temp_data.recon_utide.t_in,
        )
        temp_data.recon = temp_df
    else:
        logging.warning(
            "Warning: Could not reconstruct the data. Check tide.recon_utide for more information."
        )

    return temp_data


############################################################################
# PLOTTING FUNCTIONS
############################################################################


def contour_section(
    X: npt.ArrayLike,
    Y: npt.ArrayLike,
    Z: npt.ArrayLike,
    Z2: npt.ArrayLike = None,
    ax: Any = None,
    station_pos: npt.ArrayLike = None,
    station_text: str | npt.ArrayLike = None,
    cmap: str | npt.ArrayLike = "jet",
    Z2_contours: int | npt.ArrayLike = None,
    clabel: str = "",
    bottom_depth: npt.ArrayLike = None,
    clevels: int | npt.ArrayLike = 20,
    interp_opt: int = 1,
    tlocator: Any = None,
    cbar: bool = True,
) -> tuple[matplotlib.axes.Axes, Any, Any | None]:
    """Plots a filled contour plot of *Z*, with contour lines of *Z2* on top to
    the axes *ax*. It also displays the position of stations, if given in
    *station_pos*, adds labels to the contours of Z2, given in
    *Z2_contours*. If no labels are given, it assumes Z2 is density (sigma0)
    and adds its own labels. It adds bottom topography if given in *bottom_depth*.

    Args:
        X (array_like): (N,) X-values.
        Y (array_like): (K,) Y-values.
        Z (array_like): (K,N) The filled contour field.
        Z2 (array_like, optional): (K,N) The contour field on top. Defaults to None.
        ax (plot axes, optional): Axes object to plot on.. Defaults to current axes.
        station_pos (array_like, optional): (S,) The station positions. Defaults to None.
            - Places a arrow at the position of the station.
        station_text (str or array_like, optional): Name to label the station locations or plot. Defaults to None.
            - If a string is given, it will be used as the label for the plot.
            - If an array_like is given, it will be used as the label for each station.
        cmap (str or array_like, optional): The colormap for the filled contours. Defaults to "jet".
        Z2_contours (int or array_like, optional): The contour label positions for `Z2` or amount. Defaults to None.
        clabel (str, optional): Label to put on the colorbar. Defaults to "".
        bottom_depth (array_like, optional): (S,) List with bottom depth. Defaults to None.
        clevels (int or array_like, optional): List of color levels or number of levels to use for `Z`. Defaults to 20.
        interp_opt (int, optional): Indicator which is used to decide whether to use pcolormesh or contourf.  Defaults to 1.
            - 0: only z-interpolation, use pcolormesh
            - 1: full interpolation, use contourf.
        tlocator (matplotlib.ticker locators, optional): Special locator for the colorbar. Defaults to None.
            - Example: logarithmic locator (use `matplotlib.ticker.LogLocator`).
        cbar (bool, optional): If the colorbar is displayed. Defaults to True.

    Returns:
        tuple[Any, Any, Any | None]:
            - plot axes: The axes object with the filled contour plot.
            - contour lines: The contour lines of Z.
            - colorbar: The colorbar object from Z2, if displayed, otherwise None.
    """

    if not pd.api.types.is_list_like(X):
        raise TypeError(f"'X' should be array_like, not a {type(X).__name__}.")
    if not pd.api.types.is_list_like(Y):
        raise TypeError(f"'Y' should be array_like, not a {type(Y).__name__}.")

    if not pd.api.types.is_list_like(Z):
        raise TypeError(f"'Z' should be array_like, not a {type(Z).__name__}.")
    if not (len(Y), len(X)) == np.shape(Z):
        raise ValueError(
            f"('X', 'Y') should have the same shape as 'Z', not ({len(X),len(Y)} and {np.shape(Z)}."
        )
    if not pd.api.types.is_list_like(Z2) and Z2 is not None:
        raise TypeError(f"'Z2' should be array_like, not a {type(Z2).__name__}.")
    if Z2 is not None and not (len(Y), len(X)) == np.shape(Z2):
        raise ValueError(
            f"('X', 'Y') should have the same shape as Z2, not ({len(X),len(Y)} and {np.shape(Z2)}."
        )

    # open new figure and get current axes, if none is provided
    if not isinstance(ax, plt.Axes):
        if ax is not None:
            raise TypeError(
                f"'ax' should be a matplotlib Axes object, not a {type(ax).__name__}."
            )
    if ax is None:
        ax: plt.Axes = plt.gca()

    if not pd.api.types.is_list_like(station_pos) and station_pos is not None:
        raise TypeError(
            f"'station_pos' should be array_like, not a {type(station_pos).__name__}."
        )
    if station_text is not None:
        if station_pos is None:
            logging.warning(
                "Warning: 'station_text' is given, but 'station_pos' is None. No station positions will be plotted."
            )
        elif pd.api.types.is_list_like(station_text):
            for i in station_text:
                if not isinstance(i, str):
                    raise TypeError(
                        f"Elements of 'station_text' should be strings, not a {type(i).__name__}."
                    )
            if len(station_text) == 1:
                station_text = station_text[0]
            elif len(station_text) != len(station_pos):
                raise ValueError(
                    f"'station_text' should have the same length as 'station_pos', not {len(station_text)} and {len(station_pos)}."
                )
        elif not isinstance(station_text, str):
            raise TypeError(
                f"'station_text' should be array_like or str not a {type(station_text).__name__}."
            )
    else:
        station_text = ""

    if not (pd.api.types.is_list_like(Z2_contours) or int) and Z2_contours is not None:
        raise TypeError(
            f"'Z2_contours' should be int or array_like, not a {type(Z2_contours).__name__}."
        )

    if not isinstance(clabel, str):
        raise TypeError(f"'clabel' should be a string, not a {type(clabel).__name__}.")

    if not (isinstance(clevels, (int)) or pd.api.types.is_list_like(clevels)):
        raise TypeError(
            f"'clevels' should be a int or array_like, not a {type(clevels).__name__}."
        )

    if not isinstance(interp_opt, int):
        raise TypeError(
            f"'interp_opt' should be an int, not a {type(interp_opt).__name__}."
        )
    if interp_opt not in [0, 1]:
        raise ValueError(f"'interp_opt' should be either 0 or 1, not {interp_opt}.")

    if not isinstance(cbar, bool):
        raise TypeError(f"'cbar' should be a bool, not a {type(cbar).__name__}.")

    # get the labels for the Z2 contours
    if Z2 is not None and Z2_contours is None:
        Z2_contours = np.concatenate([list(range(21, 26)), np.arange(25.5, 29, 0.2)])
        Z2_contours: list[Any] = [
            i for i in Z2_contours if np.nanmin(Z2) < i < np.nanmax(Z2)
        ]

    # get the Y-axis limits
    if bottom_depth is not None:
        y_limits: tuple = (0, np.nanmax(bottom_depth))
    else:
        y_limits = (0, np.nanmax(Y))

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
            np.full_like(bottom_depth, y_limits[1] + 10),
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
    if isinstance(station_text, str):
        ax.set_title(station_text)

    return ax, cT, cSIG


def plot_CTD_section(
    CTD: dict | str,
    stations: list[str],
    section_name: str = "",
    clevels_T: float | npt.ArrayLike = 20,
    clevels_S: float | npt.ArrayLike = 20,
    x_type: str = "distance",
    interp_opt: int = 1,
    bottom: npt.ArrayLike = None,
    z_fine: bool = False,
) -> tuple[matplotlib.axes.Axes, matplotlib.axes.Axes, Any, Any, Any, Any]:
    """This function plots a CTD section of Temperature and Salinity,
    given CTD data either directly or via file(s).

    Args:
        CTD (dict or str): Either a dictionary with CTD data or a string with the path.
            - Path can lead to a .npy file or a directory with .cnv files, check read_CTD.
        stations (list[str]): Stations to plot.
        section_name (str, optional): Name of the Section, will appear in the plot title. Defaults to "".
        clevels_T (float or npt.ArrayLike, optional): The levels of the filled contourf for the temperature plot. Defaults to 20.
            - Either a number of levels or the specific levels.
        clevels_S (float or npt.ArrayLike, optional): The levels of the filled contourf for the salinity plot. Defaults to 20.
            - Either a number of levels or the specific levels.
        x_type (str, optional): Wheter to use 'distance' or 'time' as the x-axis. Defaults to "distance".
        interp_opt (int, optional): Integer which interpolation method to use for gridding. Defaults to 1.
            - 0: no interpolation,
            - 1: linear interpolation, fine grid (default),
            - 2: linear interpolation, coarse grid. The default is 1.
        bottom (npt.ArrayLike, optional): Where the ground is drawn. Defaults to None.
            - If None, it will be extracted from the CTD data.
        z_fine (bool, optional): Whether to use a fine z grid. Defaults to False.
            - If True, will be 10 cm, otherwise 1 m.

    Returns:
        tuple[matplotlib.axes.Axes, matplotlib.axes.Axes, Any, Any, Any, Any]:
            - Axes for temperature subplot.
            - Axes for salinity subplot.
            - Contour object for temperature.
            - Contour object for salinity.
            - Contour object for the density in the temperature plot.
            - Contour object for the density in the salinity plot.
    """
    # Check if the function has data to work with
    if not isinstance(CTD, (dict, str)):
        raise TypeError(
            f"'CTD' should be a dict or a npy file string with the data, not {type(CTD).__name__}."
        )

    # read in the data (only needed if no CTD-dict, but a file was given)
    if type(CTD) is str:
        CTD = read_CTD(CTD)
    # check if data is there
    if type(CTD) is dict and len(CTD) == 0:
        raise ValueError(
            "The CTD data is empty! Please provide a valid CTD data dictionary or file."
        )

    # Check if all stations given are found in the data
    if not pd.api.types.is_list_like(stations):
        raise TypeError(
            f"'stations' should be a list of strings, not a {type(stations).__name__}."
        )

    notfound_stations: list = [key for key in stations if not key in list(CTD.keys())]
    if len(notfound_stations) != 0:
        logging.info(
            f"The stations '{notfound_stations}' are not in the CTD data. Proceeding without them."
        )
        for i in notfound_stations:
            stations.remove(i)
        if len(stations) == 0:
            raise ValueError(f"There are no CTD stations left.")

    if not isinstance(section_name, str):
        raise TypeError(
            f"'section_name' should be a string, not a {type(section_name).__name__}."
        )

    if not isinstance(clevels_T, int) and not pd.api.types.is_list_like(clevels_T):
        raise TypeError(
            f"'clevels_T' should be an int or an array-like, not a {type(clevels_T).__name__}."
        )
    if not isinstance(clevels_S, int) and not pd.api.types.is_list_like(clevels_S):
        raise TypeError(
            f"'clevels_S' should be an int or an array-like, not a {type(clevels_S).__name__}."
        )

    if not interp_opt in [0, 1, 2]:
        raise ValueError(f"'interp_opt' should be 0, 1 or 2, not {interp_opt}.")

    if not pd.api.types.is_list_like(bottom) and bottom is not None:
        raise TypeError(
            f"'bottom' should be an array_like, not a {type(bottom).__name__}."
        )

    if not isinstance(z_fine, bool):
        raise TypeError(f"'z_fine' should be a boolean, not a {type(z_fine).__name__}.")

    # Check if x_type is either distance or time
    if x_type not in get_args(__x_type__):
        raise ValueError(f"'x_type' should be 'time' or 'distance' not '{x_type}'.")

    # select only the given stations in the data
    CTD = {key: CTD[key] for key in stations}

    # extract Bottom Depth
    if bottom is None:
        BDEPTH: np.ndarray = np.asarray([d["BottomDepth [m]"] for d in CTD.values()])
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
        fCTD["T [degC]"],
        fCTD["SIGTH [kg/m^3]"],
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
        fCTD["S []"],
        fCTD["SIGTH [kg/m^3]"],
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
    CTD: str | dict,
    stations: list[str],
    section_name: str = "",
    x_type: __x_type__ = "distance",
    parameter: str = "T [degC]",
    parameter_contourlines: str = "SIGTH [kg/m^3]",
    clabel: str = "Temperature [˚C]",
    cmap: Any = cmocean.cm.thermal,
    clevels: int | npt.ArrayLike = 20,
    contourlevels: int | npt.ArrayLike = 5,
    interp_opt: int = 1,
    bottom: npt.ArrayLike = None,
    tlocator: Any = None,
    z_fine: bool = False,
    cbar: bool = True,
) -> tuple[matplotlib.axes.Axes, Any, Any]:
    """This function plots a CTD section of a chosen variable,
    given CTD data either directly or via a file (through `CTD`).


    Args:
        CTD (dict or str): Either a dictionary with CTD data or a string with the path.
            - Path can lead to a .npy file or a directory with .cnv files, check read_CTD.
        stations (list[str]): Stations to plot.
        section_name (str, optional): Name of the Section, will appear in the plot title. Defaults to "".
        x_type (str, optional): Wheter to use 'distance' or 'time' as the x-axis. Defaults to "distance".
        parameter (str, optional): Name of the parameter to plot as filled contours. Defaults to "T [degC]".
        parameter_contourlines (str, optional): Name of the parameter to plot as contourlines. Defaults to "SIGTH [kg/m^3]".
        clabel (str, optional): The label on the colorbar axis. Defaults to "Temperature [˚C]".
        cmap (Any, optional): THe colormap to be used. Defaults to cmocean.cm.thermal.
        clevels (int or npt.ArrayLike, optional): The levels of the filled contour.. Defaults to 20.
            - Either a number of levels, or the specific levels.
        contourlevels (int or npt.ArrayLike, optional): The levels of contourlines. Defaults to 5.
            - Either a number of levels, or the specific levels.
        interp_opt (int, optional): Which interpolation method to use for gridding. Defaults to 1.
            - 0: no interpolation,
            - 1: linear interpolation, fine grid (default),
            - 2: linear interpolation, coarse grid.
        bottom (npt.ArrayLike, optional): The bottom topography. Defaults to None.
            - Either an array with values extracted from a bathymetry file, or None.
            - If None, the bottom depth from the CTD profiles will be used.
        tlocator (matplotlib.ticker locators, optional): Special locator for the colorbar. Defaults to None.
            - Example: logarithmic locator (use `matplotlib.ticker.LogLocator`).
        z_fine (bool, optional): Whether to use a fine z grid. Defaults to False.
            - If True, will be 10 cm, otherwise 1 m.
        cbar (bool, optional): If the colorbar is displayed. Defaults to True.

    Returns:
        tuple[matplotlib.axes.Axes, Any, Any]:
            - Axes for the CTD section plot.
            - Contour object for the filled contours.
            - Contour object for the contour lines.
    """
    # Check if the function has data to work with
    if not isinstance(CTD, (dict, str)):
        raise TypeError(
            f"'CTD' should be a dict or a npy file string with the data, not {type(CTD).__name__}."
        )

    # read in the data (only needed if no CTD-dict, but a file was given)
    if type(CTD) is str:
        CTD = read_CTD(CTD)
    # check if data is there
    if type(CTD) is dict and len(CTD) == 0:
        raise ValueError(
            "The CTD data is empty! Please provide a valid CTD data dictionary or file."
        )

    # Check if all stations given are found in the data
    if not pd.api.types.is_list_like(stations):
        raise TypeError(
            f"'stations' should be a list of strings, not a {type(stations).__name__}."
        )

    notfound_stations: list = [key for key in stations if not key in list(CTD.keys())]
    if len(notfound_stations) != 0:
        logging.info(
            f"The stations '{notfound_stations}' are not in the CTD data. Proceeding without them."
        )
        for i in notfound_stations:
            stations.remove(i)
        if len(stations) == 0:
            raise ValueError(f"There are no CTD stations left.")

    if not isinstance(section_name, str):
        raise TypeError(
            f"'section_name' should be a string, not a {type(section_name).__name__}."
        )

    # Check if x_type is either distance or time
    if x_type not in get_args(__x_type__):
        raise ValueError(f"'x_type' should be 'time' or 'distance' not '{x_type}'.")

    if not isinstance(parameter, str):
        raise TypeError(
            f"'parameter' should be a string, not a {type(parameter).__name__}."
        )
    if parameter not in CTD[list(CTD.keys())[0]].keys():
        raise ValueError(f"'{parameter}' is not a valid parameter in the CTD data.")

    if not isinstance(parameter_contourlines, str):
        raise TypeError(
            f"'parameter_contourlines' should be a string, not a {type(parameter_contourlines).__name__}."
        )
    if parameter_contourlines not in CTD[list(CTD.keys())[0]].keys():
        raise ValueError(
            f"'{parameter_contourlines}' is not a valid parameter in the CTD data."
        )

    if not isinstance(clabel, str):
        raise TypeError(f"'clabel' should be a string, not a {type(clabel).__name__}.")

    if not (isinstance(clevels, (int)) or pd.api.types.is_list_like(clevels)):
        raise TypeError(
            f"'clevels' should be a int or array_like, not a {type(clevels).__name__}."
        )

    if not (
        isinstance(contourlevels, (int)) or pd.api.types.is_list_like(contourlevels)
    ):
        raise TypeError(
            f"'contourlevels' should be a int or array_like, not a {type(contourlevels).__name__}."
        )

    if not isinstance(interp_opt, int):
        raise TypeError(
            f"'interp_opt' should be an int, not a {type(interp_opt).__name__}."
        )
    if interp_opt not in [0, 1, 2]:
        raise ValueError(f"'interp_opt' should be either 0, 1 or 2, not {interp_opt}.")

    if not pd.api.types.is_list_like(bottom) and bottom is not None:
        raise TypeError(
            f"'bottom' should be an array_like, not a {type(bottom).__name__}."
        )

    if not isinstance(z_fine, bool):
        raise TypeError(f"'z_fine' should be a boolean, not a {type(z_fine).__name__}.")

    if not isinstance(cbar, bool):
        raise TypeError(f"'cbar' should be a boolean, not a {type(cbar).__name__}.")

    # select only the given stations in the data
    CTD = {key: CTD[key] for key in stations}

    # extract Bottom Depth
    if bottom is None:
        if "BottomDepth [m]" in CTD[list(CTD.keys())[0]]:
            BDEPTH: np.ndarray = np.asarray(
                [d["BottomDepth [m]"] for d in CTD.values()]
            )
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
    list_das: list[xr.DataArray],
    list_cmaps: list[Any],
    list_clevels: list[npt.ArrayLike | int] = None,
    da_contours: xr.DataArray = None,
    contourlevels: int | npt.ArrayLike = 5,
    interp: bool = False,
    switch_cbar: bool = True,
    add_station_ticks: bool = True,
) -> tuple[plt.Figure, list[Any], list]:
    """Function to plot a variable number of variables from a section.
    Data can be from CTD or ADCP, but has to be provided as xarray datasets (see example notebook!)


    Args:
        list_das (list[xr.DataArray]): List of xarray DataArrays to plot.
            - Each DataArray should have 'distance' and 'depth' as coordinates.
        list_cmaps (list[Any]): Colormaps to be used for each subplot.
            - Same order as the DataArrays in `list_das`.
        list_clevels (list[npt.ArrayLike or int], optional): List with levels to use for the contourf plots. Defaults to None.
            - If None, it will use 20 levels for each DataArray.
            - Same order as the DataArrays in `list_das`.
        da_contours (xr.DataArray, optional): Dataarray with data to plot as contour lines on top of the contourf. Defaults to None.
        contourlevels (int or npt.ArrayLike, optional): Same as the clevels for the contourf plot, but for the contour lines. Defaults to 5.
            - If an int, it will use that many levels.
            - If an array_like, it will use those levels.
        interp (bool, optional): Interpolation to a finer grid. Defaults to False.
        switch_cbar (bool, optional): Adds colorbar to each contourf plot. Defaults to True.
        add_station_ticks (bool, optional): Add ticks for the locations of the CTD stations along the section. Defaults to True.

    Returns:
        tuple[plt.Figure, list[Any], list]:
            - Figure object with the subplots.
            - List of contourf objects for each subplot.
            - List of contour line objects for each subplot.
    """

    if not pd.api.types.is_list_like(list_das):
        raise TypeError(
            f"'list_das' should be a list of xarray DataArrays, not a {type(list_das).__name__}."
        )

    if not pd.api.types.is_list_like(list_cmaps):
        raise TypeError(
            f"'list_cmaps' should be a list of colormaps, not a {type(list_cmaps).__name__}."
        )

    if len(list_das) != len(list_cmaps):
        raise ValueError(
            f"'list_das' and 'list_cmaps' should have the same length, not {len(list_das)} and {len(list_cmaps)}."
        )

    if not isinstance(list_clevels, list) and list_clevels is not None:
        raise TypeError(
            f"'list_clevels' should be a list of levels, not a {type(list_clevels).__name__}."
        )
    if list_clevels is None:
        list_clevels = len(list_das) * [20]

    if len(list_das) != len(list_clevels):
        raise ValueError(
            f"'list_das' and 'list_clevels' should have the same length, not {len(list_das)} and {len(list_clevels)}."
        )

    if not isinstance(da_contours, xr.DataArray) and da_contours is not None:
        raise TypeError(
            f"'da_contours' should be an xarray DataArray, not a {type(da_contours).__name__}."
        )

    if (
        not (isinstance(da_contours, int) or pd.api.types.is_list_like(contourlevels))
        and da_contours is not None
    ):
        raise TypeError(
            f"'contourlevels' should be an int or array_like, not a {type(contourlevels).__name__}."
        )

    if not isinstance(interp, bool):
        raise TypeError(f"'interp' should be a boolean, not a {type(interp).__name__}.")

    if not isinstance(switch_cbar, bool):
        raise TypeError(
            f"'switch_cbar' should be a boolean, not a {type(switch_cbar).__name__}."
        )

    if not isinstance(add_station_ticks, bool):
        raise TypeError(
            f"'add_station_ticks' should be a boolean, not a {type(add_station_ticks).__name__}."
        )

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
        i = 0
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
            logging.info(
                "Station ticks not possible, no station data found in the provided data!"
            )

    for a in axes:
        a.set_xlim(x_limits)
        a.set_ylim(y_limits)
        a.invert_yaxis()
        a.yaxis.set_ticks_position("both")

    axes[-1].set_xlabel("Distance [km]")

    return fig, axes, pics


def plot_CTD_station(
    CTD: str | dict,
    station: str,
    lower_ax: str = "CT [degC]",
    lower_ax_label: str = "Conservative temperature [$^\\circ$C]",
    lower_color: str = "r",
    upper_ax: str = "SA [g/kg]",
    upper_ax_label: str = "Absolute salinity [g/kg]",
    upper_color: str = "b",
    axes: npt.ArrayLike = None,
    linestyle: str = "-",
) -> tuple[matplotlib.axes.Axes, matplotlib.axes.Axes]:
    """Plots the CTD data for a single station, with two axes:
    Standard temperature and absolute salinity.

    Args:
        CTD (dict or str): Either a dictionary with CTD data or a string with the path.
            - Path can lead to a .npy file or a directory with .cnv files, check read_CTD.
        station (str): The station to plot.
        lower_ax (str, optional): The name of the data in the dict. Defaults to "CT [degC]".
        lower_ax_label (str, optional): The label of the lower axes. Defaults to "Conservative temperature [$^\\circ]".
        lower_color (str, optional): The color for the lower axes. Defaults to "r".
        upper_ax (str, optional): The name of the data in the dict. Defaults to "SA [g/kg]".
        upper_ax_label (str, optional): The name of the data in the dict. Defaults to "Absolute salinity [g/kg]".
        upper_color (str, optional): The color for the upper axes. Defaults to "b".
        axes (array_like, optional): (2,) List of two axes. Defaults to None.
            - First: lower axes, second: upper axes.
            - If None, will create new axes.
        linestyle (str, optional): Linestyle to use. Defaults to "-".
            - Can be any valid matplotlib linestyle (e.g. '-','--', '-.', ':').

    Returns:
        tuple[matplotlib.axes.Axes, matplotlib.axes.Axes]:
            - Axes for the lower axes (temperature).
            - Axes for the upper axes (salinity).
    """
    # Check if the function has data to work with
    if not isinstance(CTD, (dict, str)):
        raise TypeError(
            f"'CTD' should be a dict or a npy file string with the data, not {type(CTD).__name__}."
        )

    # read in the data (only needed if no CTD-dict, but a file was given)
    if type(CTD) is str:
        CTD = read_CTD(CTD)
    # check if data is there
    if type(CTD) is dict and len(CTD) == 0:
        raise ValueError(
            "The CTD data is empty! Please provide a valid CTD data dictionary or file."
        )

    # Check if station is in the data
    if station in CTD.keys():
        # select station
        CTD = CTD[station]
    else:
        raise ValueError(f"The given station is not in the data!")

    if not isinstance(lower_ax, str):
        raise TypeError(
            f"'lower_ax' should be a string, not a {type(lower_ax).__name__}."
        )
    if lower_ax not in CTD.keys():
        raise ValueError(f"'{lower_ax}' is not a valid parameter in the CTD data.")

    if not isinstance(lower_ax_label, str):
        raise TypeError(
            f"'lower_ax_label' should be a string, not a {type(lower_ax_label).__name__}."
        )

    if not isinstance(lower_color, str):
        raise TypeError(
            f"'lower_color' should be a string, not a {type(lower_color).__name__}."
        )

    if not isinstance(upper_ax, str):
        raise TypeError(
            f"'upper_ax' should be a string, not a {type(upper_ax).__name__}."
        )
    if upper_ax not in CTD.keys():
        raise ValueError(f"'{upper_ax}' is not a valid parameter in the CTD data.")

    if not isinstance(upper_ax_label, str):
        raise TypeError(
            f"'upper_ax_label' should be a string, not a {type(upper_ax_label).__name__}."
        )

    if not isinstance(upper_color, str):
        raise TypeError(
            f"'upper_color' should be a string, not a {type(upper_color).__name__}."
        )

    if not pd.api.types.is_list_like(axes) and axes is not None:
        raise TypeError(
            f"'axes' should be a list of two axes, not a {type(axes).__name__}."
        )
    if axes is not None and len(axes) != 2:
        raise ValueError(
            f"'axes' should be a list of two axes, not a list of {len(axes)} axes."
        )
    if not isinstance(linestyle, str):
        raise TypeError(
            f"'linestyle' should be a string, not a {type(linestyle).__name__}."
        )
    # end of checks.

    if axes is None:
        ax: matplotlib.axes.Axes = plt.gca()
        ax2: matplotlib.axes.Axes = ax.twiny()
        ax.invert_yaxis()
    else:
        if len(axes) != 2:
            raise ValueError(f"'axes' should have a lenght of 2, not {len(axes)}.")
        else:
            ax = axes[0]
            ax2 = axes[1]

    # plot
    ax.plot(
        CTD[lower_ax], -CTD["z [m]"], lower_color, linestyle=linestyle, label=station
    )
    ax.set_xlabel(lower_ax_label, color=lower_color)
    ax.set_ylabel("Depth [m]")
    ax.spines["bottom"].set_color(lower_color)
    ax.tick_params(axis="x", colors=lower_color)

    if axes is not None:
        ax.legend(bbox_to_anchor=(1.0, 1.02), loc="upper left")

    ax2.plot(CTD[upper_ax], -CTD["z [m]"], upper_color, linestyle=linestyle)
    ax2.set_xlabel(upper_ax_label, color=upper_color)
    ax2.tick_params(axis="x", colors=upper_color)
    plt.tight_layout()

    return ax, ax2


def plot_CTD_map(
    CTD: str | dict,
    stations: str | npt.ArrayLike = None,
    topography: str | npt.ArrayLike = None,
    extent: npt.ArrayLike = None,
    depth_contours: npt.ArrayLike = [
        10,
        50,
        100,
        150,
        200,
        300,
        400,
        500,
        1000,
        2000,
        3000,
        4000,
        5000,
    ],
    st_labels: str | npt.ArrayLike = "",
    adjust_text: bool = False,
    min_dist: float = 250,
) -> tuple[plt.Figure, matplotlib.axes.Axes]:
    """Function which plots a very basic map of selected CTD stations.

    Args:
        CTD (dict or str): Either a dictionary with CTD data or a string with the path.
            - Path can lead to a .npy file or a directory with .cnv files, check read_CTD.
        stations (str or array_like, optional): The station(s) which should be displayed. Defaults to None.
        topography (str or array_like, optional): Topography data. Defaults to None.
            Can work with:
                - Path to the example data directory.
                - '.nc' with 'lat', 'lon' and 'z'.
                - '.mat' with 'lat', 'lon' and 'D'.
                - '.npy' with an array containing lat, lon and elevation as columns
                - array_like with lat, lon and elevation
        extent (array_like, optional): (4,) List of map extent. Defaults to None.
            - [lon0, lon1, lat0, lat1]
        depth_contours (array_like, optional): List containing countour levels for the bathymetry. Defaults to [10, 50, 100, 150, 200, 300, 400, 500, 1000, 2000, 3000, 4000, 5000].
        st_labels (str or array_like, optional): Text to label the stations. Defaults to "".
            - str: Adding to every station.
            - array_like: Needs to be the same length as stations, will override stationnumbers.
        adjust_text (bool, optional): Whether to adjust the station names so they don't overlap. Defaults to False.
        min_dist (float, optional): Minimum distance in meters for two stations to be considered duplicates. Defaults to 250.

    Returns:
        tuple[plt.Figure, matplotlib.axes.Axes]:
            - Figure object of the subplot.
            - Axes object.
    """

    if not isinstance(CTD, (dict, str)):
        raise TypeError(
            f"'CTD' should be a dict or a npy file string with the data, not {type(CTD).__name__}."
        )

    # read in the data (only needed if no CTD-dict, but a file was given)
    if type(CTD) is str:
        CTD = read_CTD(CTD)
    # check if data is there
    if type(CTD) is dict and len(CTD) == 0:
        raise ValueError(
            "The CTD data is empty! Please provide a valid CTD data dictionary or file."
        )

    if (
        not (pd.api.types.is_list_like(stations) or isinstance(stations, str))
        and stations is not None
    ):
        raise TypeError(
            f"'stations' should be a array_like or a string, not a {type(stations).__name__}."
        )
    if stations is not None and isinstance(stations, str):
        stations = [stations]
    if stations is not None:
        for s in stations:
            if not isinstance(s, str):
                raise TypeError(
                    f"Each station in 'stations' should be a string, not a {type(s).__name__}."
                )
    if stations is None:
        # if no stations are provided, just plot all stations
        stations = list(CTD.keys())
    else:
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

    if not isinstance(adjust_text, bool):
        raise ValueError(
            f"'adjust_text' should be an bool, not a {type(adjust_text).__name__}."
        )

    # select only stations
    CTD = {key: CTD[key] for key in stations}

    CTD_station: dict = {}
    for key in CTD.keys():
        if CTD[key]["unis_st"] in CTD_station.keys():
            CTD_station[CTD[key]["unis_st"]].append(key)
        else:
            CTD_station[CTD[key]["unis_st"]] = [key]

    CTD_duplicates: dict = {
        key: CTD_station[key] for key in CTD_station.keys() if len(CTD_station[key]) > 1
    }
    for key, value in CTD_duplicates.items():
        key_lat: list = []
        key_lon: list = []
        for st in value:
            key_lat.append(CTD[st]["lat"])
            key_lon.append(CTD[st]["lon"])
        key_lat: np.ndarray = np.array(key_lat)
        key_lon: np.ndarray = np.array(key_lon)
        if (
            np.std(key_lat) * 111.320 <= min_dist
            and np.std(key_lon) * np.mean(np.cos(np.radians(key_lat))) * 111.320
            <= min_dist
        ):
            CTD[key] = {}
            CTD[key]["lat"] = list(key_lat)[0]
            CTD[key]["lon"] = list(key_lon)[0]
            for st in value:
                del CTD[st]
                stations[stations.index(st)] = key

    lat: list = [value["lat"] for value in CTD.values()]
    lon: list = [value["lon"] for value in CTD.values()]

    if extent is None:
        std_lat, std_lon = np.std(lat), np.std(lon)
        lon_range: list = [min(lon) - std_lon, max(lon) + std_lon]
        lat_range: list = [min(lat) - std_lat, max(lat) + std_lat]
        extent: list = [lon_range[0], lon_range[1], lat_range[0], lat_range[1]]
    fig, ax = plot_empty_map(extent, topography, depth_contours)

    # add the points, and add labels
    if isinstance(st_labels, str):
        st_texts: list[str] = [st_labels + str(s) for s in stations]
    elif pd.api.types.is_list_like(st_labels):
        if len(st_labels) == len(stations):
            st_texts = st_labels
        else:
            raise ValueError(
                f"'st_labels' and 'stations' should be the same length, is {len(st_labels)} and {len(stations)}."
            )
    else:
        raise ValueError(
            f"'st_labels' should be string or array_like, not {type(st_labels).__name__}."
        )

    ax.plot(lon, lat, "xr", transform=ccrs.PlateCarree())
    texts: list = []
    for label, st_lat, st_lon in zip(st_texts, lat, lon):
        if extent[0] < st_lon < extent[1] and extent[2] < st_lat < extent[3]:
            texts.append(
                ax.text(
                    st_lon,
                    st_lat,
                    label,
                    horizontalalignment="center",
                    verticalalignment="bottom",
                )
            )

    if adjust_text:
        adj_txt(
            texts,
            expand_text=(1.2, 1.6),
            arrowprops=dict(arrowstyle="-", color="black"),
            ax=ax,
        )
    plt.gcf().canvas.draw()
    plt.tight_layout()

    return fig, ax


def plot_empty_map(
    extent: npt.ArrayLike,
    topography: str | npt.ArrayLike = None,
    depth_contours: npt.ArrayLike = [
        10,
        50,
        100,
        150,
        200,
        300,
        400,
        500,
        1000,
        2000,
        3000,
        4000,
        5000,
    ],
) -> tuple[plt.Figure, matplotlib.axes.Axes]:
    """Function which plots a very basic map of selected CTD stations.

    Args:
        extent (array_like): (4,) List of map extent. Defaults to None.
            - [lon0, lon1, lat0, lat1]
        topography (str or array_like, optional): Topography data. Defaults to None.
            Can work with:
                - Path to the example data directory.
                - '.nc' with 'lat', 'lon' and 'z'.
                - '.mat' with 'lat', 'lon' and 'D'.
                - '.npy' with an array containing lat, lon and elevation as columns
                - array_like with lat, lon and elevation
        depth_contours (array_like, optional): List containing countour levels for the bathymetry. Defaults to [10, 50, 100, 150, 200, 300, 400, 500, 1000, 2000, 3000, 4000, 5000].

    Returns:
        tuple[plt.Figure, matplotlib.axes.Axes]:
            - Figure object of the subplot.
            - Axes object.
    """

    if not pd.api.types.is_list_like(extent):
        raise ValueError(
            f"''extent' should be an array_like, not {type(extent).__name__}."
        )
    if len(extent) != 4:
        raise ValueError(f"'extent' should have the length 4, not {len(extent)}.")
    if not (extent[0] < extent[1] and extent[2] < extent[3]):
        raise ValueError(
            f"'extent' should be in the form [lon_min, lon_max, lat_min, lat_max], not {extent}."
        )

    if not pd.api.types.is_list_like(depth_contours):
        raise ValueError(
            f"'depth_contours' should be an array_like, not {type(depth_contours).__name__}."
        )

    fig, ax = plt.subplots(1, 1, subplot_kw={"projection": ccrs.PlateCarree()})
    ax.set_extent(extent)
    if isinstance(topography, str):
        if os.path.isdir(topography):
            topography = os.path.join(
                topography, "Svalbard_map_data", "bathymetry_svalbard.mat"
            )
        if not os.path.isfile(topography):
            raise FileNotFoundError(f"Topography file '{topography}' not found.")
        ext: str = os.path.splitext(topography)[-1].lower()
        if ext == ".mat":
            topo = loadmat(topography)
            topo_lat, topo_lon, topo_z = topo["lat"], topo["lon"], topo["D"]
        elif ext == ".npy":
            topo = np.load(topography)
            topo_lat, topo_lon, topo_z = topo[0], topo[1], topo[2]
        elif ext == ".nc":
            topo = Dataset(topography)
            topo_lat, topo_lon, topo_z = (
                topo.variables["lat"][:],
                topo.variables["lon"][:],
                topo.variables["z"][:],
            )
            if len(topo_lon.shape) == 1:
                topo_lon, topo_lat = np.meshgrid(topo_lon, topo_lat)
        else:
            raise ValueError(f"Unknown topography file extension: {ext}")
    elif pd.api.types.is_list_like(topography):
        # assume topography is array with 3 columns (lat,lon,z)
        topo_lat, topo_lon, topo_z = topography[0], topography[1], topography[2]
    elif topography is not None:
        raise ValueError(f"The submitted format for 'topography' is not valid.")
    if topography is not None:
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
        clabels: list = ax.clabel(BC, depth_contours, fontsize=4, fmt="%i")
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


def plot_CTD_ts(
    CTD: str | dict,
    stations: str | npt.ArrayLike = None,
    pref: int = 0,
    legend: bool = True,
) -> None:
    """Plots a TS diagram of selected stations from a CTD dataset.

    Args:
        CTD (dict or str): Either a dictionary with CTD data or a string with the path.
            - Path can lead to a .npy file or a directory with .cnv files, check read_CTD.
        stations (str or npt.ArrayLike, optional): The desired station(s). Defaults to None.
            - If None, selects all.
        pref (int, optional): Pressure to use. Defaults to 0.
            Options:
                - 0:    0 dbar
                - 1: 1000 dbar
                - 2: 2000 dbar
                - 3: 3000 dbar
                - 4: 4000 dbar
        legend (bool, optional): If the legend should be displayed. Defaults to True.

    Returns:
        None
    """
    # Check if the function has data to work with
    if not isinstance(CTD, (dict, str)):
        raise TypeError(
            f"'CTD' should be a dict or a npy file string with the data, not {type(CTD).__name__}."
        )

    # read in the data (only needed if no CTD-dict, but a file was given)
    if type(CTD) is str:
        CTD = read_CTD(CTD)
    # check if data is there
    if type(CTD) is dict and len(CTD) == 0:
        raise ValueError(
            "The CTD data is empty! Please provide a valid CTD data dictionary or file."
        )

    # Check if all stations given are found in the data
    if stations is None:
        stations = CTD.keys()
    if not pd.api.types.is_list_like(stations):
        raise TypeError(
            f"'stations' should be a list of strings, not a {type(stations).__name__}."
        )

    notfound_stations: list = [key for key in stations if not key in list(CTD.keys())]
    if len(notfound_stations) != 0:
        logging.info(
            f"The stations '{notfound_stations}' are not in the CTD data. Proceeding without them."
        )
        for i in notfound_stations:
            stations.remove(i)
        if len(stations) == 0:
            raise ValueError(f"There are no CTD stations left.")

    if not isinstance(pref, int):
        raise ValueError(f"'pref' should be an int, not {type(pref).__name__}.")
    if pref < 0 or pref > 4:
        raise ValueError(
            f"'pref' should be in a the range between 0 and 4, not {pref}."
        )

    if not isinstance(legend, bool):
        raise ValueError(f"'legend' should be a bool, not {type(legend).__name__}.")

    if len(stations) != len(CTD.keys()):
        CTD: dict = {key: CTD[key] for key in stations}

    max_S = max([np.nanmax(value["SA [g/kg]"]) for value in CTD.values()]) + 0.1
    min_S = min([np.nanmin(value["SA [g/kg]"]) for value in CTD.values()]) - 0.1

    max_T = max([np.nanmax(value["CT [degC]"]) for value in CTD.values()]) + 0.5
    min_T = min([np.nanmin(value["CT [degC]"]) for value in CTD.values()]) - 0.5

    create_empty_ts((min_T, max_T), (min_S, max_S), p_ref=pref)

    # Plot the data in the empty TS-diagram
    for station in CTD.values():
        plt.plot(
            station["SA [g/kg]"],
            station["CT [degC]"],
            linestyle="none",
            marker=".",
            label=station["unis_st"],
        )

    if len(CTD.keys()) > 1 and legend:
        plt.legend(
            bbox_to_anchor=(1.0, 1.02),
            loc="upper left",
            ncol=2,
            framealpha=1,
            columnspacing=0.7,
            handletextpad=0.4,
        )
    plt.tight_layout()

    return None


def create_empty_ts(
    T_extent: npt.ArrayLike, S_extent: npt.ArrayLike, p_ref: int = 0
) -> None:
    """Creates an empty TS-diagram to plot data into.

    Args:
        T_extent (array_like): (2, ) The minimum and maximum conservative temperature.
        S_extent (array_like): (2, ) The minimum and maximum absolute salinity.
        pref (int, optional): Pressure to use. Defaults to 0.
            Options:
                - 0:    0 dbar
                - 1: 1000 dbar
                - 2: 2000 dbar
                - 3: 3000 dbar
                - 4: 4000 dbar

    Returns:
        None
    """

    if not pd.api.types.is_list_like(T_extent):
        raise ValueError(
            f"'T_extent' should be array_like, not {type(T_extent).__name__}."
        )
    if len(T_extent) != 2:
        raise ValueError(f"'T_extent' should have the length 2, not {len(T_extent)}.")

    if not pd.api.types.is_list_like(S_extent):
        raise ValueError(
            f"'S_extent' should be array_like, not {type(S_extent).__name__}."
        )
    if len(S_extent) != 2:
        raise ValueError(f"'S_extent' should have the length 2, not {len(S_extent)}.")

    if not isinstance(p_ref, int):
        raise ValueError(f"'p_ref' should be an int, not {type(p_ref).__name__}.")
    if p_ref < 0 or p_ref > 4:
        raise ValueError(
            f"'p_ref' should be in a the range between 0 and 4, not {p_ref}."
        )

    sigma_functions = [gsw.sigma0, gsw.sigma1, gsw.sigma2, gsw.sigma3, gsw.sigma4]
    T = np.linspace(T_extent[0], T_extent[1], 100)
    S = np.linspace(S_extent[0], S_extent[1], 100)

    T, S = np.meshgrid(T, S)

    SIGMA = sigma_functions[p_ref](S, T)

    cs = plt.contour(S, T, SIGMA, colors="k", linestyles="--")
    plt.clabel(cs, fmt="%1.1f")

    plt.ylabel("Conservative Temperature [°C]")
    plt.xlabel("Absolute Salinity [g kg$^{-1}$]")
    plt.title("$\\Theta$ - $S_A$ Diagram")
    if p_ref > 0:
        plt.title("Density: $\\sigma_{" + str(p_ref) + "}$", loc="left", fontsize=10)

    return None


def check_VM_ADCP_map(ds: xr.Dataset) -> None:
    """Creates an interactive map of VM-ADCP measurements to help select time intervals for section plots.

    Args:
        ds (xr.Dataset): Dataset from CODAS or WinADCP.

    Returns:
        None
    """

    df = ds[["time", "lat", "lon"]].to_pandas()
    df["time"] = df.index

    # deprecated!!!
    fig = px.scatter_map(
        df,
        "lat",
        "lon",
        hover_data=["time"],
    )
    fig.update_layout(mapbox_style="open-street-map"),
    pplot(fig)

    return None


def plot_tidal_spectrum(
    data: pd.Series | tide, constituents: list[str] = ["M2"]
) -> tuple[plt.Figure, matplotlib.axes.Axes]:
    """Plots the tidal spectrum and marks specified tidal constituents.

    Args:
        data (pd.Series or tide): Time series of spectral data (output of 'calculate_tidal_spectrum' or a tide object).
        constituents (list[str], optional): List of tidal constituents to mark on the plot. Defaults to ["M2"].

    Returns:
        tuple[plt.Figure, matplotlib.axes.Axes]:
            - The figure of the plot.
            - The axes of the plot.
    """

    if isinstance(data, tide):
        if not hasattr(data, "spectrum"):
            raise ValueError(
                "The provided tide object does not have a 'spectrum' attribute."
            )
        else:
            data = data.spectrum

    omega_dict: dict = dict(
        zip(utide.ut_constants.const.name, utide.ut_constants.const.freq)
    )
    tidal_freqs = np.array([omega_dict[c] for c in constituents])

    fig, ax = plt.subplots(1, 1, figsize=(10, 5))
    data.plot(ax=ax, color="b", zorder=10)
    ax.grid()
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xticks(24 * tidal_freqs)  # per day
    ax.set_xticklabels(constituents)
    ax.minorticks_off()
    ax.set_xlim(right=data.index[-1])
    ax.set_ylabel("power spectral density")

    return fig, ax


def plot_map_tidal_ellipses(
    amp_major: npt.ArrayLike,
    amp_minor: npt.ArrayLike,
    inclin: npt.ArrayLike,
    constituents: npt.ArrayLike,
    lat_center: num.Number = 78.122,
    lon_center: num.Number = 14.26,
    map_extent: list[num.Number, num.Number, num.Number, num.Number] = [
        11.0,
        16.0,
        78.0,
        78.3,
    ],
    topography: str = None,
) -> tuple[plt.Figure, matplotlib.axes.Axes, Any]:
    """
    Plots tidal ellipses for specified tidal constituents on a map.

    Args:
        amp_major (array_like): Amplitudes along the major axis, one element for each specified tidal constituent.
        amp_minor (array_like): Amplitudes along the minor axis, one element for each specified tidal constituent.
        inclin (array_like): Inclination of the ellipses, one element for each specified tidal constituent.
        constituents (array_like): List with the names of the constituent, for the legend.
        lat_center (float, optional): Center latitude for the ellipses. Typically the position of the mooring that measured the data. Defaults to IS-E position.
        lon_center (float, optional): Center longitude for the ellipses. Typically the position of the mooring that measured the data. Defaults to IS-E position.
        map_extent (list[float], optional): List with [lon_min, lon_max, lat_min, lat_max]. Specifies the area limits to plot on the map. Defaults to [11.0, 16.0, 78.0, 78.3].
        topography (str or array_like, optional): Topography data. Can be:
            - Path to a .nc file (should contain 'lat', 'lon', 'z').
            - Path to a .mat file (should contain 'lat', 'lon', 'D').
            - Path to a .npy file (should contain array with lat, lon, elevation as columns).
            - Array_like with lat, lon, elevation as columns.
            - None for no bathymetry. Defaults to None.

    Returns:
        tuple[plt.Figure, matplotlib.axes.Axes, AxesHostAxes]:
            - Figure object of the map.
            - Axes object for the map.
            - Axes object for the ellipse inset.
    """

    if not pd.api.types.is_list_like(amp_major):
        raise ValueError(
            f"'amp_major' should be array_like, not {type(amp_major).__name__}."
        )
    if not pd.api.types.is_list_like(amp_minor):
        raise ValueError(
            f"'amp_minor' should be array_like, not {type(amp_minor).__name__}."
        )
    if not pd.api.types.is_list_like(inclin):
        raise ValueError(f"'inclin' should be array_like, not {type(inclin).__name__}.")
    if not pd.api.types.is_list_like(constituents):
        raise ValueError(
            f"'constituents' should be array_like, not {type(constituents).__name__}."
        )
    if (
        len(amp_major) != len(amp_minor)
        or len(amp_major) != len(inclin)
        or len(amp_major) != len(constituents)
    ):
        raise ValueError(
            f"'amp_major', 'amp_minor', 'inclin' and 'constituents' should have the same length, not {len(amp_major)}, {len(amp_minor)}, {len(inclin)} and {len(constituents)}."
        )

    if not isinstance(lat_center, num.Number):
        raise ValueError(
            f"'lat_center' should be a number, not {type(lat_center).__name__}."
        )

    if not isinstance(lon_center, num.Number):
        raise ValueError(
            f"'lon_center' should be a number, not {type(lon_center).__name__}."
        )

    fig, ax_map = plot_empty_map(extent=map_extent, topography=topography)

    phi = np.linspace(0, 2 * np.pi, 1000)
    inset_size = 2.2

    ellipse_inset = inset_axes(
        ax_map,
        width=inset_size,
        height=inset_size,
        loc="center",
        bbox_to_anchor=(lon_center, lat_center),
        bbox_transform=ax_map.transData,
        borderpad=0.0,
    )
    ellipse_inset.axis("off")
    ellipse_inset.set_facecolor("none")
    ellipse_inset.tick_params(labelleft=False, labelbottom=False)
    ellipse_inset.grid(False)
    ellipse_inset.set_aspect(1.0)

    for i, (a, b, t) in enumerate(zip(amp_major, amp_minor, inclin)):
        E = np.array([a * np.cos(phi), b * np.sin(phi)])
        t = np.radians(t)
        R_rot = np.squeeze(np.array([[np.cos(t), -np.sin(t)], [np.sin(t), np.cos(t)]]))
        E_rot = np.zeros((2, E.shape[1]))

        E_rot = R_rot @ E

        ellipse_inset.plot(E_rot[0, :], E_rot[1, :], c=f"C{i}", label=constituents[i])
        ellipse_inset.annotate(
            "",
            xy=E_rot[:, 0],
            xycoords="data",
            xytext=(0.0, 0.0),
            textcoords="data",
            arrowprops=dict(arrowstyle="-|>", connectionstyle="arc3", color=f"C{i}"),
        )

    if len(constituents) == 1:
        ax_map.title(constituents[0])
    else:
        n_col_size = len(constituents) if len(constituents) <= 7 else 7
        fig.legend(ncol=n_col_size, bbox_to_anchor=(0.5, 0.990), loc="lower center")

    return fig, ax_map, ellipse_inset


def plot_tidal_ellipses(
    amp_major: npt.ArrayLike,
    amp_minor: npt.ArrayLike,
    inclin: npt.ArrayLike,
    constituents: npt.ArrayLike,
    muliple_plots: bool = False,
    n_row_col: tuple[int, int] = None,
) -> tuple[plt.Figure, tuple[matplotlib.axes.Axes] | matplotlib.axes.Axes]:
    """Function to plot tidal ellipses in a TS-diagram.

    Args:
        amp_major (array_like): Amplitudes along the major axis, one element for each specified tidal constituent.
        amp_minor (array_like): Amplitudes along the minor axis, one element for each specified tidal constituent.
        inclin (array_like): Inclination of the ellipses, one element for each specified tidal constituent.
        constituents (array_like): List with the names of the constituent, for the legend.
        muliple_plots (bool, optional): If True, plot each constituent in a separate subplot. Defaults to False.
        n_row_col (tuple[int, int], optional): Tuple specifying (nrows, ncols) for subplots. If None, tries to guess. Defaults to None.

    Returns:
        tuple[plt.Figure, tuple[matplotlib.axes.Axes] | matplotlib.axes.Axes]:
            - Figure object.
            - Axes object(s).
    """
    if not pd.api.types.is_list_like(amp_major):
        raise ValueError(
            f"'amp_major' should be array_like, not {type(amp_major).__name__}."
        )
    if not pd.api.types.is_list_like(amp_minor):
        raise ValueError(
            f"'amp_minor' should be array_like, not {type(amp_minor).__name__}."
        )
    if not pd.api.types.is_list_like(inclin):
        raise ValueError(f"'inclin' should be array_like, not {type(inclin).__name__}.")
    if not pd.api.types.is_list_like(constituents):
        raise ValueError(
            f"'constituents' should be array_like, not {type(constituents).__name__}."
        )
    if (
        len(amp_major) != len(amp_minor)
        or len(amp_major) != len(inclin)
        or len(amp_major) != len(constituents)
    ):
        raise ValueError(
            f"'amp_major', 'amp_minor', 'inclin' and 'constituents' should have the same length, not {len(amp_major)}, {len(amp_minor)}, {len(inclin)} and {len(constituents)}."
        )

    if not isinstance(muliple_plots, bool):
        raise ValueError(
            f"'muliple_plots' should be a bool, not {type(muliple_plots).__name__}."
        )

    if n_row_col is not None and not isinstance(n_row_col, tuple):
        raise ValueError(
            f"'n_row_col' should be a tuple with two integers, not {type(n_row_col).__name__}."
        )
    if n_row_col is not None and len(n_row_col) != 2:
        raise ValueError(
            f"'n_row_col' should be a tuple with two integers, not {len(n_row_col)}."
        )

    phi = np.linspace(0, 2 * np.pi, 1000)
    if muliple_plots:
        if len(constituents) <= 1:
            raise ValueError(
                "If 'muliple_plots' is True, 'constituents' should have more than one element."
            )
        if n_row_col is None:
            if len(constituents) % 3 == 0:
                n_row_col = (len(constituents) // 3, 3)
            elif len(constituents) % 4 == 0:
                n_row_col = (len(constituents) // 4, 4)
            elif len(constituents) % 2 == 0:
                n_row_col = (len(constituents) // 2, 2)
            else:
                n_row_col = (len(constituents) // 3 + 1, 3)
        elif not isinstance(n_row_col, tuple) or len(n_row_col) != 2:
            raise ValueError(
                f"'n_row_col' should be a tuple with two integers, not {n_row_col}."
            )
        fig, axes = plt.subplots(
            ncols=n_row_col[1],
            nrows=n_row_col[0],
            figsize=(n_row_col[1] * 3, n_row_col[0] * 3),
        )
        flat_axes = axes.flatten()
        max_a = np.max(amp_major) * 1.1
    else:
        fig, axes = plt.subplots(ncols=1, nrows=1, figsize=(3, 3))
        flat_axes = [axes]
        max_a = np.max(amp_major) * 1.1

    for i, (a, b, t, n) in enumerate(zip(amp_major, amp_minor, inclin, constituents)):
        if not muliple_plots:
            ind = 0  # always plot in the first axes
        else:
            ind: int = i  # as index for the axes
            i = 0  # same color for all ellipses
            flat_axes[ind].set_xlim(-max_a, max_a)
            flat_axes[ind].set_ylim(-max_a, max_a)
            flat_axes[ind].set_aspect("equal")
            flat_axes[ind].grid()

        E = np.array([a * np.cos(phi), b * np.sin(phi)])
        R_rot = np.squeeze(np.array([[np.cos(t), -np.sin(t)], [np.sin(t), np.cos(t)]]))
        E_rot = np.zeros((2, E.shape[1]))
        E_rot = R_rot @ E

        flat_axes[ind].plot(E_rot[0, :], E_rot[1, :], label=n, color=f"C{i}")
        flat_axes[ind].annotate(
            "",
            xy=(E_rot[0, 0], E_rot[1, 0]),
            xycoords="data",
            xytext=(0.0, 0.0),
            textcoords="data",
            arrowprops=dict(arrowstyle="-|>", connectionstyle="arc3", color=f"C{i}"),
        )

        if muliple_plots:
            flat_axes[ind].set_title(n)

    if not muliple_plots:
        axes.grid()
        n_col_size: int = len(constituents) if len(constituents) <= 7 else 7
        fig.legend(ncol=n_col_size, bbox_to_anchor=(0.5, 0.90), loc="lower center")
    else:
        fig.subplots_adjust(hspace=0.3, wspace=0.3)

    return fig, axes


def plot_tidal_time_series(
    data: npt.ArrayLike,
    time: npt.ArrayLike,
    transparent_data: npt.ArrayLike | None = None,
    data_args: list[dict] | dict | None = None,
    transparent_data_args: dict | None = None,
    label: str | list[str] | None = None,
    transparent_labels: str | list[str] | None = None,
    moon_phase: list[list, list] | bool = True,
    moon_apsides: list[list, list] | bool = True,
    figsize: tuple[int, int] = None,
) -> tuple[plt.figure, matplotlib.axes.Axes]:
    """Plots a time series of tidal data with optional transparency and moon phase/orbit overlays.

    Args:
        data (array_like): Data to be plotted, can be 1D or 2D.
        time (array_like): Time values corresponding to the data, should match the length of data.
        transparent_data (array_like or None, optional): Data to be plotted ontop of 'data'. Defaults to None.
        data_args (list[dict] or dict or None, optional): Arguments to give to plt.plot. Defaults to None.
            - If None, uses default arguments.
            - If dict, applies the same arguments to all data.
            - If list of dicts, applies each dict to the corresponding data.
        transparent_data_args (dict or None, optional): Arguments to give to plt.plot. Defaults to None.
            - If None, uses default arguments.
            - If dict, applies the same arguments to all transparent data.
            - If list of dicts, applies each dict to the corresponding transparent data.
        label (str or list[str] or None, optional): Label for the normal data. Defaults to None.
            - If None, no label is used.
            - If str, applies the same label to all data.
            - If list of str, applies each label to the corresponding data.
        transparent_labels (str or list[str] or None, optional): Label for the transparent data. Defaults to None.
            - If None, no label is used.
            - If str, applies the same label to all transparent data.
            - If list of str, applies each label to the corresponding transparent data.
        moon_phase (list[list, list] or bool, optional): Used to include the moon phases (full and new). Defaults to True.
            - If True, uses the default moon phases.
            - If list, should contain two lists with the full and new moon dates.
            - If False, no moon phases are plotted.
        moon_apsides (list[list, list] or bool, optional): Used to include the apsides (apogee and perigee). Defaults to True.
            - If True, uses the default apsides.
            - If list, should contain two lists with the apogee and perigee dates.
            - If False, no apsides are plotted.
        figsize (tuple[int, int], optional): Sets the used figsize. Defaults to None.
            - If None, uses the default figsize of (10, 5 * number_of_subplots / 1.5).
            - Also used to set the size of markers in the plot.

    Returns:
        tuple[plt.figure, matplotlib.axes.Axes]:
            - Figure object of the plot.
            - Axes object of the plot.
    """

    if not pd.api.types.is_list_like(data):
        raise ValueError(f"'data' should be array_like, not {type(data).__name__}.")
    data = np.asarray(data)
    number = 1 if data.ndim == 1 else data.shape[0]

    if not pd.api.types.is_list_like(time):
        raise ValueError(f"'time' should be array_like, not {type(time).__name__}.")
    if data.ndim == 1 and len(time) != len(data):
        raise ValueError(
            f"'data' and 'time' should have the same length, not {len(data)} and {len(time)}."
        )
    elif data.ndim == 2 and len(time) != data.shape[1]:
        raise ValueError(
            f"'data' and 'time' should have the same length, not {data.shape[1]} and {len(time)}."
        )
    elif data.ndim > 2:
        raise ValueError(f"'data' should be 1D or 2D array_like, not {data.ndim}.")

    if transparent_data is not None:
        if not pd.api.types.is_list_like(transparent_data):
            raise ValueError(
                f"'transparent_data' should be array_like, not {type(transparent_data).__name__}."
            )
        transparent_data = np.asarray(transparent_data)
        if transparent_data.ndim == 1 and len(time) != len(transparent_data):
            raise ValueError(
                f"'transparent_data' and 'time' should have the same length, not {len(transparent_data)} and {len(time)}."
            )
        elif transparent_data.ndim == 2 and len(time) != transparent_data.shape[1]:
            raise ValueError(
                f"'transparent_data' and 'time' should have the same length, not {transparent_data.shape[1]} and {len(time)}."
            )
        elif transparent_data.ndim > 2:
            raise ValueError(
                f"'transparent_data' should be 1D or 2D array_like, not {transparent_data.ndim}."
            )

    if data_args is None:
        data_args = [{} for _ in range(number)]
    elif isinstance(data_args, dict):
        data_args = [data_args for _ in range(number)]
    elif pd.api.types.is_list_like(data_args):
        if len(data_args) != number:
            raise ValueError(
                f"'data_args' should have the same length as 'data', not {len(data_args)}."
            )
        for i in data_args:
            if not isinstance(i, dict):
                raise ValueError(
                    f"Each element in 'data_args' should be a dict, not {type(i).__name__}."
                )
    else:
        raise ValueError(
            f"'data_args' should be a dict or a list of dicts, not {type(data_args).__name__}."
        )

    if transparent_data is not None and transparent_data_args is None:
        transparent_data_args = [{} for _ in range(number)]
    elif isinstance(transparent_data_args, dict):
        transparent_data_args = [transparent_data_args for _ in range(number)]
    elif pd.api.types.is_list_like(transparent_data_args):
        if len(transparent_data_args) != number:
            raise ValueError(
                f"'transparent_data_args' should have the same length as 'data', not {len(transparent_data_args)}."
            )
        for i in transparent_data_args:
            if not isinstance(i, dict):
                raise ValueError(
                    f"Each element in 'transparent_data_args' should be a dict, not {type(i).__name__}."
                )
    elif transparent_data is not None:
        raise ValueError(
            f"'transparent_data_args' should be a dict or a list of dicts, not {type(transparent_data_args).__name__}."
        )

    if not isinstance(label, (str, list, type(None))):
        raise ValueError(
            f"'label' should be a string, list of strings or None, not {type(label).__name__}."
        )
    if isinstance(label, str):
        label = [label for _ in range(number)]
    label = label if label is not None else ["" for _ in range(number)]
    if len(label) != number:
        raise ValueError(
            f"'label' should have the same length as 'data', not {len(label)}."
        )

    if transparent_data is not None:
        if isinstance(transparent_labels, str):
            transparent_labels = [transparent_labels for _ in range(number)]
        elif transparent_labels is None:
            transparent_labels = ["" for _ in range(number)]
        elif not pd.api.types.is_list_like(transparent_labels):
            raise ValueError(
                f"'transparent_labels' should be a string, list of strings or None, not {type(transparent_labels).__name__}."
            )
        if len(transparent_labels) != number:
            raise ValueError(
                f"'transparent_labels' should have the same length as 'data', not {len(transparent_labels)}."
            )

    if not isinstance(moon_phase, (list, bool)):
        raise ValueError(
            f"'moon_phase' should be a list or bool, not {type(moon_phase).__name__}."
        )
    elif isinstance(moon_phase, list):
        for i in moon_phase:
            try:
                pd.to_datetime(i)
            except (ValueError, TypeError):
                raise ValueError(
                    f"Each element in 'moon_phase' should be a valid date string or datetime object, not {i}."
                )
    if moon_phase is not False:
        if isinstance(moon_phase, list):
            moon_phase = (
                pd.to_datetime(moon_phase)
                .sort_values()
                .clip(lower=min(time), upper=max(time))
            )
        else:
            current = eph.Date(min(time))
            end = eph.Date(max(time))
            full_moons = []
            new_moons = []

            next_full = eph.next_full_moon(current)
            next_new = eph.next_new_moon(current)

            while next_full < end:
                full_moons.append(eph.localtime(next_full))
                next_full = eph.next_full_moon(next_full)

            while next_new < end:
                new_moons.append(eph.localtime(next_new))
                next_new = eph.next_new_moon(next_new)
            moon_phase = [full_moons, new_moons]

    if not isinstance(moon_apsides, (list, bool)):
        raise ValueError(
            f"'moon_apsides' should be a list or bool, not {type(moon_apsides).__name__}."
        )
    elif isinstance(moon_apsides, list):
        for i in moon_apsides:
            try:
                pd.to_datetime(i)
            except (ValueError, TypeError):
                raise ValueError(
                    f"Each element in 'moon_apsides' should be a valid date string or datetime object, not {i}."
                )
    if moon_apsides is not False:
        if isinstance(moon_apsides, list):
            moon_apsides = pd.to_datetime(moon_apsides).sort_values()
            moon_apsides = moon_apsides[
                (moon_apsides >= min(time)) & (moon_apsides <= max(time))
            ]

        else:
            observer = eph.Observer()
            dates = []
            distances = []
            current = eph.Date(min(time)) - 30
            end = eph.Date(max(time)) + 30

            while current < end:
                observer.date = current
                moon = eph.Moon(observer)
                dates.append(eph.localtime(current))
                distances.append(moon.earth_distance)
                current += 0.1

            distances = np.array(distances)
            dates = np.array(dates)

            apogee_indices, _ = signal.find_peaks(distances, distance=150)
            apogee_dates = []
            # debuging
            # apogee_dists = []
            for i in apogee_indices:
                apogee_dates.append(dates[i])
                # apogee_dists.append(distances[i])

            perigee_indices, _ = signal.find_peaks(-distances, distance=150)
            perigee_dates = []
            # debuging
            # perigee_dists = []
            for i in perigee_indices:
                perigee_dates.append(dates[i])
                # perigee_dists.append(distances[i])

            apogee_dates = pd.to_datetime(apogee_dates)
            apogee_dates = apogee_dates[
                (apogee_dates >= min(time)) & (apogee_dates <= max(time))
            ]

            perigee_dates = pd.to_datetime(perigee_dates)
            perigee_dates = perigee_dates[
                (perigee_dates >= min(time)) & (perigee_dates <= max(time))
            ]
            moon_apsides = [
                list(apogee_dates.to_pydatetime()),
                list(perigee_dates.to_pydatetime()),
            ]

    if figsize is None:
        figsize = (10, 5 * number / 1.5)
    if not isinstance(figsize, tuple):
        raise ValueError(
            f"'figsize' should be a tuple of two integers, not {type(figsize).__name__}."
        )
    if len(figsize) != 2:
        raise ValueError(
            f"'figsize' should be a tuple of two integers, not {len(figsize)}."
        )

    fig, axes = plt.subplots(number, 1, figsize=figsize, sharex=True)
    if number == 1:
        axes = [axes]
    for i, (ax, data_arg, name) in enumerate(zip(axes, data_args, label)):
        data_arg.setdefault("color", f"C{i*2}")
        data_arg.setdefault("label", name)
        if transparent_data is not None:
            data_arg.setdefault("alpha", 0.5)
        if number > 1:
            ax.plot(time, data[i, :], **data_arg)
            data_max = np.nanmax(data[i, :])
            data_min = np.nanmin(data[i, :])
        else:
            ax.plot(time, data, **data_arg)
            data_max = np.nanmax(data)
            data_min = np.nanmin(data)

        if transparent_data is not None:
            transparent_data_args[i].setdefault("color", f"C{i*2+1}")
            transparent_data_args[i].setdefault("alpha", 0.7)
            transparent_data_args[i].setdefault("label", transparent_labels[i])
            if number > 1:
                ax.plot(
                    time,
                    transparent_data[i, :],
                    **transparent_data_args[i],
                )
                if np.nanmax(transparent_data[i, :]) > data_max:
                    data_max = np.nanmax(transparent_data[i, :])
                if np.nanmin(transparent_data[i, :]) < data_min:
                    data_min = np.nanmin(transparent_data[i, :])
            else:
                ax.plot(
                    time,
                    transparent_data,
                    **transparent_data_args[i],
                )
                if np.nanmax(transparent_data) > data_max:
                    data_max = np.nanmax(transparent_data)
                if np.nanmin(transparent_data) < data_min:
                    data_min = np.nanmin(transparent_data)

        if len(name) != 0:
            ax.legend()

        if moon_phase is not False or moon_apsides is not False:
            marker_size = figsize[0] * figsize[1] * 1.5 / number
        if moon_phase is not False:
            y_pos = data_max + (data_max - data_min) * 0.05
            full_m = ax.scatter(
                moon_phase[0],
                np.full_like(moon_phase[0], y_pos),
                edgecolor="black",
                color="black",
                s=marker_size,
                marker="o",
                label="Full Moon",
            )
            new_m = ax.scatter(
                moon_phase[1],
                np.full_like(moon_phase[1], y_pos),
                edgecolor="black",
                color="white",
                s=marker_size,
                marker="o",
                label="New Moon",
            )

        if moon_apsides is not False:
            if moon_phase is not False:
                y_pos = data_max + (data_max - data_min) * 0.1
            else:
                y_pos = data_max + (data_max - data_min) * 0.05
            apo = ax.scatter(
                moon_apsides[0],
                np.full_like(moon_apsides[0], y_pos),
                color="red",
                s=marker_size,
                marker="+",
                label="Apogee",
            )
            peri = ax.scatter(
                moon_apsides[1],
                np.full_like(moon_apsides[1], y_pos),
                color="blue",
                s=marker_size,
                marker="1",
                label="Perigee",
            )

        ax.xaxis.grid(True)

    handels: list = []
    labels: list = []
    if moon_apsides is not False:
        handels.append(apo)
        labels.append(apo.get_label())
        handels.append(peri)
        labels.append(peri.get_label())
    if moon_phase is not False:
        handels.append(full_m)
        labels.append(full_m.get_label())
        handels.append(new_m)
        labels.append(new_m.get_label())
    if len(handels) > 0:
        fig.legend(
            handles=handels,
            labels=labels,
            loc="upper left",
            bbox_to_anchor=(0.99, 0.94),
        )

    fig.tight_layout(h_pad=0.2, rect=[0, 0, 1, 0.95])

    return fig, axes


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
