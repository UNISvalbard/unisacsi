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
import unisacsi
from seabird.cnv import fCNV
import gsw
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from netCDF4 import Dataset
import glob
from scipy.interpolate import interp1d,griddata
import scipy.io as spio
from scipy.io import loadmat
from matplotlib.dates import date2num,datestr2num
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
from mpl_toolkits.axes_grid1.inset_locator import InsetPosition
import uptide
import spectrum
from scipy import signal

import re
import pathlib
import zipfile
import posixpath
import pyTMD.utilities


############################################################################
# MISCELLANEOUS FUNCTIONS
############################################################################
def cal_dist_dir_on_sphere(longitude, latitude):
    """
    function to calculate a series of distances between
    coordinate points (longitude and latitude)
    of the drifter between sequential timesteps

    Parameters
    ----------
    longitude : pd.Series
         time Series of logitudinal coordinates [deg] of the ship
    latitude : pd.Series
        time Series of latitudinal coordinates [deg] of the ship

    Returns
    -------
    speed : pd.Series
        speed the drifter travelled between each of the timesteps
    heading : pd.Series
        direction drifter headed between each of the timesteps

    """

    # Define the Earths Radius (needed to estimate distance on Earth's sphere)
    R = 6378137. # [m]

    # Convert latitude and logitude to radians
    lon = longitude * np.pi/180.
    lat = latitude  * np.pi/180.

    # Calculate the differential of lon and lat between the timestamps
    dlon = lon.diff()
    dlat = lat.diff()

    # Create a shifted time Series
    lat_t1 = lat.shift(periods=1)
    lat_t2 = lat.copy()

    # Calculate interim stage
    alpha = np.sin(dlat/2.)**2 + np.cos(lat_t1) * np.cos(lat_t2) * np.sin(dlon/2.)**2

    distance = 2*R*np.arctan2(np.sqrt(alpha),np.sqrt(1-alpha))#(np.arcsin(np.sqrt(alpha))

    time_delta = pd.Series((lat.index[1:]-lat.index[0:-1]).seconds, index = lat.index[1::])
    speed = (distance/time_delta)

    # Calculate the ships heading
    arg1 = np.sin(dlon) * np.cos(lat_t2)
    arg2 = np.cos(lat_t1) * np.sin(lat_t2) -np.sin(lat_t1) * np.cos(lat_t2) * np.cos(dlon)

    heading = np.arctan2(arg1,arg2) * (-180./np.pi) + 90.0
    heading[heading<0.0] = heading + 360.
    heading[heading>360.0] = heading - 360.

    return speed, heading



def cart2pol(u,v,ctype='math'):
    '''
    Converts cartesian velocity (u,v) to polar velocity (angle,speed),
    using either
    1) mathematical
    2) oceanographical, or
    3) meteorological
    definition.
    Parameters
    ----------
    u : numeric, or array-like
        u-Component of velocity.
    v : numeric, or array-like
        v-Component of velocity.
    ctype : string, optional
        Type of definitition, 'math', 'ocean' or 'meteo'. The default is 'math'.
    Returns
    -------
    angle : numeric, or array-like
        Angle of polar velocity.
    speed : numeric, or array-like
        Speed of polar velocity.
    '''
    speed = np.sqrt(u**2 + v**2)
    if ctype == 'math':
        angle = 180/np.pi* np.arctan2(v,u)
    if ctype in ['meteo','ocean']:
        angle = 180 / np.pi * np.arctan2(u,v)
        if ctype == 'meteo':
            angle = (angle+180)%360

    return angle,speed



def pol2cart(angle,speed,ctype='math'):
    '''
    Converts polar velocity (angle,speed) to cartesian velocity (u,v),
    using either
    1) mathematical
    2) oceanographical, or
    3) meteorological
    definition.
    Parameters
    ----------
    angle : numeric, or array-like
        Angle of polar velocity.
    speed : numeric, or array-like
        Speed of polar velocity.
    ctype : string, optional
        Type of definitition, 'math', 'ocean' or 'meteo'. The default is 'math'.
    Returns
    -------
    u : numeric, or array-like
        u-Component of velocity.
    v : numeric, or array-like
        v-Component of velocity.
    '''
    if ctype == 'math':
        u = speed * np.cos(angle*np.pi/180.)
        v = speed * np.sin(angle*np.pi/180.)
    elif ctype == 'meteo':
        u = -speed * np.sin(angle*np.pi/180.)
        v = -speed * np.cos(angle*np.pi/180.)
    elif ctype == 'ocean':
        u = speed * np.sin(angle*np.pi/180.)
        v = speed * np.cos(angle*np.pi/180.)

    return u,v



def create_latlon_text(lat,lon):
    '''
    Creates two strings which contain a text for latitude and longitude
    Parameters
    ----------
    lat : scalar
        latitude.
    lon : scalar
        longitude.
    Returns
    -------
    latstring : str
        the string for the latitude.
    lonstring : str
        the string for the longitude.
    '''
    lat_minutes = str(np.round((np.abs(lat - int(lat)))*60,5))
    if lat < 0:
        lat_letter = 'S'
    else:
        lat_letter = 'N'
    latstring = str(int(np.abs(lat)))+ ' ' + lat_minutes + ' ' + lat_letter

    lon_minutes = str(np.round((np.abs(lon - int(lon)))*60,5))
    if lon < 0:
        lon_letter = 'W'
    else:
        lon_letter = 'E'
    lonstring = str(int(np.abs(lon)))+ ' ' + lon_minutes + ' ' + lon_letter

    return latstring,lonstring



def CTD_to_grid(CTD,stations=None,interp_opt= 1,x_type='distance',z_fine=False):
    '''
    This function accepts a CTD dict of dicts, finds out the maximum
    length of the depth vectors for the given stations, and fills all
    fields to that maximum length, using np.nan values.
    Parameters
    ----------
    CTD : dict of dicts
        CTD data. Is created by `read_CTD`
    stations : array_like, optional
        list of stations to select from `CTD`.
    interp_opt : int, optional
        flag how to interpolate over X (optional).
                     0: no interpolation,
                     1: linear interpolation, fine grid (default),
                     2: linear interpolation, coarse grid. The default is 1.
    x_type : str, optional
        whether X is 'time' or 'distance'. The default is 'distance'.
    z_fine: Whether to use a fine z grid. If True, will be 10 cm, otherwise 1 m
    Returns
    -------
    fCTD : dict
        dict with the gridded CTD data.
    Z : array_like
        common depth vector.
    X : array_like
        common X vector.
    station_locs : array_like
        locations of the stations as X units.
    '''


    # if no stations are given, take all stations available
    if stations is None:
        stations = list(CTD.keys())
    else:
        CTD = {key:CTD[key] for key in stations}

    # construct the Z-vector from the max and min depth of the given stations
    maxdepth = np.nanmax([np.nanmax(-CTD[i]['z']) for i in stations])
    mindepth = np.nanmin([np.nanmin(-CTD[i]['z']) for i in stations])
    if z_fine:
        Z = np.linspace(mindepth,maxdepth,int((maxdepth-mindepth)*10)+1)
    else:
        Z = np.linspace(mindepth,maxdepth,int(maxdepth-mindepth)+1)

    # construct the X-vector, either distance or time
    if x_type == 'distance':
        LAT = np.asarray([d['LAT'] for d in CTD.values()])
        LON = np.asarray([d['LON'] for d in CTD.values()])
        X = np.insert(np.cumsum(gsw.distance(LON,LAT)/1000),0,0)
    elif x_type == 'time':
        X = np.array([date2num(d['datetime']) for d in CTD.values()])
        X = (X - X[0])*24

    # this X vector is where the stations are located, so save that
    station_locs = X[:]
    fields = list(set([field for field in CTD[stations[0]]
                        if np.size(CTD[stations[0]][field]) > 1]))
    

    # original grids
    X_orig,Z_orig = [f.ravel() for f in np.meshgrid(X,Z)]
    # new grids in case of 2-d interpolation
    if interp_opt == 1:
        X_int = np.linspace(np.min(X),np.max(X),len(X)*20) # create fine X grid
        Z_int = Z[:]
    elif interp_opt == 2:
        X_int = np.linspace(np.min(X),np.max(X),20) # create coarse X grid
        Z_int = np.linspace(mindepth,maxdepth,50)

    fCTD = {}
    for field in fields:
        try:
            # grid over Z
            temp_array = []
            for value in CTD.values():
                if field in value:
                    temp_array.append(interp1d(-value['z'],value[field],
                                            bounds_error=False)(Z))
                else:
                    temp_array.append(interp1d(Z,Z*np.nan,
                                            bounds_error=False)(Z))
            temp_array = np.array(temp_array).transpose()

            if interp_opt == 0: # only grid over Z
                fCTD[field] = temp_array
            else: # grid over Z and X
                temp_array = temp_array.ravel()
                mask = np.where(~np.isnan(temp_array)) # NaN mask
                # grid in X and Z
                fCTD[field] = griddata((X_orig[mask],Z_orig[mask]), # old grid
                                     temp_array[mask], # data
                                     tuple(np.meshgrid(X_int,Z_int))) # new grid
            
            if field == "water_mass":
                fCTD["water_mass"] = np.round(fCTD["water_mass"])
        except:
            print('Warning: No gridding possible for '+field+'. Maybe ' \
                      'no valid data? Setting to nan...')
            if interp_opt == 0:
                fCTD[field] = np.meshgrid(X,Z)[0] * np.nan
            else:
                fCTD[field] = np.meshgrid(X_int,Z_int)[0] * np.nan

    if interp_opt > 0:
        X,Z = X_int,Z_int

    return fCTD,Z,X,station_locs



def CTD_to_xarray(CTD,switch_xdim='station'):
    """
    Function to store CTD data in a xarray dataset instead of a dictionary.

    Parameters
    ----------
    CTD : dict of dicts
        CTD data. Is created by `read_CTD`
    switch_xdim : str	
	Keyword to switch between time and station as x dimension for the returned data set. Default is station (UNIS station number). 

    Returns
    -------
    ds : xarray dataset with two dimensions depth and distance along the section, and all measured variables

    """

    #CTD_i,Z,X,_ = CTD_to_grid(CTD,stations,interp_opt=0)

# take all stations available
    stations = list(CTD.keys())

    # construct the Z-vector from the max and min depth of the given stations
    maxdepth = np.nanmax([np.nanmax(-CTD[i]['z']) for i in stations])
    mindepth = np.nanmin([np.nanmin(-CTD[i]['z']) for i in stations])

    Z = np.linspace(mindepth,maxdepth,int(maxdepth-mindepth)+1)


    # collect station numbers and other metadata
    ship_station = np.array([d['st'] for d in CTD.values()])
    station = np.array([d['unis_st'] for d in CTD.values()])
    lat = np.array([d['LAT'] for d in CTD.values()])
    lon = np.array([d['LON'] for d in CTD.values()])
    bdepth = np.array([d['BottomDepth'] for d in CTD.values()])

    # construct the X-vector
    X = np.array([d['datetime'] for d in CTD.values()])
    #X = pd.to_datetime(X-719529., unit='D').round('1s')

    fields = list(set([field for field in CTD[stations[0]]
                        if np.size(CTD[stations[0]][field]) > 1]))
    
    fCTD = {}
    for field in fields:
        try:
            # grid over Z
            temp_array = []
            for value in CTD.values():
                if field in value:
                    temp_array.append(interp1d(-value['z'],value[field],
                                            bounds_error=False)(Z))
                else:
                    temp_array.append(interp1d(Z,Z*np.nan,
                                            bounds_error=False)(Z))
            temp_array = np.array(temp_array).transpose()

   
            fCTD[field] = temp_array
           
            
            if field == "water_mass":
                fCTD["water_mass"] = np.round(fCTD["water_mass"])
        except:
            print('Warning: No gridding possible for '+field+'. Maybe ' \
                      'no valid data? Setting to nan...')
         
            fCTD[field] = np.ones([len(X),len(Z)]) * np.nan


    list_da = []
    for vari in fCTD.keys():
        list_da.append(xr.DataArray(data=fCTD[vari], dims=["depth", "time"], coords={"depth": Z, "time": X, "station": ("time", station), "ship_station": ("time", ship_station), "lat": ("time", lat), "lon": ("time", lon), "bottom_depth": ("time", bdepth)}, name=vari))
    
    ds = xr.merge(list_da)

    ds = ds.sortby("time")
    ds = ds.interp(depth=np.arange(np.ceil(ds.depth[0]), np.floor(ds.depth[-1])+1.))

    if switch_xdim == 'station':
        ds = ds.swap_dims({'time':'station'})

    ds["SA"].attrs["long_name"] = "Absolute Salinity [g/kg]"
    ds["S"].attrs["long_name"] = "Salinity [PSU]"
    ds["CT"].attrs["long_name"] = "Conservative Temperature [°C]"
    ds["T"].attrs["long_name"] = "Temperature [°C]"
    ds["C"].attrs["long_name"] = "Conductivity [S/cm]"
    ds["P"].attrs["long_name"] = "Pressure [dbar]"
    ds["SIGTH"].attrs["long_name"] = "Density (sigma-theta) [kg/m3]"
    ds["OX"].attrs["long_name"] = "Oxygen [ml/l]"

    return ds


def section_to_xarray(ds,stations=None,time_periods=None):
    """
    Function to extract one section from the CTD/ADCP dataset from the whole cruise and return a new dataset, where distance along the section is the new dimension.
    
    Parameters
    ----------
    ds : xarray dataset
        Data from CTD or ADCP, read and transformed with the respective functions (see example notebook).
    stations : list	
	    List with the UNIS station numbers in the section. This is used for CTD and LADCP.
    time_preiods : list
        List with the start and end points for each time period that contributes to the section. This is used for the VM-ADCPs. 

    Returns
    -------
    ds : xarray dataset with two dimensions depth and distance along the section, and all measured variables

    """

    if ((stations == None) & (time_periods != None)):   # for VM-ADCP
        ds_section = []
        for (start, end) in time_periods:
            ds_section.append(ds.sel(time=slice(start, end)))
        ds_section = xr.concat(ds_section, dim="time")
        if time_periods[0][0] > time_periods[-1][0]:
            ds_section = ds_section.sortby('time', ascending =False) 	

        ds_section['distance'] = xr.DataArray(np.insert(np.cumsum(gsw.distance(ds_section.lon.values,ds_section.lat.values)/1000),0,0),dims = ['time'],coords={'distance':ds_section.time})
        ds_section = ds_section.swap_dims({'time':'distance'}).dropna("depth", "all")
        ds_section = ds_section.transpose("depth", "distance")
        return ds_section

    
    elif ((stations != None) & (time_periods == None)):       # for CTD and L-ADCP
        ds_section = ds.sel(station = stations)	
        ds_section['distance'] = xr.DataArray(np.insert(np.cumsum(gsw.distance(ds_section.lon.values,ds_section.lat.values)/1000),0,0),dims = ['station'],coords={'distance':ds_section.station})
        ds_section = ds_section.swap_dims({'station':'distance'}).dropna("depth", "all")
        ds_section = ds_section.transpose("depth", "distance")
        return ds_section
        
    else:
        print("Please specify either stations (for CTD and L-ADCP) or time_periods (for VM_ADCP)!")
        return None
    
    
    

    
    




def mooring_into_xarray(dict_of_instr):
    """
    Function to store mooring data (T, S, SIGTH) from a mooring in an xarray dataset. The returned dataset can be regridded onto a regular time/depth grid using the xarray methods interpolate_na and interp.

    Parameters
    ----------
    dict_of_instr : dictionary with the dataframes returned from the respective read functions for the different instruments, keys: depth levels

    Returns
    -------
    ds : xarray dataset with two dimensions depth and time, and three variables T, S and SIGTH.

    """
    
    all_varis = ["T", "S", "SIGTH"]
    
    for d in dict_of_instr.keys():
        varis_instr = [v for v in all_varis if v in list(dict_of_instr[d].columns)]
        dict_of_instr[d] = dict_of_instr[d][varis_instr]
    
    
    list_da = []
    for vari in all_varis:
        list_df = []
        for d, df_instr in dict_of_instr.items():
            if vari in list(df_instr.keys()):
                list_df.append(df_instr[vari].rename(d))
        df_vari = pd.concat(list_df, axis=1)
        df_vari = df_vari.resample("20min").mean()
                
        list_da.append(xr.DataArray(data=df_vari, dims=["time", "depth"], coords={"depth": np.array(list(df_vari.columns), dtype=float), "time": df_vari.index.values}, name=vari))
    
    ds = xr.merge(list_da)
    
    return ds
	

def calc_freshwater_content(salinity,depth,ref_salinity=34.8):
    '''
    Calculates the freshwater content from a profile of salinity and depth.

    Parameters
    ----------
    salinity : array-like
        The salinity vector.
    depth : TYPE
        The depth vector.
    ref_salinity : float, optional
        The reference salinity. The default is 34.8.
    Returns
    -------
    float
        The freshwater content for the profile, in meters
    '''
    try:
        idx = np.where(salinity>ref_salinity)[0][0]
        salinity = salinity[:idx]
        depth = depth[:idx]
    except:
        pass

    salinity = np.mean([salinity[1:],salinity[:-1]])

    dz = np.diff(depth)

    return np.sum((salinity-ref_salinity)/ref_salinity *dz)



def myloadmat(filename):
    '''
    this function should be called instead of direct spio.loadmat
    as it cures the problem of not properly recovering python dictionaries
    from mat files. It calls the function check keys to cure all entries
    which are still mat-objects
    '''
    def _check_keys(d):
        '''
        checks if entries in dictionary are mat-objects. If yes
        todict is called to change them to nested dictionaries
        '''
        for key in d:
            if isinstance(d[key], spio.matlab.mat_struct):
                d[key] = _todict(d[key])
        return d

    def _todict(matobj):
        '''
        A recursive function which constructs from matobjects nested dictionaries
        '''
        d = {}
        for strg in matobj._fieldnames:
            elem = matobj.__dict__[strg]
            if isinstance(elem, spio.matlab.mat_struct):
                d[strg] = _todict(elem)
            elif isinstance(elem, np.ndarray):
                d[strg] = _tolist(elem)
            else:
                d[strg] = elem
        return d

    def _tolist(ndarray):
        '''
        A recursive function which constructs lists from cellarrays
        (which are loaded as numpy ndarrays), recursing into the elements
        if they contain matobjects.
        '''
        elem_list = []
        for sub_elem in ndarray:
            if isinstance(sub_elem, spio.matlab.mat_struct):
                elem_list.append(_todict(sub_elem))
            elif isinstance(sub_elem, np.ndarray):
                elem_list.append(_tolist(sub_elem))
            else:
                elem_list.append(sub_elem)
        return np.asarray(elem_list)

    data = spio.loadmat(filename, struct_as_record=False, squeeze_me=True)
    return _check_keys(data)



def mat2py_time(matlab_dnum):
    '''
    Converts matlab datenum to python datetime objects
    Parameters
    ----------
    matlab_dnum : int
        The matlab datenum.
    Returns
    -------
    pydate : datetime object
        The python datetime
    '''
    return pd.to_datetime(np.asarray(matlab_dnum)-719529, unit='D').round('1s')
    # try:
    #     len(matlab_dnum)
    # except:
    #     matlab_dnum = [matlab_dnum]
    # return [datetime.fromordinal(int(t)) + timedelta(days=t%1) - \
    #                         timedelta(days = 366) for t in matlab_dnum]


def present_dict(d,offset=''):
    '''
    Iterative function to present the contents of a dictionary. Prints in
    the terminal.
    Parameters
    ----------
    d : dict
        The dictionary.
    offset : str, optional
        Offset used for iterative calls. The default is ''.
    Returns
    -------
    None.
    '''
    if len(d.keys()) > 50:
        print(offset,'keys:',list(d.keys()))
        print(offset,'first one containing:')
        f = d[list(d.keys())[0]]
        if type(f) == dict:
            present_dict(f,offset=' |'+offset+'       ')
        else:
            print(' |'+offset+'       ',type(f),', size:',np.size(f))
    else:
        for i,k in d.items():
            if type(k) == dict:
                print(offset,i,': dict, containing:')
                present_dict(k,offset=' |'+offset+'       ')
                print()
            elif (1 < np.size(k) < 5) and (type(k[0]) != dict):
                print(offset,i,':',k)
            elif np.size(k) == 1:
                print(offset,i,':',k)
            elif np.size(k) > 1 and type(k[0]) == dict:
                print(offset,i,': array of dicts, first one containing:')
                present_dict(k[0],offset=' |'+offset+'       ')
            else:
                print(offset,i,':',type(k),', size:',np.size(k))


def ctd_identify_water_masses(CTD, water_mass_def, stations=None):
    """
    Function to assign each ctd measurement tuple of T and S the corresponding water mass (AW, TAW, LW etc.)

    Parameters
    ----------
    CTD : dict
        CTD data. Is created by `read_CTD`
    stations : array_like, optional
        list of stations to select from `CTD`.
    water_mass_def : pandas DataFrame
        contains the water mass abbreviations, T and S limits and colorcodes

    Returns
    -------
    CTD : dict
        dict with the ctd data, each station has a new variable 'water_mass'
    """
    
    # if no stations are given, take all stations available
    if stations is None:
        stations = list(CTD.keys())
    else:
        CTD = {key:CTD[key] for key in stations}
        
    for s in stations:
        CTD[s]["water_mass"] = np.ones_like(CTD[s]["T"]) * np.nan
        for index, row in water_mass_def.iterrows():
            if row["Abbr"] != "ArW":
                ind = np.all(np.array([CTD[s]["T"] > row['T_min'],
                                        CTD[s]["T"] <= row['T_max'],
                                        CTD[s]["S"] > row['S_psu_min'],
                                        CTD[s]["S"] <= row['S_psu_max']]), axis=0)
                CTD[s]["water_mass"][ind] = index
    
    return CTD





############################################################################
#READING FUNCTIONS
############################################################################

def read_ADCP_CODAS(filename):
    '''
    Reads ADCP data from a netCDF file processed by CODAS. To be used with the *short* file!
    Parameters:
    -------
    filename: str
        String with path to filename
    Returns
    -------
    ds : xarray dataset
        Dataset containing the adcp data. Current velocities are adjusted for the ship's motion.'
    '''
    
    with xr.open_dataset(filename) as f:
        ds = f[["u", "v", "lat", "lon", "depth", "amp", "pg", "heading", "uship", "vship"]].load()
        
    ds = ds.set_coords(("depth", "lon", "lat"))
    
    ds['speed_ship'] = xr.apply_ufunc(np.sqrt, ds['uship']**2. + ds['vship']**2.)
    ds["speed_ship"].attrs["name"] = "speed_ship"
    ds["speed_ship"].attrs["units"] = "m/s"
    ds["speed_ship"].attrs["long_name"] = "total ship speed"
    
    calc_crossvel = lambda u, v, angle_deg: v * np.sin(np.deg2rad(angle_deg)) - u * np.cos(np.deg2rad(angle_deg))
    ds['crossvel'] = xr.apply_ufunc(calc_crossvel, ds['u'], ds['v'], ds['heading'])
    ds["crossvel"].attrs["name"] = "crossvel"
    ds["crossvel"].attrs["units"] = "m/s"
    ds["crossvel"].attrs["long_name"] = "current component perpendicular to ship track"
    
    ds["u"].attrs["long_name"] = "Eastward current velocity [m/s]"
    ds["v"].attrs["long_name"] = "Northward current velocity [m/s]"
    ds["uship"].attrs["long_name"] = "Eastward ship speed [m/s]"
    ds["vship"].attrs["long_name"] = "Northward ship speed [m/s]"
    ds["pg"].attrs["long_name"] = "Percent good"
    ds["heading"].attrs["long_name"] = "Ship heading [°]"
    ds["speed_ship"].attrs["long_name"] = "Ship speed [m/s]"
    ds["crossvel"].attrs["long_name"] = "Current velocity perpendicular to the ship track [m/s]"
    
    return ds


def split_CODAS_resolution(ds):
    """
    Splits the full ADCP time series into seperate datasets containing only timesteps with the same depth resolution.

    Parameters
    ----------
    ds : xarray dataset
        Dataset containing the full (CODAS-processed) ADCP timeseries (the return from the function read_ADCP_CODAS)

    Returns
    -------
    list_of_ds : list
        List of xarray datasets with different depth resolutions

    """
    
    ds["depth_binsize"] = ds.depth.isel(depth_cell=slice(0,2)).diff(dim="depth_cell").squeeze("depth_cell", drop=True).drop("depth")
    
    depth_resolutions = sorted(list(ds.groupby("depth_binsize").groups.keys()))
    
    one_d_varis = ["heading", "uship", "vship", "speed_ship"]
    
    list_of_ds = []
    for d in depth_resolutions:
         ds_d = ds.where(ds.depth_binsize == d, np.nan)
         ds_d["depth"] = ds_d.depth.isel(time=0)
         ds_d = ds_d.swap_dims({"depth_cell": "depth"}).drop("depth_binsize")
         ds_dd = ds_d[one_d_varis]
         ds_d = ds_d.where(ds_d.depth.notnull(), drop=True)
         for vari in one_d_varis:
             ds_d[vari] = ds_dd[vari]
         ds_d = ds_d.transpose("depth", "time")
         list_of_ds.append(ds_d)
        
    return list_of_ds
    


def read_WinADCP(filename):
    '''
    Reads data from a .mat data file processed with WinADCP.
    Parameters:
    -------
    filename: str
        String with path to file
    Returns
    -------
    df : pandas dataframe
        a pandas dataframe with time as index and the individual variables as columns.
    '''


    data = myloadmat(filename)
        
    depth = np.round(data['RDIBin1Mid'] + (data["SerBins"]-1)*data["RDIBinSize"])
        
    time = [pd.Timestamp(year=2000+y, month=m, day=d, hour=H, minute=M, second=s) for y,m,d,H,M,s in 
            zip(data["SerYear"], data["SerMon"], data["SerDay"], data["SerHour"], data["SerMin"], data["SerSec"])]
    
    glattributes = {name: data[name] for name in ['RDIFileName', 'RDISystem', 'RDIBinSize', 'RDIPingsPerEns', 'RDISecPerPing']}
        
    ds = xr.Dataset(data_vars=dict(temperature=(["time"], data["AnT100thDeg"]/100., {'units':'degC', "name": "temperature", "long_name": "sea water temperature"}),
                                  u_raw=(["time", "depth"], data["SerEmmpersec"]/1000., {'units':'m/s', "name": "u_raw", "long_name": "zonal velocity component (rel. to ship)"}),
                                  v_raw=(["time", "depth"], data["SerNmmpersec"]/1000., {'units':'m/s', "name": "v_raw", "long_name": "meridional velocity component (rel. to ship)"}),
                                  pg=(["time", "depth"], data['SerPG4']/100., {'units':'percent', "name": "pg", "long_name": "percent good"}),
                                  uship=(["time"], data["AnNVEmmpersec"]/1000., {'units':'m/s', "name": "uship", "long_name": "ship zonal velocity component"}),
                                  vship=(["time"], data["AnNVNmmpersec"]/1000., {'units':'m/s', "name": "vship", "long_name": "ship meridional velocity component"})),
                   coords=dict(time=time,depth=depth),
                               #lat_start=(["time"], data["AnFLatDeg"]),
                               #lat_end=(["time"], data["AnLLatDeg"]),
                               #lon_start=(["time"], data["AnFLonDeg"]),
                               #lon_end=(["time"], data["AnLLonDeg"])),
                   attrs=glattributes)
    
    
    ds["lat"] = xr.DataArray(0.5*(data["AnFLatDeg"]+data["AnLLatDeg"]),dims=['time'],coords={'time':ds.time})
    ds["lon"] = xr.DataArray(0.5*(data["AnFLonDeg"]+data["AnLLonDeg"]),dims=['time'],coords={'time':ds.time})
    
    
    ds = ds.set_coords(("lat", "lon"))
    
    ds["u"] = ds["u_raw"] + ds["uship"]
    ds["u"].attrs["name"] = "u"
    ds["u"].attrs["units"] = "m/s"
    ds["u"].attrs["long_name"] = "zonal velocity component"
    
    ds["v"] = ds["v_raw"] + ds["vship"]
    ds["v"].attrs["name"] = "v"
    ds["v"].attrs["units"] = "m/s"
    ds["v"].attrs["long_name"] = "meridional velocity component"
    
    calc_heading = lambda u, v: (((np.rad2deg(np.arctan2(-u,-v)) + 360.) % 360.) + 180.) % 360.
    ds["heading"] = xr.apply_ufunc(calc_heading, ds['uship'], ds['vship'])
    ds["heading"].attrs["name"] = "heading"
    ds["heading"].attrs["units"] = "deg"
    ds["heading"].attrs["long_name"] = "ship heading"
    
    
    calc_crossvel = lambda u, v, angle_deg: v * np.sin(np.deg2rad(angle_deg)) - u * np.cos(np.deg2rad(angle_deg))
    ds['crossvel'] = xr.apply_ufunc(calc_crossvel, ds['u'], ds['v'], ds['heading'])
    ds["crossvel"].attrs["name"] = "crossvel"
    ds["crossvel"].attrs["units"] = "m/s"
    ds["crossvel"].attrs["long_name"] = "current component perpendicular to ship track"

    ds['speed_ship'] = xr.apply_ufunc(np.sqrt, ds['uship']**2. + ds['vship']**2.)
    ds["speed_ship"].attrs["name"] = "speed_ship"
    ds["speed_ship"].attrs["units"] = "m/s"
    ds["speed_ship"].attrs["long_name"] = "total ship speed"

    ds = ds.transpose("depth", "time")
    
    ds["u"].attrs["long_name"] = "Eastward current velocity [m/s]"
    ds["v"].attrs["long_name"] = "Northward current velocity [m/s]"
    ds["u_raw"].attrs["long_name"] = "Raw eastward current velocity [m/s]"
    ds["v_raw"].attrs["long_name"] = "Raw northward current velocity [m/s]"
    ds["uship"].attrs["long_name"] = "Eastward ship speed [m/s]"
    ds["vship"].attrs["long_name"] = "Northward ship speed [m/s]"
    ds["pg"].attrs["long_name"] = "Percent good"
    ds["heading"].attrs["long_name"] = "Ship heading [°]"
    ds["speed_ship"].attrs["long_name"] = "Ship speed [m/s]"
    ds["crossvel"].attrs["long_name"] = "Current velocity perpendicular to the ship track [m/s]"

    return ds


def read_LADCP(filename, station_dict,switch_xdim='station'):
    """
    Function to read the data from the LADCP-mat-files.

    Parameters
    ----------
    filename : str
        String with path to the datafile
    station_dict : dict
        dictionary connecting the ship station numbers to the UNIS station numbers. Can be generated from the CTD-dict with 'stations_dict = {CTD[i]["st"]: i for i in CTD.keys()}'.
        Be aware that if a UNIS station has been measured several times and not named differently (e.g. 987_1, 987_2 etc.), only the last measurement will be present in the CTD dict and the previous station numbers are missing.
        In this case, it is easiest to manually rename the UNIS station numbers in the respective CTD data files (.cnv)
    switch_xdim : str	
	Keyword to switch between time and station as x dimension for the returned data set. Default is station (UNIS station number).
    Returns
    -------
    ds : xarray dataset containing the l-adcp data.

    """

    adcp = myloadmat(filename)
    
    list_of_das = []
    for vari in ["U", "V", "U_detide", "V_detide"]:
        list_of_dfs = []
        for st in range(len(adcp["stnr"])):
            max_depth = np.floor((np.nanmax(adcp["Z"][:,st])))
            grid = np.arange(max_depth)
            df = pd.DataFrame(adcp[vari][:,st], index=adcp["Z"][:,st], columns=[station_dict[adcp["stnr"][st]]])
            df = df.drop_duplicates().dropna()
            df_resampled = df.reindex(df.index.union(grid)).interpolate('values').loc[grid]
            list_of_dfs.append(df_resampled)
            
        df_total = pd.concat(list_of_dfs, axis=1)
            
        list_of_das.append(xr.DataArray(data=df_total, dims=["depth", "station"], coords={"station": df_total.columns, "depth": df_total.index}, name=vari))
        
    ds = xr.merge(list_of_das)
    ds["ship_station"] = xr.DataArray(adcp["stnr"],dims= ["station"], coords = {"station": ds.station})
    
    
    aux_variables = {"LAT": "lat", "LON": "lon", "ED": "Echodepth"}
    
    for vari_old, vari_new in aux_variables.items():
        ds[vari_new] = xr.DataArray(adcp[vari_old], dims=["station"], coords={"station": ds.station})
        
    ds["time"] = xr.DataArray(pd.to_datetime(np.asarray(adcp["DT"])-719529., unit='D').round('1s'),dims= ["station"], coords = {"station": ds.station})
    ds = ds.set_coords(['lat','lon','ship_station', "Echodepth"])
    if switch_xdim == 'time':
        ds = ds.swap_dims({'station':'time'})
        
    ds = ds.rename({'U':'u','V':'v','U_detide':'u_detide','V_detide':'v_detide', "Echodepth": "bottom_depth"})
    
    ds["u"] = ds["u"] / 100.
    ds["v"] = ds["v"] / 100.
    ds["u_detide"] = ds["u_detide"] / 100.
    ds["v_detide"] = ds["v_detide"] / 100.
    
    ds["u"].attrs["long_name"] = "Eastward current velocity [m/s]"
    ds["v"].attrs["long_name"] = "Northward current velocity [m/s]"
    ds["u_detide"].attrs["long_name"] = "Detided eastward current velocity [m/s]"
    ds["v_detide"].attrs["long_name"] = "Detided northward current velocity [m/s]"
   
    return ds



def read_CTD(inpath,cruise_name='cruise',outpath=None,stations=None, salt_corr=(1.,0.),oxy_corr = (1.,0.), use_system_time=False):
    '''
    This function reads in the CTD data from cnv files in `inpath`
    for the stations `stations` and returns a list of dicts containing
    the data. Conductivity correction (if any) can be specified in `corr`
    Parameters
    ----------
    inpath : str
        Either the path to a folder where the cnv files are stored, or the path
        to a .npy file with the data. In the latter case, NO correction can be
        applied.
    cruise_name : str, optional
        name of the cruise. The default is 'cruise'
    outpath : str, optional
        path where to store the output. The default is None.
    stations : array_like, optional
        list of stations to read in (optional). If not given,
        the function will read all stations in `inpath`. The default is None.
    salt_corr : tuple, optional
        tuple with 2 values containing (slope,intersect) of
                      linear correction model. The default is (1.,0.).
    oxy_corr : tuple, optional
        tuple with 2 values containing (slope,intersect) of
                      linear correction model. The default is (1.,0.).
    use_system_time : bool, optional
        Switch to use the system upload time stamp instead of the NMEA one. By default, the NMEA time stamp is used.
        
    Returns
    -------
    CTD_dict : dict
        a dict of dicts containing the data for
                    all the relevant station data.
    '''
    # first, check if the infile is a npy file. In that case, just read the
    # npy file and return the dict. No correction can be applied.
    if inpath[-4::] == '.npy':
        CTD_dict = np.load(inpath,allow_pickle=True).item()
        if stations is not None:
            try:
                CTD_dict = {k:CTD_dict[k] for k in stations}
            except:
                assert False, 'Some of the stations you provide don\'t exist'\
                    'in the data!'
        return CTD_dict

    # If a folder is given, read single cnv files.
    # create a dict that converts the variable names in the cnv files to
    # the variable names used by us:
    var_names = {'DEPTH': "D", 'PRES': "P", 'prdM': "P", 'TEMP': "T", 'tv290C': "T", 'CNDC': "C", 'c0mS/cm': "C", 'PSAL': "S", 'sigma_t': 'SIGTH', 'soundspeed': "Cs", 'sbeox0PS': "OXsat", 'seaTurbMtr': "TURB", 'par/sat/log': "PAR", 'oxygen_ml_L': "OX", 'potemperature': "Tpot", 'oxsolML/L': "OXsol"}

    # get all CTD station files in inpath
    files = glob.glob(inpath+'*.cnv')
    #If stations are provided, select the ones that exist
    if stations is not None:
        use_files = [i for i in files for j in stations if str(j) in i]
        assert len(use_files) > 0, 'None of your provided stations exists!'
        if len(use_files) < len(stations):
            print('Warning: Some stations you provided do not exist!')
        files = use_files

    files = sorted(files)

    # Read in the data, file by file
    CTD_dict = {}
    for file in files:
        # get all the fields, construct a dict with the fields
        profile = fCNV(file)
        p = {var_names[name]:profile[name]
            for name in profile.keys() if name in var_names}
        

        # get the interesting header fields and append it to the dict
        p.update(profile.attrs)
        
        # get the UNIS station number
        found_unis_station = False
        with open(file, encoding = "ISO-8859-1") as f:
            while not found_unis_station:
                line = f.readline()
                if (("unis station" in line.lower()) or ("unis-station" in line.lower())):
                    found_unis_station = True
                    if ":" in line:
                        unis_station = ((line.split(":"))[-1]).strip()
                    else:
                        unis_station = ((line.split(" "))[-1]).strip()
        if not found_unis_station:
            unis_station = "unknown"
            
        # if NMEA time is wrong, replace with system upload time (needs to be manually switched on)
        if use_system_time:
            found_system_time = False
            with open(file, encoding = "ISO-8859-1") as f:
                while not found_system_time:
                    line = f.readline()
                    if "system upload time" in line.lower():
                        found_system_time = True
                        p['datetime'] = datetime.datetime.strptime(((line.split("="))[-1]).strip(), '%b %d %Y %H:%M:%S')
            if not found_system_time:
                p['datetime'] = datetime.datetime(1970,1,1,0,0,0)
        
        # if time is present: convert to dnum
        try:
            p['dnum'] = date2num(p['datetime'])
        except:
            pass
        # rename the most important ones to the same convention used in MATLAB,
        # add other important ones
                
        p['LAT'] = p.pop('LATITUDE')
        p['LON'] = p.pop('LONGITUDE')
        p['z'] = gsw.z_from_p(p['P'],p['LAT'])
        p['BottomDepth'] = np.round(np.nanmax(np.abs(p['z']))+8)
        if np.nanmin(p["C"]) > 10.:
            p["C"] /= 10.
        p['C'][p['C']<1] = np.nan
        p['T'][p['T']<-2] = np.nan
        p['S'] = salt_corr[0] * p['S'] + salt_corr[1] # apply correction
        p['S'][p['S']<20] = np.nan
        p['C'][p['S']<20] = np.nan
        p['SA'] = gsw.SA_from_SP(p['S'],p['P'],p['LON'],p['LAT'])
        p['CT'] = gsw.CT_from_t(p['SA'],p['T'],p['P'])
        p['SIGTH'] = gsw.sigma0(p['SA'],p['CT'])
        p['st'] = int(p['filename'].split('.')[0].split('_')[0][-4::])
        p["unis_st"] = unis_station
        if 'OX' in p:
            p['OX'] = oxy_corr[0] * p['OX'] + oxy_corr[1]
        CTD_dict[p['unis_st']]= p
        
        
    # if all keys are integers (original UNIS station numbers) --> change str keys to int
    all_keys_int = True
    for sta in CTD_dict.keys():
        try:
            int(sta)
        except ValueError:
            all_keys_int = False
            break
        
    if all_keys_int:
        CTD_dict = dict((k_int,v) for k_int, v in zip([int(i) for i in CTD_dict.keys()], CTD_dict.values()))
    

    # save data if outpath was given
    if outpath is not None:
        np.save(outpath+cruise_name+'_CTD',CTD_dict)

    return CTD_dict


def read_CTD_from_mat(matfile):
    '''
    Reads CTD data from matfile
    Parameters
    ----------
    matfile : str
        The full path to the .mat file. This should contain a struct with the
        name CTD. This is the common output style of the cruise matlab scripts.
    Returns
    -------
    CTD : dict
        The dictionary with the CTD Data.
    '''
    # read the raw data using scipy.io.loadmat
    raw_data = loadmat(matfile, squeeze_me=True, struct_as_record=False)['CTD']
    # convert to dictionary
    CTD = {}
    for record in raw_data:
        station = record.__dict__['st']
        CTD[station] = record.__dict__
        CTD[station].pop('_fieldnames',None)

        # correct dnum parameter, because MATLAB and PYTHON
        # datenumbers are different
        CTD[station]['dnum'] = datestr2num(CTD[station]['date'])

    if 'note' in CTD[next(iter(CTD))]:
        print('Note: This CTD data is already calibrated.')

    return CTD



def read_mini_CTD(file,corr=(1,0),lon=0,lat=60.,station_name = 'miniCTD'):
    '''
    Reads files generated by the processing software of the mini CTD instrument.
    Calculated absolute salinity, conservative temperature, depth

    Parameters
    ----------
    file : string
        File containing the data (.TOB).

    corr : tuple (2), optional
        Tuple containing correction values (a,b) of linear correction, where
        a is the slope and b is the intercept. Defaults to (1,0)
    lon : float, optional
        Longitude of profile. Defaults to 0.
    lat : float, optional
        Latitude of profile. Defaults to 60.
    station_name : str, optional
        Name of the mini CTD station. Defaults to 'miniCTD'.

    Returns
    -------
    a dictionary containing the data
    '''
    # map norwegian months to padded numbers
    d2n = {'januar':'01','februar':'02','mars':'03','april':'04','mai':'05',
           'juni':'06','juli':'07','august':'08','september':'09',
           'oktober':'10','november':'11','desember':'12'}

    # open file
    f = open(file,encoding="ISO-8859-1")
    lines = f.readlines(10000) # read first lines of file
    f.close()

    # read time string, prepare for datetime parsing
    time_str = lines[2].replace(':','.').split(' ')[1::]
    time_str[1] = d2n[time_str[1]]
    time_str[0] = time_str[0].zfill(3)

    header_line = lines[25].replace(';','').split(' ')
    while '' in header_line: header_line.remove('')
    while '\n' in header_line: header_line.remove('\n')

    if 'IntT' in header_line: # Check if instrument recorded time
        header_line[header_line.index('IntT')] = 'Time'
        header_line[header_line.index('IntD')] = 'Date'


    dd = pd.read_csv(file,encoding="ISO-8859-1",skiprows=28,
                     engine='python',delim_whitespace=True,
                     skip_blank_lines=False,names =list(header_line),
                     na_values='########')

    p = {key:dd[key].to_numpy()[1::] for key in dd.columns}
    p['z'] = gsw.z_from_p(p['Press'],lat)
    p['Cond'][p['Cond']<0] = np.nan
    p['Cond'] = corr[0]*p['Cond'] + corr[1] # apply correction
    p['Temp'][p['Temp']<-2.5] = np.nan
    p['Prac_Sal'] = gsw.SP_from_C(p['Cond'],p['Temp'],p['Press'])
    p['Prac_Sal'][p['Prac_Sal']<0] = np.nan
    p['Cond'][p['Prac_Sal']<0] = np.nan
    p['SA'] = gsw.SA_from_SP(p['Prac_Sal'],p['Press'],lon,lat)
    p['CT'] = gsw.CT_from_t(p['SA'],p['Temp'],p['Press'])
    p['SIGTH'] = gsw.sigma0(p['SA'],p['CT'])
    p['st'] = station_name
    p['file_time'] = pd.to_datetime(''.join(time_str)[0:-1],format='%d.%m%Y%H.%M.%S')
    if 'Date' in p:
        p['datetime'] = [pd.to_datetime(a+' '+b,format='%d.%m.%Y %H:%M:%S')
                                    for (a,b) in zip(p['Date'],p['Time'])]
        del p['Date'], p['Time']

    return p



def read_MSS(files,excel_file=None):
    '''
    Parameters
    ----------
    file : str
        Full path to the .mat file.
    Returns
    -------
    None.
    '''
    # first, handle the excel file
    if excel_file is not None:
        exc = pd.read_excel(excel_file)
        exc.columns = np.arange(len(exc.columns))
        st = exc[[a for a in exc.columns
                    if 'Station name' in exc[a].to_numpy()][0]]
        lat_deg = exc[[a for a in exc.columns
                    if 'Latitude/ N' in exc[a].to_numpy()][0]]
        lat_min = exc[[a for a in exc.columns
                    if 'Latitude/ N' in exc[a].to_numpy()][0]+1]
        lon_deg = exc[[a for a in exc.columns
                    if 'Longitude/ E' in exc[a].to_numpy()][0]]
        lon_min = exc[[a for a in exc.columns
                    if 'Longitude/ E' in exc[a].to_numpy()][0]+1]

    # Determine if folder or file is given
    if  '.mat' in files:
        files = [files]
    else:
        files = glob.glob(files+'*.mat')

    out_data = {'CTD':{},'MIX':{},'DATA':{}}
    for file in files:
        st_name = int(file.split('.mat')[0][-4:])
        raw_data = myloadmat(file)
        data = {k:raw_data[k] for k in ['CTD','MIX','DATA']}

        for name in ['CTD','MIX']:
            for var in ['LON','LAT','fname','date']:
                data[name][var] = raw_data['STA'][var]

            if excel_file is not None:
                try:
                    index = np.where(st == st_name)[0][0]
                    data[name]['LON'] = lon_deg[index] + float(lon_min[index])/60
                    data[name]['LAT'] = lat_deg[index] + float(lat_min[index])/60
                except:
                    pass
            try:
                data[name]['z'] = gsw.z_from_p(data[name]['P'],data[name]['LAT'])
            except: # just use 60N as lat if lat is not provided
                data[name]['z'] = gsw.z_from_p(data[name]['P'],60)

            # Something weird in the data...
            data[name]['z'][np.isnan(data[name]['SIGTH'])] = np.nan
            data[name]['BottomDepth'] = np.nanmax(-data[name]['z'])
            data[name]['datetime'] = pd.to_datetime(data[name]['date']
                                                    ,format='%d-%b-%Y %H:%M:%S')




        for name in ['CTD','MIX','DATA']:
            out_data[name][st_name] = data[name]

    return out_data['CTD'],out_data['MIX'],out_data['DATA']



def read_mooring_from_mat(matfile):
    '''
    Read mooring data prepared in a .mat file.
    Parameters
    ----------
    matfile : str
        Full path to the .mat file.
    Returns
    -------
    raw_data : dict
        Dictionary with the mooring data.
    '''
    # read raw data using scipy.io.loadmat, plus more complicated changes
    raw_data = myloadmat(matfile)
    variable_name = list(raw_data.keys())[-1]
    raw_data = raw_data[variable_name]

    for key in raw_data.keys():
        if key[:4] == "date":
            raw_data[key] = mat2py_time(np.asarray(raw_data[key]))

    return raw_data



def read_mooring(file):
    '''
    Read mooring data prepared in a either a .npy or .mat file.
    Parameters
    ----------
    file : str
        Full path to the .mat or .npy file.
    Returns
    -------
    raw_data : dict
        Dictionary with the mooring data.
    '''
    ext = file.split('.')[-1]
    if ext == 'mat':
        raw_data = read_mooring_from_mat(file)
    elif ext == 'npy':
        raw_data = np.load(file,allow_pickle=True).item()

    return raw_data




def read_Seaguard(filename, header_len=4):
    '''
    Reads data from one data file from a Seaguard.

    Parameters:
    -------
    filename: str
        String with path to file
    header_len: int
        Number of header lines that have to be skipped.
    Returns
    -------
    df : pandas dataframe
        a pandas dataframe with time as index and the individual variables as columns.
    '''
    df = pd.read_csv(filename, sep="\t", header=header_len, parse_dates=["Time tag (Gmt)"], dayfirst=True)
    df.rename({"Time tag (Gmt)": "TIMESTAMP", 'East(cm/s)': "U", 'North(cm/s)': "V", 'Temperature(DegC)': "T", 'Pressure(kPa)': "P"}, axis=1, inplace=True)
    df = df.set_index("TIMESTAMP")
    df.sort_index(axis=0, inplace=True)
    
    return df


def read_Minilog(filename):
    '''
    Reads data from one data file from a Minilog temperature sensor.

    Parameters:
    -------
    filename: str
        String with path to file
    Returns
    -------
    df : pandas dataframe
        a pandas dataframe with time as index and temperature as column.
    '''

    with open(filename, "r", encoding = "ISO-8859-1", ) as f:
        for i in range(7):
            f.readline()
        col_names = f.readline().strip().split(",")
    
    if (("date" in col_names[0].lower()) and ("time" in  col_names[0].lower())):
        df = pd.read_csv(filename, sep=",", skiprows=7, parse_dates=[col_names[0]], encoding = "ISO-8859-1")
        df.rename({f"{col_names[0]}": "TIMESTAMP"}, axis=1, inplace=True)
    else:
        df = pd.read_csv(filename, sep=",", skiprows=7, parse_dates=[[col_names[0], col_names[1]]], encoding = "ISO-8859-1")
        df.rename({f"{col_names[0]}_{col_names[1]}": "TIMESTAMP"}, axis=1, inplace=True)
    df = df.set_index("TIMESTAMP")
    df.sort_index(axis=0, inplace=True)
    
    return df
    

def read_SBE37(filename):
    '''
    Reads data from one data file from a SB37 Microcat sensor.

    Parameters:
    -------
    filename: str
        String with path to file
    Returns
    -------
    df : pandas dataframe
        a pandas dataframe with time as index and the individual variables as columns.
    '''
    
    var_names = {'cond0S/m': "C", 'sigma-�00': "SIGTH", 'prdM': "P", 'potemperature': "Tpot", 'tv290C': "T", 'timeS': "Time", 'PSAL': "S"}
    
    data = fCNV(filename)
    
    d = {var_names[name]:data[name]
        for name in data.keys() if name in var_names}
    
    d.update(data.attrs)
    
    d["TIMESTAMP"] = pd.to_datetime(d["Time"], unit='s', origin=pd.Timestamp(d["start_time"].split("[")[0].strip()))

    df = pd.DataFrame(0., index=d["TIMESTAMP"], columns=list(set([field for field in d if ((np.size(d[field]) > 1) and (field not in ["Time", "TIMESTAMP"]))])))
    for k in df.columns:
        df[k] = d[k]
    df.sort_index(axis=0, inplace=True)
    
    return df


def read_RBR(filename):
    '''
    Reads data from a .rsk data file from a RBR logger (concerto, solo, ...).

    Parameters:
    -------
    filename: str
        String with path to file
    Returns
    -------
    df : pandas dataframe
        a pandas dataframe with time as index and the individual variables as columns.
    '''
    
    with RSK(filename) as rsk:
        rsk.readdata()
        rsk.deriveseapressure()
        variables = list(rsk.channelNames)
        time = pd.to_datetime(rsk.data["timestamp"])
        
        if "conductivity" in variables:
            rsk.derivesalinity()
            rsk.derivesigma()
            # variables.append("salinity")
            # variables.append("density")
        variables = list(rsk.channelNames)
        
        data = rsk.data[variables]
        
        df = pd.DataFrame(data, index=time, columns=variables)
        
        df.rename({"condictivity": "C", "temperature": "T", "salinity": "S", "pressure": "P", "sea_pressure": "Ps", "density_anomaly": "SIGTH"}, axis=1, inplace=True)
        df.sort_index(axis=0, inplace=True)
        
    return df


def read_Thermosalinograph(filename, use_system_time):
    '''
    Reads data from one data file from the Helmer Hanssen thermosalinograph.

    Parameters:
    -------
    filename: str
        String with path to file(s)
        If several files shall be read, specify a string including UNIX-style wildcards
    use_system_time : bool, optional
        Switch to use the system upload time stamp instead of the NMEA one. By default, the NMEA time stamp is used.
    Returns
    -------
    df : pandas dataframe
        a pandas dataframe with time as index and the individual variables as columns.
    '''
    
    var_names = {'CNDC': "C", 'sigma-�00': "SIGTH", 'prM': "P", 'potemperature': "Tpot", 'TEMP': "T", 'timeS': "Time", 'PSAL': "S", "LATITUDE": "LAT", "LONGITUDE": "LON"}
    
    if os.path.isfile(filename):
        list_of_files = [filename]
    else:
        list_of_files = sorted(glob.glob(filename))
    
    
    list_of_df = []
    
    for file in list_of_files:
        data = fCNV(file)
        
        d = {var_names[name]:data[name]
            for name in data.keys() if name in var_names}
        
        d.update(data.attrs)
        
        if use_system_time:
            found_system_time = False
            with open(file, encoding = "ISO-8859-1") as f:
                while not found_system_time:
                    line = f.readline()
                    if "system upload time" in line.lower():
                        found_system_time = True
                        d['start_time'] = line.split("=")[-1].strip()
        else:
            d["start_time"] = d["start_time"].split("[")[0].strip()
            
        d["TIMESTAMP"] = pd.to_datetime(d["Time"], unit='s', origin=pd.Timestamp(d["start_time"]))
    
        df = pd.DataFrame(0., index=d["TIMESTAMP"], columns=list(set([field for field in d if ((np.size(d[field]) > 1) and (field not in ["Time", "TIMESTAMP"]))])))
        for k in df.columns:
            df[k] = d[k]
        df.sort_index(axis=0, inplace=True)
        
        list_of_df.append(df)
        
    df_total = pd.concat(list_of_df)
    df_total.sort_index(axis=0, inplace=True)
    
    return df_total




############################################################################
# TIDE FUNCTIONS
############################################################################

def download_tidal_model(model="Arc2kmTM", outpath=pathlib.Path.cwd()):
    """
    Function to download a tidal model later used to calculate e.g. tidal currents at a certain location with the pyTMD package. This only needs to be done once.

    Parameters
    ----------
    model : str, optional
        String specifying the tidal model to download. Valid options are 'AODTM-5', 'AOTIM-5', 'AOTIM-5-2018', 'Arc2kmTM' and 'Gr1kmTM'. The default is "Arc2kmTM".
    outpath : TYPE, optional
        Path where a new folder with the tidal data will be created. The default is the current directory.

    Returns
    -------
    None.

    """

    if pyTMD.utilities.check_connection('https://arcticdata.io'):
        print("starting download...")
        
        # digital object identifier (doi) for each Arctic tide model
        DOI = {}
        DOI['AODTM-5'] = '10.18739/A2901ZG3N'
        DOI['AOTIM-5'] = '10.18739/A2S17SS80'
        DOI['AOTIM-5-2018'] = '10.18739/A21R6N14K'
        DOI['Arc2kmTM'] = '10.18739/A2D21RK6K'
        DOI['Gr1kmTM'] = '10.18739/A2B853K18'
        # local subdirectory for each Arctic tide model
        LOCAL = {}
        LOCAL['AODTM-5'] = 'aodtm5_tmd'
        LOCAL['AOTIM-5'] = 'aotim5_tmd'
        LOCAL['AOTIM-5-2018'] = 'Arc5km2018'
        LOCAL['Arc2kmTM'] = 'Arc2kmTM'
        LOCAL['Gr1kmTM'] = 'Gr1kmTM'
    
        # recursively create directories if non-existent
        DIRECTORY = pathlib.Path(outpath).expanduser().absolute()
        local_dir = DIRECTORY.joinpath(LOCAL[model])
        local_dir.mkdir(0o775, parents=True, exist_ok=True)
    
        # build host url for model
        resource_map_doi = f'resource_map_doi:{DOI[model]}'
        HOST = ['https://arcticdata.io','metacat','d1','mn','v2','packages',
            pyTMD.utilities.quote_plus(posixpath.join('application','bagit-097')),
            pyTMD.utilities.quote_plus(resource_map_doi)]
        # download zipfile from host
        zfile = zipfile.ZipFile(pyTMD.utilities.from_http(HOST, timeout=360))
        # find model files within zip file
        rx = re.compile(r'(grid|h[0]?|UV[0]?|Model|xy)_(.*?)',re.VERBOSE)
        members = [m for m in zfile.filelist if rx.search(m.filename)]
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
        

    return


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
    resolution = (data.index[1] - data.index[0]).seconds / 3600.
    delta = resolution*(1./24.)     #in days

    N = len(timeseries)

    # Periodogram
    freq, _ = signal.periodogram(timeseries, fs=1./delta, return_onesided=False)

    # Multitapering
    Sk_complex, weights, eigenvalues = spectrum.mtm.pmtm(timeseries, NW=bandwidth, NFFT=N, k=int(2*bandwidth - 1), method='adapt', show=False)
    Sk = np.abs(Sk_complex)**2.
    Sk = Sk.T
    multitap = np.mean(Sk*weights, axis=1) * delta

    return pd.Series(multitap[freq>=0.], index=freq[freq>=0.])


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

    time_seconds = np.array([(t-data.index[0]).total_seconds() for t in data.index])

    tide = uptide.Tides(constituents)
    tide.set_initial_time(pd.Timestamp(data.index[0]).to_pydatetime())
    amp,pha = uptide.harmonic_analysis(tide, data.values, time_seconds)

    detid = data - pd.Series(tide.from_amplitude_phase(amp, pha, time_seconds), index=data.index)

    
    if "Z0" in constituents:
        amp = list(amp)
        pha = list(pha)
        i = constituents.index("Z0")
        del amp[i]
        del pha[i]
        constituents.remove("Z0")
        amp = np.asarray(amp)
        pha = np.asarray(pha),
    tide = uptide.Tides(constituents)
    tide.set_initial_time(pd.Timestamp(data.index[0]).to_pydatetime())

    tidal_ts = []
    for a, p in zip(amp,pha):
        tidal_ts.append(pd.Series(tide.from_amplitude_phase([a], [p], time_seconds), index=data.index))

    return  amp, pha, detid, tidal_ts




############################################################################
# PLOTTING FUNCTIONS
############################################################################

def contour_section(X,Y,Z,Z2=None,ax=None,station_pos=None,cmap='jet',Z2_contours=None,
                    clabel='',bottom_depth=None,clevels=20,station_text='',
                    interp_opt=1,tlocator=None, cbar=True):
    '''
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
    '''
    # open new figure and get current axes, if none is provided
    if ax is None:
        ax = plt.gca()

    # get the labels for the Z2 contours
    if Z2 is not None and Z2_contours is None:
        Z2_contours = np.concatenate([list(range(21,26)),np.arange(25.5,29,0.2)])
        Z2_contours = [i for i in Z2_contours
                        if np.nanmin(Z2) < i < np.nanmax(Z2)]

    # get the Y-axis limits
    y_limits = (0,np.nanmax(Y))
    if bottom_depth is not None:
        y_limits = (0,np.nanmax(bottom_depth))

    if interp_opt == 0: #only z-interpolation: use pcolormesh
        norm = None
        if type(clevels) == int:
            if tlocator == 'logarithmic':
                norm = matplotlib.colors.LogNorm(np.nanmin(Z),np.nanmax(Z))
            cmap = plt.cm.get_cmap(cmap,clevels)
        else:
            norm = matplotlib.colors.BoundaryNorm(clevels,
                                                  ncolors=len(clevels)-1,
                                                  clip=False)
            if tlocator == 'logarithmic':
                norm = matplotlib.colors.LogNorm(np.min(clevels),np.max(clevels))
            cmap = plt.cm.get_cmap(cmap,len(clevels))

        cT = ax.pcolormesh(X,Y,Z,cmap=cmap,shading='auto',norm=norm) # draw Z
        plt.xlim(np.nanmin(X),np.nanmax(X))
    else: # full interpolation: use contours
        locator = None
        if tlocator == 'logarithmic':
            locator = matplotlib.ticker.LogLocator()
        cT = ax.contourf(X,Y,Z,cmap=cmap,levels=clevels,extend='both',
                         locator=locator) # draw Z


    if Z2 is not None:
        cSIG = ax.contour(X,Y,Z2,levels = Z2_contours,
                           colors='k',linewidths=[1],alpha=0.6) # draw Z2
        clabels = plt.clabel(cSIG, cSIG.levels, fontsize=8,fmt = '%1.1f') # add contour labels
        [txt.set_bbox(dict(facecolor='white', edgecolor='none',
                           pad=0,alpha=0.6)) for txt in clabels]
    else:
        cSIG = None

    if cbar:
        plt.colorbar(cT,ax = ax,label=clabel,pad=0.01) # add colorbar
    
    ax.set_ylim(y_limits)
    ax.invert_yaxis()

    # add bathymetry
    if bottom_depth is not None:
        # make sure bottom_depth is an np.array
        bottom_depth = np.asarray(bottom_depth)

        ax.fill_between(station_pos,bottom_depth*0+y_limits[1]+10,bottom_depth,
                     zorder=999,color='gray')

    ax.xaxis.set_ticks_position('both')
    ax.yaxis.set_ticks_position('both')

    # add station ticks
    if station_pos is not None:
        for i,pos in enumerate(station_pos):
            ax.text(pos,0,'v',ha='center',fontweight='bold')
            if len(station_text) == len(station_pos):
                ax.annotate(str(station_text[i]),(pos,0),xytext=(0,10),
                        textcoords='offset points',ha='center')

    return ax, cT, cSIG



def plot_CTD_section(CTD,stations,section_name = '',clevels_T=20,clevels_S=20,
                     x_type='distance',interp_opt = 1,bottom=False,z_fine=False):
    '''
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
    '''
    # Check if the function has data to work with
    assert type(CTD) in [dict,str], 'Parameter *CTD*: You must provide either\n'\
            ' a) a data dict or \n b) a npy file string with the data !'

    # read in the data (only needed if no CTD-dict, but a file was given)
    if type(CTD) is str:
        print('reading file...')
        CTD = np.load(CTD,allow_pickle=True).item()

    # Check if all stations given are found in the data
    assert min([np.isin(st,list(CTD.keys())) for st in stations]), 'Not all '\
            'of the provided stations were found in the CTD data! \n'\
            'The following stations were not found in the data: '\
            +''.join([str(st)+' ' for st in stations if ~np.isin(st,list(CTD.keys()))])
    # Check if x_type is either distance or time
    assert x_type in ['distance','time'], 'x_type must be eigher distance or '\
            'time!'



    # select only the given stations in the data
    CTD = {key:CTD[key] for key in stations}

    # extract Bottom Depth
    if type(bottom) == bool:
        BDEPTH = np.asarray([d['BottomDepth'] for d in CTD.values()])
    else:
        BDEPTH = bottom

    # put the fields (the vector data) on a regular, common pressure and X grid
    # by interpolating.
    fCTD,Z,X,station_locs = CTD_to_grid(CTD,x_type=x_type,
                                        interp_opt=interp_opt,z_fine=z_fine)

    # plot the figure
    fig,[axT,axS] = plt.subplots(2,1,figsize=(8,9), sharex=True)

    # Temperature
    _,Ct_T,C_T = contour_section(X,Z,fCTD['T'],fCTD['SIGTH'],ax = axT,
                          station_pos=station_locs,cmap=cmocean.cm.thermal,
                          clabel='Temperature [˚C]',bottom_depth=BDEPTH,clevels=clevels_T,
                          station_text=stations,interp_opt=interp_opt)
    # Salinity
    _,Ct_S,C_S = contour_section(X,Z,fCTD['S'],fCTD['SIGTH'],ax=axS,
                          station_pos=station_locs,cmap=cmocean.cm.haline,
                          clabel='Salinity [g kg$^{-1}$]',bottom_depth=BDEPTH,clevels=clevels_S,
                          interp_opt=interp_opt)
    # Add x and y labels
    axT.set_ylabel('Depth [m]')
    axS.set_ylabel('Depth [m]')
    if x_type == 'distance':
        axS.set_xlabel('Distance [km]')
    else:
        axS.set_xlabel('Time [h]')

    # add title
    fig.suptitle(section_name,fontweight='bold')

    # tight_layout
    fig.tight_layout(h_pad=0.1,rect=[0,0,1,0.95])

    return axT, axS, Ct_T, Ct_S, C_T, C_S



def plot_CTD_single_section(CTD,stations,section_name='',
                     x_type='distance',parameter='T',parameter_contourlines="SIGTH",clabel='Temperature [˚C]',
                     cmap=cmocean.cm.thermal,clevels=20,contourlevels=5,interp_opt = 1,bottom=False,
                     tlocator=None,z_fine=False, cbar=True):
    '''
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
    '''
    # Check if the function has data to work with
    assert type(CTD) in [dict,str], 'Parameter *CTD*: You must provide either\n'\
            ' a) a data dict or \n b) a npy file string with the data !'

    # read in the data (only needed if no CTD-dict, but a file was given)
    if type(CTD) is str:
        print('reading file...')
        CTD = np.load(CTD,allow_pickle=True).item()

    # Check if all stations given are found in the data
    assert min([np.isin(st,list(CTD.keys())) for st in stations]), 'Not all '\
            'of the provided stations were found in the CTD data! \n'\
            'The following stations were not found in the data: '\
            +''.join([str(st)+' ' for st in stations if ~np.isin(st,list(CTD.keys()))])
    # Check if x_type is either distance or time
    assert x_type in ['distance','time'], 'x_type must be eigher distance or '\
            'time!'

    # select only the given stations in the data
    CTD = {key:CTD[key] for key in stations}

    # extract Bottom Depth
    if type(bottom) == bool:
        BDEPTH = np.asarray([d['BottomDepth'] for d in CTD.values()])
    else:
        BDEPTH = bottom

    # put the fields (the vector data) on a regular, common pressure and X grid
    # by interpolating.
    fCTD,Z,X,station_locs = CTD_to_grid(CTD,x_type=x_type,
                                        interp_opt=interp_opt,z_fine=z_fine)
    

    # plot the figure
    fig,ax = plt.subplots(1,1,figsize=(8,5))

    # plot the cross section
    _,Ct,C = contour_section(X,Z,fCTD[parameter],fCTD[parameter_contourlines],ax = ax,
                          station_pos=station_locs,cmap=cmap,
                          clabel=clabel,bottom_depth=BDEPTH,
                          station_text=stations,clevels=clevels,Z2_contours=contourlevels,
                          interp_opt=interp_opt,tlocator=tlocator, cbar=cbar)
    
    
    # Add x and y labels
    ax.set_ylabel('Depth [m]')
    if x_type == 'distance':
        ax.set_xlabel('Distance [km]')
    else:
        ax.set_xlabel('Time [h]')

    # add title
    fig.suptitle(section_name,fontweight='bold')

    # tight_layout
    fig.tight_layout(h_pad=0.1,rect=[0,0,1,0.95])
    return ax, Ct, C


def plot_xarray_sections(list_das, list_cmaps, list_clevels=None, da_contours=None,
                  contourlevels=5, interp=False, switch_cbar=True, add_station_ticks=True):
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
        list_clevels = len(list_das)*[20]
    

    N_subplots = len(list_das)

    fig, axes = plt.subplots(N_subplots, 1, sharey=True, sharex=True, figsize=(12,N_subplots*4))
    if N_subplots == 1:
        axes = [axes]
    pics = []
    for i, da in enumerate(list_das):
        if interp:
            X = da.distance.to_numpy()
            Z = da.depth.to_numpy()
            # original grids
            X_orig,Z_orig = [f.ravel() for f in np.meshgrid(X,Z)]
            # new grids
            X_int = np.linspace(np.min(X),np.max(X),len(X)*20) # create fine X grid
            Z_int = Z[:]
            temp_array = da.to_numpy().ravel()
            mask = np.where(~np.isnan(temp_array)) # NaN mask
            # grid in X and Z
            data_to_plot = griddata((X_orig[mask],Z_orig[mask]), # old grid
                                 temp_array[mask], # data
                                 tuple(np.meshgrid(X_int,Z_int))) # new grid
            if da.name == "water_mass":
                data_to_plot = np.round(data_to_plot)
            pic = axes[i].contourf(X_int,Z_int,data_to_plot,cmap=list_cmaps[i],levels=list_clevels[i],extend='both')
            pics.append(pic)
            if switch_cbar:
                cbar = plt.colorbar(pic, ax=axes[i])
                cbar.ax.set_ylabel(da.attrs["long_name"])
        else:
            pic = da.plot.pcolormesh(x="distance", y="depth", ax=axes[i], shading="nearest", cmap=list_cmaps[i], levels=list_clevels[i], add_colorbar=switch_cbar, infer_intervals=False, robust=True, extend='both')
            pics.append(pic)

        if da_contours is not None:
            if interp:
                X = da_contours.distance.to_numpy()
                Z = da_contours.depth.to_numpy()
                # original grids
                X_orig,Z_orig = [f.ravel() for f in np.meshgrid(X,Z)]
                # new grids
                X_int = np.linspace(np.min(X),np.max(X),len(X)*20) # create fine X grid
                Z_int = Z[:]
                temp_array = da_contours.to_numpy().ravel()
                mask = np.where(~np.isnan(temp_array)) # NaN mask
                # grid in X and Z
                data_to_plot = griddata((X_orig[mask],Z_orig[mask]), # old grid
                                     temp_array[mask], # data
                                     tuple(np.meshgrid(X_int,Z_int))) # new grid
                contourlines = axes[i].contour(X_int, Z_int, data_to_plot, levels=contourlevels, colors="k", linewidths=[1], alpha=0.6)
            else:
                contourlines = da_contours.plot.contour(x="distance", y="depth", ax=axes[i], levels=contourlevels, colors="k", linewidths=[1], alpha=0.6)
            clabels = plt.clabel(contourlines, contourlines.levels, fontsize=8, fmt = '%1.1f')
            [txt.set_bbox(dict(facecolor='white', edgecolor='none',
                               pad=0,alpha=0.6)) for txt in clabels]
    
        axes[i].set_xlabel("")
        axes[i].set_ylabel("Depth [m]")
                
    
    # extract bathymetry
    bottom = None
    found_bottom = False
    i = 0
    while ((found_bottom == False) & (i < len(list_das))):
        if "bottom_depth" in list_das[i].coords:
            bottom = list_das[i]["bottom_depth"].to_numpy()
            bottom_x = list_das[i]["distance"].to_numpy()
            found_bottom = True 
        i += 1
    if ((found_bottom == False) & (da_contours is not None)):
        if "station" in da_contours.coords:
            bottom = da_contours.coords["bottom_depth"].to_numpy()
            bottom_x = da_contours.coords["distance"].to_numpy()
            found_bottom = True 
                
    # get the axis limits
    if bottom is not None:
        y_limits = (0,np.nanmax(bottom))
    else:
        da_max_depths = [da.depth.max().values for da in list_das]
        y_limits = (0,np.nanmax(da_max_depths))
        
    da_max_distances = [da.distance.max().values for da in list_das]
    if da_contours is not None:
        da_max_distances = da_max_distances + [da_contours.distance.max().values]
    x_limits = (0,np.nanmin(da_max_distances))
            
    if bottom is not None:
        for a in axes:
            a.fill_between(bottom_x,bottom*0+y_limits[1]+10,bottom,
                      zorder=999,color='gray')
    else:
        print("No bottom data found!")
            
            
    # add station ticks
    if add_station_ticks:
        found_stations = False
        i == 0
        while ((found_stations == False) & (i < len(list_das))):
            if "station" in list_das[i].coords:
                stations = list_das[i]["station"].to_numpy()
                distances = list_das[i]["distance"].to_numpy()
                found_stations = True
            i += 1
        if ((found_stations == False) & (da_contours is not None)):
            if "station" in da_contours.coords:
                stations = da_contours["station"].to_numpy()
                distances = da_contours["distance"].to_numpy()
                found_stations = True
            
            
            
        if found_stations:
            for s,d in zip(stations, distances):
                for a in axes:
                    a.text(d,0,'v',ha='center',fontweight='bold')
                axes[0].annotate(str(s),(d,0),xytext=(0,10),
                            textcoords='offset points',ha='center')
        else:
            print("Station ticks not possible when only VM-ADCP data is provided!")

    for a in axes:
        a.set_xlim(x_limits)
        a.set_ylim(y_limits)
        a.invert_yaxis()
        a.yaxis.set_ticks_position('both')
        
        
    axes[-1].set_xlabel("Distance [km]")

    return fig, axes, pics



def plot_CTD_station(CTD,station,axes = None, add = False,linestyle='-'):
    '''
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
    '''
    # Check if the function has data to work with
    assert type(CTD) in [dict,str], 'Parameter *CTD*: You must provide either\n'\
            ' a) a data dict or \n b) a npy file string with the data !'

    # read in the data (only needed if no CTD-dict, but a file was given)
    if type(CTD) is str:
        print('reading file...')
        CTD = np.load(CTD,allow_pickle=True).item()

    # Check if all stations given are found in the data
    assert np.isin(station,list(CTD.keys())), 'The station was not found in '\
            'the CTD data! \n The following stations are in the data: '\
            +''.join([str(st) +' ' for st in CTD.keys()])

    # end of checks.

    # select station
    CTD = CTD[station]

    if axes == None:
        ax = plt.gca()
        ax2 = ax.twiny()
        ax.invert_yaxis()
    else:
        assert len(axes) == 2, 'You need to provide a list of two axes'
        ax = axes[0]
        ax2 = axes[1]


    # plot
    ax.plot(CTD['CT'],-CTD['z'],'r',linestyle=linestyle)
    ax.set_xlabel('Conservative temperature [˚C]',color='r')
    ax.set_ylabel('Depth [m]')
    ax.spines['bottom'].set_color('r')
    ax.tick_params(axis='x', colors='r')


    ax2.plot(CTD['SA'],-CTD['z'],'b',linestyle=linestyle)
    ax2.set_xlabel('Absolute salinity [g / kg]',color='b')
    ax2.tick_params(axis='x', colors='b')
    plt.tight_layout()

    return ax,ax2



def plot_CTD_map(CTD,stations=None,topography=None,extent=None,
                 depth_contours=[10,50,100,150,200,300,400,500,1000,2000,
                          3000,4000,5000],st_labels='',adjust_text=False):
    '''
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
    '''

    assert type(st_labels) in [str,list,tuple], 'st_labels must either be' \
        'a string, a tuple or a list.'
    # if no stations are provided, just plot all stations
    if stations is None:
        stations = CTD.keys()

    # select only stations
    CTD = {key:CTD[key] for key in stations}
    lat = [value['LAT'] for value in CTD.values()]
    lon = [value['LON'] for value in CTD.values()]
    std_lat,std_lon = np.std(lat),np.std(lon)
    lon_range = [min(lon)-std_lon,max(lon)+std_lon]
    lat_range = [min(lat)-std_lat,max(lat)+std_lat]

    ax = plt.axes(projection=ccrs.PlateCarree())
    if extent is None:
        extent = [lon_range[0],lon_range[1],lat_range[0],lat_range[1]]
    ax.set_extent(extent)

    if topography is not None:
        if type(topography) is str:
            ext = topography.split('.')[-1]
            if ext == 'mat':
                topo = loadmat(topography)
                topo_lat,topo_lon,topo_z = topo['lat'],topo['lon'],topo['D']
            elif ext == 'npy':
                topo = np.load(topography)
                topo_lat,topo_lon,topo_z = topo[0],topo[1],topo[2]
            elif ext == 'nc':
                topo = Dataset(topography)
                topo_lat,topo_lon,topo_z = topo.variables['lat'][:], \
                                           topo.variables['lon'][:], \
                                           topo.variables['z'][:]
                if len(topo_lon.shape) == 1:
                    topo_lon,topo_lat = np.meshgrid(topo_lon,topo_lat)
            else:
                assert False, 'Unknown topography file extension!'
        else: # assume topography is array with 3 columns (lat,lon,z)
            topo_lat,topo_lon,topo_z = topography[0],topography[1],topography[2]

        topo_z[topo_z < -1] = -1 # discard elevation above sea level

        BC = ax.contour(topo_lon,topo_lat,topo_z,colors='lightblue',
                   levels=depth_contours,linewidths=0.3,
                    transform=ccrs.PlateCarree())
        clabels = ax.clabel(BC, depth_contours,fontsize=4,fmt = '%i')
        print(clabels)
        if clabels is not None:
            for txt in clabels:
                txt.set_bbox(dict(facecolor='none', edgecolor='none',
                               pad=0,alpha=0.))
        ax.contour(topo_lon,topo_lat,topo_z,levels=[0],colors='k',linewidths=0.5)
        ax.contourf(topo_lon,topo_lat,topo_z,levels=[-1,1],
                    colors=['lightgray','white'])
    else: # if no topography is provided
        ax.add_feature(cartopy.feature.GSHHSFeature(scale='auto',
                                                    facecolor='lightgray',
                                                    linewidth=0.5))

    # add the points, and add labels
    if type(st_labels) == str:
        st_texts = [st_labels+str(s) for s in stations]
    else:
        st_texts = st_labels

    ax.plot(lon,lat,'xr',transform=ccrs.PlateCarree())
    texts = []
    for i,station in enumerate(stations):
        if extent[0]<lon[i]<extent[1] and extent[2]<lat[i]<extent[3]:
            texts.append(ax.text(lon[i],lat[i],st_texts[i],horizontalalignment='center',
                    verticalalignment='bottom'))

    gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True,
                  linewidth=1, color='gray', alpha=0.5, linestyle='--')
    gl.xformatter = LONGITUDE_FORMATTER
    gl.yformatter = LATITUDE_FORMATTER

    # make sure aspect ration of the axes is not too extreme
    ax.set_aspect('auto')
    if adjust_text:
        adj_txt(texts, expand_text=(1.2,1.6),
            arrowprops=dict(arrowstyle='-', color='black'), ax=ax)
    plt.gcf().canvas.draw()
    plt.tight_layout()



def plot_empty_map(extent,topography=None,
                 depth_contours=[10,50,100,150,200,300,400,500,1000,2000,
                          3000,4000,5000]):
    '''
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
    '''

    fig, ax = plt.subplots(1,1, subplot_kw={'projection': ccrs.PlateCarree()})
    ax.set_extent(extent)
    if topography is not None:
        if type(topography) is str:
            ext = topography.split('.')[-1]
            if ext == 'mat':
                topo = loadmat(topography)
                topo_lat,topo_lon,topo_z = topo['lat'],topo['lon'],topo['D']
            elif ext == 'npy':
                topo = np.load(topography)
                topo_lat,topo_lon,topo_z = topo[0],topo[1],topo[2]
            elif ext == 'nc':
                topo = Dataset(topography)
                topo_lat,topo_lon,topo_z = topo.variables['lat'][:], \
                                           topo.variables['lon'][:], \
                                           topo.variables['z'][:]
                if len(topo_lon.shape) == 1:
                    topo_lon,topo_lat = np.meshgrid(topo_lon,topo_lat)
            else:
                assert False, 'Unknown topography file extension!'
        else: # assume topography is array with 3 columns (lat,lon,z)
            topo_lat,topo_lon,topo_z = topography[0],topography[1],topography[2]

        topo_z[topo_z < -1] = -1 # discard elevation above sea level
        BC = ax.contour(topo_lon,topo_lat,topo_z,colors='lightblue',
                   levels=depth_contours,linewidths=0.3,
                    transform=ccrs.PlateCarree())
        clabels = ax.clabel(BC, depth_contours,fontsize=4,fmt = '%i')
        print(clabels)
        if clabels is not None:
            for txt in clabels:
                txt.set_bbox(dict(facecolor='none', edgecolor='none',
                               pad=0,alpha=0.))
        ax.contour(topo_lon,topo_lat,topo_z,levels=[0.1],colors='k',linewidths=0.5)
        ax.contourf(topo_lon,topo_lat,topo_z,levels=[-1,1],
                    colors=['lightgray','white'])
    else: # if no topography is provided
        ax.add_feature(cartopy.feature.GSHHSFeature(scale='auto',
                                                    facecolor='lightgray',
                                                    linewidth=0.5))

    gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True,
                  linewidth=1, color='gray', alpha=0.5, linestyle='--')
    gl.xformatter = LONGITUDE_FORMATTER
    gl.yformatter = LATITUDE_FORMATTER

    # make sure aspect ration of the axes is not too extreme
    ax.set_aspect('auto')
    plt.gcf().canvas.draw()
    plt.tight_layout()

    return fig, ax



def plot_CTD_ts(CTD,stations=None,pref = 0):
    '''
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
    '''
    # select only input stations
    if stations is not None:
        CTD = {key:CTD[key] for key in stations}

    max_S = max([np.nanmax(value['SA']) for value in CTD.values()]) + 0.1
    min_S = min([np.nanmin(value['SA']) for value in CTD.values()]) - 0.1

    max_T = max([np.nanmax(value['CT']) for value in CTD.values()]) + 0.5
    min_T = min([np.nanmin(value['CT']) for value in CTD.values()]) - 0.5


    create_empty_ts((min_T,max_T),(min_S,max_S),p_ref=pref)

    # Plot the data in the empty TS-diagram
    for station in CTD.values():
        plt.plot(station['SA'],station['CT'],linestyle='none',marker='.',
                 label=station['unis_st'])

    if len(CTD.keys()) > 1:
        plt.legend(ncol=2,framealpha=1,columnspacing=0.7,handletextpad=0.4)



def create_empty_ts(T_extent,S_extent,p_ref = 0):
    '''
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
    '''

    sigma_functions = [gsw.sigma0,gsw.sigma1,gsw.sigma2,gsw.sigma3,gsw.sigma4]
    T = np.linspace(T_extent[0],T_extent[1],100)
    S = np.linspace(S_extent[0],S_extent[1],100)

    T,S = np.meshgrid(T,S)

    SIGMA = sigma_functions[p_ref](S,T)

    cs = plt.contour(S,T,SIGMA,colors='k',linestyles='--')
    plt.clabel(cs,fmt = '%1.1f')

    plt.ylabel('Conservative Temperature [°C]')
    plt.xlabel('Absolute Salinity [g kg$^{-1}$]')
    plt.title('$\Theta$ - $S_A$ Diagram')
    if p_ref > 0:
        plt.title('Density: $\sigma_{'+str(p_ref)+'}$',loc='left',fontsize=10)



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

    fig, ax = plt.subplots(1,1, figsize=(10,5))
    data.plot(ax=ax, color="b", zorder=10)
    ax.grid()
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xticks((24.*3600.)/((2.*np.pi)/tidal_freqs))
    ax.set_xticklabels(constituents)
    ax.minorticks_off()
    ax.set_xlim(right=data.index[-1])
    ax.set_ylabel('power spectral density')

    return fig, ax



def plot_map_tidal_ellipses(amp_major, amp_minor, inclin, theta, constituents, lat_center=78.122, lon_center=14.26, map_extent=[11., 16., 78., 78.3], topography=None):
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


    phi = np.linspace(0, 2*np.pi, 1000)

    fig, ax_map = plot_empty_map(extent=map_extent, topography=topography)

    inset_size = .3
    
    x, y = ax_map.projection.transform_point(lon_center, lat_center, ccrs.PlateCarree())
    data2axes = (ax_map.transAxes + ax_map.transData.inverted()).inverted()
    xp, yp = data2axes.transform((x, y))
    ip = InsetPosition(ax_map, [xp - inset_size / 2, yp - inset_size / 2, inset_size, inset_size])
    ellipse_inset = fig.add_axes((0, 0, 1, 1))
    ellipse_inset.set_axes_locator(ip)
    ellipse_inset.axis("off")
    ellipse_inset.set_facecolor('none')
    ellipse_inset.tick_params(labelleft=False, labelbottom=False)
    ellipse_inset.grid(False)
    ellipse_inset.set_aspect(1.)


    for i, (a, b, t, g) in enumerate(zip(amp_major, amp_minor, inclin, theta)):
        E = np.array([a*np.cos(phi) , b*np.sin(phi)])
        R_rot = np.squeeze(np.array([[np.cos(t) , -np.sin(t)],[np.sin(t) , np.cos(t)]]))
        E_rot = np.zeros((2,E.shape[1]))
        for j in range(E.shape[1]):
            E_rot[:,j] = np.dot(R_rot,E[:,j])

        ellipse_inset.plot(E_rot[0,:] , E_rot[1,:], c=f"C{i}", label=constituents[i])

        ind = np.where(abs(t-g) == np.nanmin(abs(t-g)))[0][0]

        ellipse_inset.annotate("",
                    xy=(E_rot[0,ind], E_rot[1,ind]), xycoords='data',
                    xytext=(0., 0.), textcoords='data',
                    arrowprops=dict(arrowstyle="-|>", connectionstyle="arc3", color=f"C{i}"),
                    )
        
        ellipse_inset.legend(ncol=len(constituents), bbox_to_anchor=(1.65, 3.85),
                         loc='upper right')


    return fig, ax_map, ellipse_inset
