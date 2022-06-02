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

The functions were developed at the University Centre in Svalbard. They were
optimized for the file formats typically used in the UNIS courses.
"""

import pandas as pd
import dask.dataframe as ddf
import xarray as xr
import glob


############################################################################
#READING FUNCTIONS
############################################################################        
                
def read_MET_AWS(filename):
    ''' 
    Reads data from a csv file downloaded from seklima.met.no.
    The csv file should only include data from one station.
    
    Parameters:
    -------
    filename: str
        String with path to file
    Returns
    -------
    df : pandas dataframe
        a pandas dataframe with time as index and the individual variables as columns.
    '''
    
    with open(filename) as f:
        ncols = len(f.readline().split(';'))

    df = pd.read_csv(filename, dayfirst=True, sep=";", skipfooter=1, header=0, usecols=range(2,ncols+1), parse_dates=[0], decimal=",")
    
    try:
        df["Tid"] = df["Tid(norsk normaltid)"] - pd.Timedelta("1h")
        df.set_index("Tid", inplace=True)
        df.drop(["Tid(norsk normaltid)"], axis=1, inplace=True)
    except KeyError:
        df["Time"] = df["Time(norwegian mean time)"] - pd.Timedelta("1h")
        df.set_index("Time", inplace=True)
        df.drop(["Time(norwegian mean time)"], axis=1, inplace=True)
            
    return df



def read_Campbell_AWS(filename):
    ''' 
    Reads data from one or several data files from the Campbell AWS output files.
    Make sure to only specify files with the same temporal resolution.
    
    Parameters:
    -------
    filename: str
        String with path to file(s)
        If several files shall be read, specify a string including UNIX-style wildcards
    Returns
    -------
    df : pandas dataframe
        a pandas dataframe with time as index and the individual variables as columns.
    '''
    
    df = ddf.read_csv(filename, skiprows=[0,2,3,4], dayfirst=True, parse_dates=["TIMESTAMP"])
    df = df.compute()
    df.set_index("TIMESTAMP", inplace=True)
    df.drop(["ID"], axis=1, inplace=True)
    
    return df



def read_Campbell_radiation(filename):
    ''' 
    Reads data from one or several data files from the Campbell radiation output files.
    
    Parameters:
    -------
    filename: str
        String with path to file(s)
        If several files shall be read, specify a string including UNIX-style wildcards
    Returns
    -------
    df : pandas dataframe
        a pandas dataframe with time as index and the individual variables as columns.
    '''
    
    df = ddf.read_csv(filename, skiprows=[0,2,3], dayfirst=True, parse_dates=["TIMESTAMP"])
    df = df.compute()
    df.set_index("TIMESTAMP", inplace=True)
    
    return df



def read_Irgason_flux(filename):
    ''' 
    Reads data from a Irgason flux output file.
    
    Parameters:
    -------
    filename: str
        String with path to file
    Returns
    -------
    df : pandas dataframe
        a pandas dataframe with time as index and the individual variables as columns.
    '''
    
    df = pd.read_csv(filename, skiprows=[0,2,3], dayfirst=True, parse_dates=["TIMESTAMP"])
    df.set_index("TIMESTAMP", inplace=True)
    
    return df



def read_CSAT3_flux(filename):
    ''' 
    Reads data from a CSAT3 flux data file processed with TK3.
    
    Parameters:
    -------
    filename: str
        String with path to file
    Returns
    -------
    df : pandas dataframe
        a pandas dataframe with time as index and the individual variables as columns.
    '''

    sonic_header = pd.read_csv(filename, nrows=1)
    sonic_header = [key.strip() for key in sonic_header.columns]
    
    df = pd.read_csv(filename, names=sonic_header, header=0, dayfirst=True, parse_dates=["T_begin", "T_end", "T_mid"], na_values=-9999.9003906)
    df.set_index("T_mid", inplace=True)
    
    return df



def read_Tinytag(filename, sensor):
    ''' 
    Reads data from one or several data files from the Tinytag output files.
    
    Parameters:
    -------
    filename: str
        String with path to file(s)
        If several files shall be read, specify a string including UNIX-style wildcards
    sensor: str
        One of "TT", "TH" or "CEB"
    Returns
    -------
    df : pandas dataframe
        a pandas dataframe with time as index and the individual variables as columns.
    '''
    
    
    if sensor == "TT":
        df = ddf.read_csv(filename, delimiter="\t", skiprows=5, parse_dates=[1], names=["RECORD", "TIMESTAMP", "T_black", "T_white"])
    elif sensor == "TH":
        df = ddf.read_csv(filename, delimiter="\t", skiprows=5, parse_dates=[1], names=["RECORD", "TIMESTAMP", "T", "RH"])
    elif sensor == "CEB":
        df = ddf.read_csv(filename, delimiter="\t", skiprows=5, parse_dates=[1], names=["RECORD", "TIMESTAMP", "T"])
    else:
        print('Sensortype of Tinytag not known. Should be one of "TT", "TH" or "CEB".')
        df = None

    df = df.compute()
    df.set_index("TIMESTAMP", inplace=True)
    
    return df
            


def read_IWIN(filename):
    ''' 
    Reads data from one or several netCDF data files from IWIN stations.
    
    Parameters:
    -------
    filename: str
        String with path to file(s)
        If several files shall be read, specify a string including UNIX-style wildcards
    Returns
    -------
    ds : xarray dataset
        a xarray dataset representing the netCDF file(s)
    '''
    
    files = sorted(glob.glob(filename))
    
    if len(files) == 1:
        ds = xr.open_dataset(filename)
    elif len(files) > 1:
        ds = xr.open_mfdataset(files)
    else:
        print("Wrong data path.")
        ds = None

    return ds



def read_AROME(filename):
    ''' 
    Reads data from one or several netCDF data files from AROME-Arctic.
    
    Parameters:
    -------
    filename: str
        String with path to file(s)
        If several files shall be read, specify a string including UNIX-style wildcards
    Returns
    -------
    ds : xarray dataset
        a xarray dataset representing the netCDF file(s)
    '''
    
    files = sorted(glob.glob(filename))
    
    if len(files) == 1:
        ds = xr.open_dataset(filename)
    elif len(files) > 1:
        ds = xr.open_mfdataset(files)
    else:
        print("Wrong data path.")
        ds = None

    return ds





