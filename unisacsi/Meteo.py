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

import unisacsi
import pandas as pd
import dask.dataframe as ddf
import xarray as xr
import glob
import geopandas as gpd
import numpy as np
import cmocean as cmo
import matplotlib as mpl
from shapely.geometry import LineString
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import rioxarray as rxr
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER


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
        df = ddf.read_csv(filename, delimiter="\t", skiprows=5, parse_dates=[1], names=["RECORD", "TIMESTAMP", "T_black", "T_white"], encoding = "ISO-8859-1")
    elif sensor == "TH":
        df = ddf.read_csv(filename, delimiter="\t", skiprows=5, parse_dates=[1], names=["RECORD", "TIMESTAMP", "T", "RH"], encoding = "ISO-8859-1")
    elif sensor == "CEB":
        df = ddf.read_csv(filename, delimiter="\t", skiprows=5, parse_dates=[1], names=["RECORD", "TIMESTAMP", "T"], encoding = "ISO-8859-1")
    else:
        assert False, 'Sensortype of Tinytag not known. Should be one of "TT", "TH" or "CEB".'

    df = df.compute()
    df.set_index("TIMESTAMP", inplace=True)

    for key in list(df.columns):
        if key == "RECORD":
            pass
        else:
            data = [float(i.split(" ")[0]) for i in df[key]]
            unit = df[key].iloc[0].split(" ")[1]
            if unit == "??C":
                unit = "degC"
            new_key = f"{key}_{unit}"

            df[new_key] = data

            df.drop(key, axis=1, inplace=True)

    return df




def read_HOBO(filename):
    '''
    Reads data from one or several data files from the HOBO output files.

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

    df = ddf.read_csv(filename, delimiter=";", skiprows=1, parse_dates=["Date Time, GMT+00:00"], dayfirst=True, encoding = "ISO-8859-1")
    df = df.compute()
    df.rename({"Date Time, GMT+00:00": "TIMESTAMP"}, axis=1, inplace=True)
    df.set_index("TIMESTAMP", inplace=True)

    new_names = []
    for i in list(df.columns):
        old_split = i.split(",")
        if len(old_split) == 1:
            new_names.append(old_split[0])
        else:
            name = f"{old_split[0].replace(' ', '_')}"
            unit = f"_{old_split[1].split(' ')[1].replace('??', 'deg').replace('??', '2').replace('??', 'deg')}"
            sn = f"_sn{old_split[2].split(' ')[3][:-1]}"
            new_names.append(name+sn+unit)
    df.rename({old : new for old, new in zip(list(df.columns), new_names)}, axis=1, inplace=True)

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
        assert False, "No data found for the specified path."

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
        assert False, "No data found for the specified path."

    return ds


















############################################################################
#PLOTTING FUNCTIONS
############################################################################



def initialize_empty_map(lat_limits, lon_limits):
    """
    Function to initialize an empty map using a Mercator projection.
    This function neither draws features like coastlines, topography etc.
    nor actual meteorological data. It should be used before adding the
    above-mentioned elements using the respective functions included in
    this package.

    Parameters
    ----------
    lat_limits : list
        List with two elements: [lat_min, lat_max]
    lon_limits : list
        List with two elements: [lon_min, lon_max]

    Returns
    -------
    fig : matplotlib.figure.Figure
        The empty figure
    ax : cartopy.mpl.geoaxes.GeoAxesSubplot
        The axis of the figure

    """

    fig, ax = plt.subplots(1,1, figsize=(12,12), subplot_kw={'projection': ccrs.Mercator()})
    ax.set_extent(lon_limits+lat_limits, crs=ccrs.PlateCarree())
    gl = ax.gridlines(draw_labels=False)
    gl.left_labels = True
    gl.bottom_labels = True
    gl.xformatter = LONGITUDE_FORMATTER
    gl.yformatter = LATITUDE_FORMATTER

    return fig, ax



def map_add_coastline(fig, ax, option, color, lat_limits, lon_limits, path_mapdata):
    """
    Function to add a coastline to a previously created map figure.
    Several options are available for the resolution of the data used to draw
    the coastline (see below).

    Parameters
    ----------
    fig : matplotlib.figure.Figure
        The map figure to which the coastline should be added
    ax : cartopy.mpl.geoaxes.GeoAxesSubplot
        The axis of the map figure
    option : int
        Switch to distinguish different resolutions. Valid options:
            0 : Low resolution coastline based on the cartopy database
            1 : Medium resolution (250 m) based on data from the Norwegian Polar Institute
            2 : High resolution (100 m) based on data from the Norwegian Polar Institute
    color : str or RGB
        Color for the coastline.
    lat_limits : list
        List with two elements: [lat_min, lat_max]
    lon_limits : list
        List with two elements: [lon_min, lon_max]
    path_mapdata : str
        Absolute or relative path to the directory including the map data.

    Returns
    -------
    fig : matplotlib.figure.Figure
        The figure with the coastline added
    ax : cartopy.mpl.geoaxes.GeoAxesSubplot
        The axis of the figure

    """

    if option == 0:
        ax.coastlines(resolution="10m", linewidth=1., edgecolor=color)
    elif option == 1:
        input_file = f"{path_mapdata}NP_S250_SHP/S250_Land_l.shp"
        df_maplayer = gpd.read_file(input_file)
        df_maplayer = df_maplayer.to_crs(ccrs.Mercator().proj4_init)
        df_maplayer.plot(ax=ax, edgecolor=color, facecolor="none", zorder=20, lw=1.)
        ax.set_extent(lon_limits+lat_limits, crs=ccrs.PlateCarree())
    elif option == 2:
        input_file = f"{path_mapdata}NP_S100_SHP/S100_Land_l.shp"
        df_maplayer = gpd.read_file(input_file)
        df_maplayer = df_maplayer.to_crs(ccrs.Mercator().proj4_init)
        df_maplayer.plot(ax=ax, edgecolor=color, facecolor="none", zorder=20, lw=1.)
        ax.set_extent(lon_limits+lat_limits, crs=ccrs.PlateCarree())
    else:
        assert False, f"{option} not a valid option!"

    ax.set_title(None)
    ax.set_xlabel(None)
    ax.set_ylabel(None)

    return fig, ax


def map_add_land_filled(fig, ax, option, color, lat_limits, lon_limits, path_mapdata):
    """
    Function to fill land areas in a previously created map figure with the specified color.
    Several options are available for the resolution of the data used to define
    the land area (see below).

    Parameters
    ----------
    fig : matplotlib.figure.Figure
        The map figure to which the land area should be added
    ax : cartopy.mpl.geoaxes.GeoAxesSubplot
        The axis of the map figure
    option : int
        Switch to distinguish different resolutions. Valid options:
            0 : Low resolution based on the cartopy database
            1 : Medium resolution (250 m) based on data from the Norwegian Polar Institute
            2 : High resolution (100 m) based on data from the Norwegian Polar Institute
    color : str or RGB
        Color for the land patches
    lat_limits : list
        List with two elements: [lat_min, lat_max]
    lon_limits : list
        List with two elements: [lon_min, lon_max]
    path_mapdata : str
        Absolute or relative path to the directory including the map data.

    Returns
    -------
    fig : matplotlib.figure.Figure
        The figure with the land area colored
    ax : cartopy.mpl.geoaxes.GeoAxesSubplot
        The axis of the figure

    """

    if option == 0:
        ax.add_feature(cfeature.LAND.with_scale('10m'), facecolor=color)
    elif option == 1:
        input_file = f"{path_mapdata}NP_S250_SHP/S250_Land_f.shp"
        df_maplayer = gpd.read_file(input_file)
        df_maplayer = df_maplayer.to_crs(ccrs.Mercator().proj4_init)
        df_maplayer.plot(ax=ax, facecolor=color)
        ax.set_extent(lon_limits+lat_limits, crs=ccrs.PlateCarree())
    elif option == 2:
        input_file = f"{path_mapdata}NP_S100_SHP/S100_Land_f.shp"
        df_maplayer = gpd.read_file(input_file)
        df_maplayer = df_maplayer.to_crs(ccrs.Mercator().proj4_init)
        df_maplayer.plot(ax=ax, facecolor=color)
        ax.set_extent(lon_limits+lat_limits, crs=ccrs.PlateCarree())
    else:
        assert False, f"{option} not a valid option!"

    ax.set_title(None)
    ax.set_xlabel(None)
    ax.set_ylabel(None)

    return fig, ax




def map_add_bathymetry(fig, ax, option, color, resolution, lat_limits, lon_limits, path_mapdata):
    """
    Function to plot either contour lines of the bathymetry of an ocean area
    with the specified color or colored contours of bathymetry.
    Several options are available for the style of
    the bathymetry contours (see below).

    Parameters
    ----------
    fig : matplotlib.figure.Figure
        The map figure to which the bathymetry should be added
    ax : cartopy.mpl.geoaxes.GeoAxesSubplot
        The axis of the map figure
    option : int
        Switch to distinguish different styles. Valid options:
            0 : Bathymetry as contour lines
            1 : Bathymetry as filled contours, no colorbar
            2 : Bathymetry as filled contours, with colorbar
    color : str or RGB
        Color for the bathymetry contour lines (only used with option 0)
    resolution: float
        Resolution (distance between contour levels) of the bathymetry
    lat_limits : list
        List with two elements: [lat_min, lat_max]
    lon_limits : list
        List with two elements: [lon_min, lon_max]
    path_mapdata : str
        Absolute or relative path to the directory including the map data.

    Returns
    -------
    fig : matplotlib.figure.Figure
        The figure with the bathymetry added
    ax : cartopy.mpl.geoaxes.GeoAxesSubplot
        The axis of the figure

    """

    path_ibcao = f"{path_mapdata}IBCAO/IBCAO_v4_1_200m_t4x1y0.tif"

    bathy = rxr.open_rasterio(path_ibcao, masked=True).squeeze()
    bathy.rio.set_crs(3996)
    bathy = bathy.rio.reproject("EPSG:4326")
    bathy = bathy.rio.clip_box(minx=lon_limits[0], miny=lat_limits[0], maxx=lon_limits[1], maxy=lat_limits[1])
    bathy = bathy.rio.reproject(ccrs.Mercator().proj4_init)
    bathy = bathy.where(bathy <= 10.)


    if option == 0:
        pic = bathy.plot.contour(ax=ax, linestyles="-", linewidths=0.5, colors=color, levels=np.arange(resolution * np.floor(np.nanmin(bathy)/resolution), 1., resolution))
        ax.clabel(pic, pic.levels, inline=True, fmt="%.0f", fontsize=10)
    elif option == 1:
        bathy.plot.imshow(ax=ax, cmap=cmo.cm.deep_r, levels=np.arange(resolution * np.floor(np.nanmin(bathy)/resolution), 1., resolution), interpolation=None, add_colorbar=False)
    elif option == 2:
        pic = bathy.plot.imshow(ax=ax, cmap=cmo.cm.deep_r, levels=np.arange(resolution * np.floor(np.nanmin(bathy)/resolution), 1., resolution), interpolation=None, add_colorbar=False)
        cbar = plt.colorbar(pic, ax=ax, pad=0.02, extend="neither")
        cbar.ax.tick_params('y', labelsize=10)
        cbar.ax.set_ylabel('Height [m]', fontsize=10)
    else:
        assert False, f"{option} not a valid option!"

    ax.set_title(None)
    ax.set_xlabel(None)
    ax.set_ylabel(None)

    return fig, ax



def map_add_total_topography(fig, ax, option, resolution, lat_limits, lon_limits, path_mapdata):
    """
    Function to plot either contour lines of the total topography (above and below sea level)
    with the specified color or colored contours of the total topography.
    Several options are available for the style of
    the topography contours (see below).

    Parameters
    ----------
    fig : matplotlib.figure.Figure
        The map figure to which the total topography should be added
    ax : cartopy.mpl.geoaxes.GeoAxesSubplot
        The axis of the map figure
    option : int
        Switch to distinguish different styles. Valid options:
            0 : Topography as contour lines
            1 : Topography as filled contours, no colorbar
            2 : Topography as filled contours, with colorbar
    color : str or RGB
        Color for the topography contour lines (only used with option 0)
    resolution: float
        Resolution (distance between contour levels) of the topography
    lat_limits : list
        List with two elements: [lat_min, lat_max]
    lon_limits : list
        List with two elements: [lon_min, lon_max]
    path_mapdata : str
        Absolute or relative path to the directory including the map data.

    Returns
    -------
    fig : matplotlib.figure.Figure
        The figure with the total topography added
    ax : cartopy.mpl.geoaxes.GeoAxesSubplot
        The axis of the figure

    """

    bathy = rxr.open_rasterio(f"{path_mapdata}IBCAO/IBCAO_v4_1_200m_t4x1y0.tif", masked=True).squeeze()
    bathy.rio.set_crs(3996)
    bathy = bathy.rio.reproject("EPSG:4326")
    bathy = bathy.rio.clip_box(minx=lon_limits[0], miny=lat_limits[0], maxx=lon_limits[1], maxy=lat_limits[1])
    bathy = bathy.rio.reproject(ccrs.Mercator().proj4_init)


    if option == 0:
        pic = bathy.plot.contour(ax=ax, linestyles="-", linewidths=0.5, colors='k',
                                 levels=np.arange(resolution * np.floor(np.nanmin(bathy)/resolution), resolution * np.ceil(np.nanmax(bathy)/resolution)+1., resolution))
        ax.clabel(pic, pic.levels, inline=True, fmt="%.0f", fontsize=10)
    elif option == 1:
        bathy.plot.imshow(ax=ax, cmap=cmo.cm.topo, norm=mpl.colors.TwoSlopeNorm(0., resolution * np.floor(np.nanmin(bathy)/resolution), resolution * np.ceil(np.nanmax(bathy)/resolution)),
                          interpolation=None, add_colorbar=False)
    elif option == 2:
        pic = bathy.plot.imshow(ax=ax, cmap=cmo.cm.topo, norm=mpl.colors.TwoSlopeNorm(0., resolution * np.floor(np.nanmin(bathy)/resolution), resolution * np.ceil(np.nanmax(bathy)/resolution)),
                                interpolation=None, add_colorbar=False)
        cbar = plt.colorbar(pic, ax=ax, pad=0.02, extend="neither")
        cbar.ax.tick_params('y', labelsize=10)
        cbar.ax.set_ylabel('Height [m]', fontsize=10)
    else:
        assert False, f"{option} not a valid option!"

    ax.set_title(None)
    ax.set_xlabel(None)
    ax.set_ylabel(None)

    return fig, ax


def map_add_topography(fig, ax, option, resolution, lat_limits, lon_limits, path_mapdata):
    """
    Function to plot either contour lines of the topography
    with the specified color or colored contours of the topography.
    Several options are available for the style of
    the topography contours (see below).

    Parameters
    ----------
    fig : matplotlib.figure.Figure
        The map figure to which the topography should be added
    ax : cartopy.mpl.geoaxes.GeoAxesSubplot
        The axis of the map figure
    option : int
        Switch to distinguish different styles. Valid options:
            0 : Topography as contour lines
            1 : Topography as filled contours, no colorbar
            2 : Topography as filled contours, with colorbar
    color : str or RGB
        Color for the topography contour lines (only used with option 0)
    resolution: float
        Resolution (distance between contour levels) of the topography
    lat_limits : list
        List with two elements: [lat_min, lat_max]
    lon_limits : list
        List with two elements: [lon_min, lon_max]
    path_mapdata : str
        Absolute or relative path to the directory including the map data.

    Returns
    -------
    fig : matplotlib.figure.Figure
        The figure with the topography added
    ax : cartopy.mpl.geoaxes.GeoAxesSubplot
        The axis of the figure

    """


    if ((option == 0) | (option == 2) | (option == 4)):
        dem = rxr.open_rasterio(f"{path_mapdata}NP_S0_DTM50/S0_DTM50.tif", masked=True).squeeze()
    elif ((option == 1) | (option == 3)| (option == 5)):
        dem = rxr.open_rasterio(f"{path_mapdata}NP_S0_DTM20/S0_DTM20.tif", masked=True).squeeze()
    else:
        assert False, f"{option} not a valid option!"
    dem = dem.rio.reproject("EPSG:4326")
    dem = dem.rio.clip_box(minx=lon_limits[0], miny=lat_limits[0], maxx=lon_limits[1], maxy=lat_limits[1])
    dem = dem.rio.reproject(ccrs.Mercator().proj4_init)



    if ((option == 0) | (option == 1)):
        dem = dem.where(dem >= 0.)
        pic = dem.plot.contour(ax=ax, linestyles="-", linewidths=0.5, colors='k',
                                 levels=np.arange(0, resolution * np.ceil(np.nanmax(dem)/resolution)+1., resolution))
        ax.clabel(pic, pic.levels, inline=True, fmt="%.0f", fontsize=10)
    elif ((option == 2) | (option == 3)):
        dem = dem.where(dem > 0.)
        dem.plot.imshow(ax=ax, cmap=cmo.cm.turbid, levels=np.arange(0, resolution * np.ceil(np.nanmax(dem)/resolution)+1., resolution),
                          interpolation=None, add_colorbar=False)
    elif ((option == 4) | (option == 5)):
        dem = dem.where(dem > 0.)
        pic = dem.plot.imshow(ax=ax, cmap=cmo.cm.turbid, levels=np.arange(0, resolution * np.ceil(np.nanmax(dem)/resolution)+1., resolution),
                                interpolation=None, add_colorbar=False)
        cbar = plt.colorbar(pic, ax=ax, pad=0.02, extend="neither")
        cbar.ax.tick_params('y', labelsize=10)
        cbar.ax.set_ylabel('Height [m]', fontsize=10)
    else:
        assert False, f"{option} not a valid option!"

    ax.set_title(None)
    ax.set_xlabel(None)
    ax.set_ylabel(None)

    return fig, ax





def map_add_surface_cover(fig, ax, option, lat_limits, lon_limits, path_mapdata):
    """
    Function to colorize different types of surface cover on a previously created
    map figure. Several options are available for the resolution of the data used
    to define the surface cover areas (see below).

    Parameters
    ----------
    fig : matplotlib.figure.Figure
        The map figure to which the surface cover pathes should be added
    ax : cartopy.mpl.geoaxes.GeoAxesSubplot
        The axis of the map figure
    option : int
        Switch to distinguish different resolutions. Valid options:
            0 : Low resolution (1km) based on data from the Norwegian Polar Institute
            1 : Medium resolution (250 m) based on data from the Norwegian Polar Institute
            2 : High resolution (100 m) based on data from the Norwegian Polar Institute
    lat_limits : list
        List with two elements: [lat_min, lat_max]
    lon_limits : list
        List with two elements: [lon_min, lon_max]
    path_mapdata : str
        Absolute or relative path to the directory including the map data.

    Returns
    -------
    fig : matplotlib.figure.Figure
        The figure with the surface cover added
    ax : cartopy.mpl.geoaxes.GeoAxesSubplot
        The axis of the figure

    """

    colors = {'Elvesletter': '#CCBDA9',
              'Hav': '#B3FFFF',
              'Isbreer': '#FFFFFF',
              'Land': '#E7D3B8',
              'Morener': '#CED8D9',
              'TekniskSituasjon': '#FF8080',
              'Vann': '#99CAEB'}

    if option == 0:
        layers = ['Land', 'Vann', 'Isbreer']
        res = "1000"
    elif option == 1:
        layers = ['Land', 'Vann', 'Elvesletter', 'Isbreer', 'Morener', 'TekniskSituasjon']
        res = "250"
    elif option == 2:
        layers = ['Land', 'Vann', 'Elvesletter', 'Isbreer', 'Morener', 'TekniskSituasjon']
        res = "100"
    else:
        assert False, f"{option} not a valid option!"

    ax.set_facecolor('#B3FFFF')
    for layer in layers:
        input_file = f'{path_mapdata}NP_S{res}_SHP/S{res}_{layer}_f.shp'
        df_layer = gpd.read_file(input_file)
        df_layer = df_layer.to_crs(ccrs.Mercator().proj4_init)
        df_layer.plot(ax=ax, edgecolor=None, facecolor=colors[layer])
        ax.set_extent(lon_limits+lat_limits, crs=ccrs.PlateCarree())

    ax.set_title(None)
    ax.set_xlabel(None)
    ax.set_ylabel(None)

    return fig, ax



def map_add_crosssection_line(fig, ax, lat, lon, color='k', **kwargs):
    """
    Function to add a line indicating the position of a cross section
    on a previously created map.

    Parameters
    ----------
    fig : matplotlib.figure.Figure
        The map figure to which the cross section line should be added
    ax : cartopy.mpl.geoaxes.GeoAxesSubplot
        The axis of the map figure
    lat_limits : list
        List with two elements: [lat_min, lat_max]
    lon_limits : list
        List with two elements: [lon_min, lon_max]
    color : str or RGB, optional
        Color for the cross section line. The default is 'k'.

    Returns
    -------
    fig : matplotlib.figure.Figure
        The figure with the cross section line added
    ax : cartopy.mpl.geoaxes.GeoAxesSubplot
        The axis of the figure

    """

    df = pd.DataFrame({'latitude': lat, 'longitude': lon, 'section': 1})
    gdf = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df.longitude, df.latitude))
    gdf = gdf.groupby(['section'])['geometry'].apply(lambda x: LineString(x.tolist()))
    gdf = gpd.GeoDataFrame(gdf, geometry='geometry', crs="EPSG:4326")
    gdf = gdf.to_crs(ccrs.Mercator().proj4_init)
    gdf.plot(ax=ax, color=color, **kwargs)

    ax.set_title(None)
    ax.set_xlabel(None)
    ax.set_ylabel(None)

    return fig, ax




def map_add_points(fig, ax, option, lat, lon, color, size, label, marker="o"):
    """
    Function to add a set of points on a previously created map.
    The points can either all be in the same color and size, e.g. to indicate the locations
    of instruments, or their color and/or size can be set according to data,
    e.g. colorcode the temperature at different locations.

    Parameters
    ----------
    fig : matplotlib.figure.Figure
        The map figure to which the points should be added
    ax : cartopy.mpl.geoaxes.GeoAxesSubplot
        The axis of the map figure
    option: int
        Switch to distinguish different resolutions. Valid options:
            0 : All points with the same color and size
            1 : All points with the same color, but their size varying according
                to input data
            2 : All points with the same size, but their color varying according
                to input data
            3: All points varying in color and size according to input data
    lat : array_like
        List or array with the latitudes of the points to plot
    lon : array_like
        List or array with the longitudes of the points to plot
    color : str/RGB (options 0 and 1) or array_like (options 2 and 3)
        Color for the points. Either one fixed color or data that is going to be
        represented by the coloring.
    size : float (options 0 and 2) or array_like (options 1 and 3)
        Size of the points. Either one fixed value or data that is going to be
        represented by the size.
    label : str
        Label for the colorbar (only used for options 2 and 3)
    marker : str, optional
        A string specifying the marker style. The default is "o".

    Returns
    -------
    fig : matplotlib.figure.Figure
        The figure with the points added
    ax : cartopy.mpl.geoaxes.GeoAxesSubplot
        The axis of the figure

    """

    if option == 0:
        df = pd.DataFrame({'latitude': lat, 'longitude': lon})
    elif option == 1:
        df = pd.DataFrame({'latitude': lat, 'longitude': lon, 'size': size})
    elif option == 2:
        df = pd.DataFrame({'latitude': lat, 'longitude': lon, "color": color})
    elif option == 3:
        df = pd.DataFrame({'latitude': lat, 'longitude': lon, "color": color, 'size': size})
    else:
        assert False, f"{option} not a valid option!"

    gdf = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df.longitude, df.latitude), crs="EPSG:4326")
    gdf = gdf.to_crs(ccrs.Mercator().proj4_init)


    if option == 0:
        gdf.plot(ax=ax, color=color, markersize=size, marker=marker, zorder=10)
    elif option == 1:
        pic = ax.scatter(x=gdf["longitude"], y=gdf["latitude"], c=color, s=gdf["size"], zorder=10, cmap=cmo.cm.thermal, transform=ccrs.PlateCarree())
    elif option == 2:
        pic = ax.scatter(x=gdf["longitude"], y=gdf["latitude"], c=gdf["color"], s=size, zorder=10, cmap=cmo.cm.thermal, transform=ccrs.PlateCarree())
        cbar = plt.colorbar(pic, ax=ax, pad=0.1, orientation='horizontal')
        cbar.ax.tick_params('x', labelsize=10)
        cbar.ax.set_xlabel(label, fontsize=10)
    elif option == 3:
        pic = ax.scatter(x=gdf["longitude"], y=gdf["latitude"], c=gdf["color"], s=gdf["size"], zorder=10, cmap=cmo.cm.thermal, transform=ccrs.PlateCarree())
        cbar = plt.colorbar(pic, ax=ax, pad=0.1, orientation='horizontal')
        cbar.ax.tick_params('x', labelsize=10)
        cbar.ax.set_xlabel(label, fontsize=10)

    ax.set_title(None)
    ax.set_xlabel(None)
    ax.set_ylabel(None)

    return fig, ax


def map_add_wind_arrows(fig, ax, lat, lon, u, v, length=10, lw=1):
    """
    Function to add meteorological wind arrows on a previously created map.
    The scaling of the arrows follows the meteorological convention:
    small bar = 5 knots
    large bar = 10 knots
    tiangle = 50 knots

    Parameters
    ----------
    fig : matplotlib.figure.Figure
        The map figure to which the points should be added
    ax : cartopy.mpl.geoaxes.GeoAxesSubplot
        The axis of the map figure
    lat : array_like
        List or array with the latitudes of the points to plot
    lon : array_like
        List or array with the longitudes of the points to plot
    lat : array_like
        List or array with the x component of the winds
    lon : array_like
        List or array with the y component of the winds
    length : int, optional
        Reference for the length of the arrows. The default is 10.
    lw : int, optional
        Line thickness of the arrows. The default is 1.

    Returns
    -------
    fig : matplotlib.figure.Figure
        The figure with the arrows added
    ax : cartopy.mpl.geoaxes.GeoAxesSubplot
        The axis of the figure

    """

    u = np.array(u) * 1.94384
    v = np.array(v) * 1.94384

    df = pd.DataFrame({'latitude': lat, 'longitude': lon, 'u': u, 'v': v})
    gdf = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df.longitude, df.latitude), crs="EPSG:4326")
    gdf = gdf.to_crs(ccrs.Mercator().proj4_init)

    ax.barbs(gdf['geometry'].x, gdf['geometry'].y, gdf['u'], gdf['v'], length=length, linewidth=lw)

    ax.set_title(None)
    ax.set_xlabel(None)
    ax.set_ylabel(None)

    return fig, ax
