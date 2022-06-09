# -*- coding: utf-8 -*-
"""
This module contains scripts to download data from the AROME-Arctic weather model
from the Norwegian Meteorological Institute. The code is optimized
for the use in the UNIS courses.
"""
import unisacsi
import numpy as np
import copy




def Rotate_uv_components(ur,vr,cordsx,cordsy,longitude,data_type,model="AA"):
    """
    Function rotating the u and v wind components from the model grid to regular
    West-East and South-North components

    Parameters
    ----------
    ur : array
        u wind component in the model coordinate system
    vr : array
        v wind component in the model coordinate system
    cordsx : array
        x-coordinates of the data points in the model coordinate system
    cordsy : array
        x-coordinates of the data points in the model coordinate system
    longitude : TYPE
        longitudes of the data points
    data_type : int
        Switch to distinguish between point (1) and 2D (2) input data
    model : str, optional
        Name of the model, either "AA" or "MC"

    Returns
    -------
    list
        List with arrays of u and v as entries.

    """

    if model == "AA":
        truelat1 = 77.5 # true latitude
        stdlon   = -25  # standard longitude
    elif model == "MC":
        truelat1 = 63.3 # true latitude
        stdlon   = 15.  # standard longitude
    else:
        assert False, "Model name not recognized, specify either 'AA' or 'MC'."

    cone = np.sin(np.abs(np.deg2rad(truelat1))) # cone factor

    diffn = stdlon - longitude
    diffn[diffn>180.] -= 360
    diffn[diffn<-180.] += 360

    alpha  = np.deg2rad(diffn) * cone

    if data_type == 1:
        alphan = alpha[cordsx,cordsy]
    elif data_type == 2:
        alphan = alpha[cordsx,:][:,cordsy]
    else:
        assert False, "Input data type not recognized, specify either '1' for point data \
            or '2' for 2D data."

    u = np.squeeze(ur) * np.cos(alphan) - np.squeeze(vr) * np.sin(alphan)
    v = np.squeeze(vr) * np.cos(alphan) + np.squeeze(ur) * np.sin(alphan)

    return [u,v]




def lonlat2xy(lon,lat,lons,lats,data_type):
    """
    Function to find the closest model grid point to the specified lon-lat location.

    Parameters
    ----------
    lon : float or array
        Longitude of interest
    lat : float or array
        Latitude of interest
    lons : array
        Longitudes of the whole model domain (from the static fields file)
    lats : float
        Latitudes of the whole model domain (from the static fields file)
    data_type : int
        Switch to distinguish between point (1) and 2D (2) input data

    Returns
    -------
    list
        List with the x and y coordinates of the nearest model grid point\
            together with the respective longitude and latitude

    """



    if data_type == 1:

        latt2 = lats
        lonn2 = lons

        latt1 = lat
        lonn1 = lon

        radius = 6378.137
        lat1 = latt1 * (np.pi/180.)
        lat2 = latt2 * (np.pi/180.)
        lon1 = lonn1 * (np.pi/180.)
        lon2 = lonn2 * (np.pi/180.)
        deltaLat = lat2 - lat1
        deltaLon = lon2 - lon1

        x = deltaLon * np.cos((lat1+lat2)/2.)
        y = deltaLat
        d2km = radius * np.sqrt(x**2. + y**2.)


        xx, yy = np.unravel_index(d2km.argmin(), d2km.shape)

        output1 = xx
        output2 = yy
        output3 = lons[xx, yy]
        output4 = lats[xx, yy]


    elif data_type == 2:

        lon1 = lon[0]
        lon2 = lon[1]
        lat1 = lat[0]
        lat2 = lat[1]

        lon_corners  = [lon1, lon2, lon2, lon1]
        lat_corners  = [lat1, lat1, lat2, lat2]

        coords_xx = np.zeros(len(lon_corners))
        coords_yy = np.zeros(len(lon_corners))

        for qq in range(len(lon_corners)):
            latt2 = lats
            lonn2 = lons

            lonn1 = lon_corners[qq]
            latt1 = lat_corners[qq]

            radius = 6378.137
            lat1 = latt1 * (np.pi/180.)
            lat2=latt2 * (np.pi/180.)
            lon1=lonn1 * (np.pi/180.)
            lon2=lonn2 * (np.pi/180.)
            deltaLat = lat2 - lat1
            deltaLon = lon2 - lon1

            x = deltaLon * np.cos((lat1+lat2)/2.)
            y = deltaLat
            d2km = radius * np.sqrt(x**2. + y**2.)

            coords_xx[qq], coords_yy[qq] = np.unravel_index(d2km.argmin(), d2km.shape)

        lonmin_id = int(np.min(coords_xx))
        lonmax_id = int(np.max(coords_xx))
        latmin_id = int(np.min(coords_yy))
        latmax_id = int(np.max(coords_yy))

        output1 = np.array([lonmin_id, latmin_id])
        output2 = np.array([np.abs(lonmax_id-lonmin_id), np.abs(latmax_id-latmin_id)])
        output3 = [1,1]
        output4 = np.nan

    else:
        assert False, "Input data type not recognized, specify either '1' for point data \
            or '2' for 2D data."


    return [output1, output2, output3, output4]





def Calculate_height_levels_and_pressure(hybrid,ap,b,t0,PSFC,T,data_type):
    """
    Function to calcualte the height of the model levels

    Parameters
    ----------
    hybrid : array
        hybrid-sigma-pressure_coordinate
    ap : array
        conversion factor
    b : array
        conversion factor
    t0 : array
        surface temperature
    PSFC : array
        surface pressure
    T : array
        temperature
    data_type : int
        Switch to distinguish between single profiles (1) and 3D (3) input data

    Returns
    -------
    list
        List with two entries: height and pressure of the model levels

    """

    R = 287. # ideal gas constant
    g = 9.81 # acceleration of gravity

    if data_type == 1:
        PN = np.zeros(len(T)+1)

        # Calculating pressure levels
        for n in range(len(T)):
            PN[n] = ap[n] + b[n]*PSFC

        # Adding surface data as the lowest level
        PN[-1] = PSFC
        TN = copy.copy(T)
        TN = np.append(TN, t0)

        heightt = np.zeros(len(hybrid)+1)

        # Calculating height levels (in metres) based on the hypsometric equation and assuming a dry atmosphere
        for n in range(len(T),0,-1):
            pd = PN[n]/PN[n-1]
            TM = np.mean([TN[n], TN[n-1]])
            heightt[n-1] = heightt[n] + R*TM/g*np.log(pd)

        height = heightt[:-1]
        P = PN[:-1]

    elif data_type == 3:
        PN = np.zeros((T.shape[0], T.shape[1], T.shape[2]+1))

        # Calculating pressure levels
        for k in range(T.shape[2]):
            PN[:,:,k] = ap[k] + b[k]*PSFC

        # Adding surface data as the lowest level
        PN[:,:,-1] = PSFC[:,:]
        TN = np.concatenate((T[:,:,:], np.expand_dims(t0, axis=2)), axis=2)

        heightt = np.zeros((T.shape[0], T.shape[1], T.shape[2]+1))

        # Calculating height levels (in metres) based on the hypsometric equation assuming a dry atmosphere
        for n in range(T.shape[2],0,-1):
            pd = PN[:,:,n]/PN[:,:,n-1]
            TM = np.mean(np.array([TN[:,:,n], TN[:,:,n-1]]), axis=0)
            heightt[:,:,n-1] = heightt[:,:,n] + R*TM/g*np.log(pd)

        height = heightt[:,:,:-1]
        P = PN[:,:,:-1]

    else:
        assert False, "Input data type not recognized, specify either '2' for 2D data \
            or '3' for 3D data."

    return [height,P]
