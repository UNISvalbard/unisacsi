# -*- coding: utf-8 -*-
"""
Created on Tue Jun  7 09:47:46 2022

@author: lukasf
"""

import Ocean as Oc
import Meteo as Met
import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import cartopy.crs as ccrs
import gsw
import cmocean as cmo


path_data = "C:/Users/lukasf/Desktop/Example_data/"



def example_CTD():
    
    plt.close("all")
    
    global path_data                # use variable from outside the function
    
    help(Oc.read_CTD)               # get help on a certain function (works for all Python functions)
    
    CTD = Oc.read_CTD(f"{path_data}CTD/")       # read all CTD data files
    
    print(CTD.keys())                           # print a list of all CTD stations

    print(CTD[1241].keys())                     # print a list of variables at a certain station
    
    storfjorden_section = [1241] + list(range(1243,1250))       # define the Storfjorden section as a list of certain stations
    
    
    # plot the density profiles from all stations of the Storfjorden section
    plt.figure()
    for station in storfjorden_section:
        plt.plot(CTD[station]['SIGTH'],-CTD[station]['z'],label=station)
    plt.gca().invert_yaxis()
    plt.xlabel('Density')
    plt.ylabel('Depth (m)')
    plt.grid()
    plt.legend()
    
    
    
    # Convert the CTD data to pandas dataframes

    # option 1: Convert a single station to a dataframe, with z as index, and the variables as columns
    station_df = pd.DataFrame(CTD[1241])
    station_df.index = -station_df['z']
    print(station_df)
    
    # option 2: convert one variable (in this case: Oxygen) of several stations to a DataFrame, with z as index, and the stations as columns
    CTD_i,Z,_,_ = Oc.CTD_to_grid(CTD,storfjorden_section,interp_opt=0)
    variable_df = pd.DataFrame(CTD_i['OX'],index=Z,columns=storfjorden_section)
    print(variable_df)
    
    
    
    
    
    # plot maps of the CTD stations

    # one map including all stations
    plt.figure()
    Oc.plot_CTD_map(CTD, extent=[7.,25.,75.5,80.], topography=f"{path_data}bathymetry.mat")
    
    # zoom into a certain area and plot only the Storfjorden section
    plt.figure()
    Oc.plot_CTD_map(CTD, extent=[16.,23.,76.,79.], stations=storfjorden_section, topography=f"{path_data}bathymetry.mat", adjust_text=True)
    
    
    
    
    
    # plot crosssections of a variable along a CTD section
    
    # with linear interpolation
    Oc.plot_CTD_single_section(CTD, storfjorden_section, parameter='OX',
                               clabel='Oxygen (mL/L)',
                              cmap='cmo.thermal',cruise_name='Storfjorden')
    
    # without interpolation
    Oc.plot_CTD_single_section(CTD, storfjorden_section, parameter='OX',
                               clabel='Oxygen (mL/L)',
                              cmap='cmo.thermal',cruise_name='Storfjorden', interp_opt=0)
    
    
    
    
    
    
    
    # plot a TS diagram of all stations from the Storfjorden section

    # automatic
    plt.figure()
    Oc.plot_CTD_ts(CTD,storfjorden_section)
    
    # manual
    plt.figure()
    Oc.create_empty_ts(T_extent=[-2., 5.],S_extent=[28.,38.])
    for station in storfjorden_section:
        S = plt.scatter(CTD[station]['SA'],CTD[station]['CT'],
                    s=5,c=CTD[station]['OX'],cmap='cmo.matter',
                       vmin=5.8,vmax=7.8)
    plt.colorbar(S,label='Oxygen (mL/L)')
    
    
    plt.show()
    
    return





###############################################################################
###############################################################################






def example_mooring():
    
    plt.close("all")
    
    global path_data
    
    
    mooring = Oc.read_mooring(f"{path_data}IS1617.mat")

    # print all information
    print(mooring.keys())
    
    # print e.g. the latitude of the mooring
    print(mooring["lat"])
    
    # convert the time arrays into Python timestamps:
    for key in mooring.keys():
        if key[:4] == "date":
            mooring[key] = Oc.mat2py_time(np.asarray(mooring[key]))
            
    print(mooring["date5453"])
    
    #%% plot a time series from the mooring
    
    plt.figure(figsize=(20,3))
    plt.plot(mooring["date556"], mooring["T556"])
    plt.ylabel('Temperature [degC]')
    plt.grid()
    
    plt.show()
    
    return



    
    
###############################################################################
###############################################################################






def example_ADCP():
    
    plt.close("all")
    
    global path_data
    
    adcp = Oc.read_ADCP_UNIS(f"{path_data}os75bb_long.nc")
    
    
    # Plot a map of the section, and show the velocities in the top ADCP layer
    
    # select the start and the end time of the section 
    sec_start = mpl.dates.datestr2num('2021-10-08 02:20') 
    sec_end = mpl.dates.datestr2num('2021-10-08 09:00')
    
    # Cut the ADCP data to this time window
    t_index = np.where((sec_start<adcp['time'])&(adcp['time']<sec_end))[0]
    
    time,lat,lon = [adcp[a][t_index]   for a in ['time','lat','lon'] ]
    u,v,crossvel = [adcp[a][t_index,:] for a in ['u','v','crossvel'] ]
    depth = adcp['depth'][-1,:]
    
    extent=[7.,25.,75.5,80.]
    data_proj = ccrs.PlateCarree()
    plt.figure(figsize=(9,9))
    
    ax = Oc.plot_empty_map(extent=extent, topography=f"{path_data}bathymetry.mat");
    
    # First plot the shiptrack
    ax.plot(lon,lat,transform=data_proj, color='b')
    
    # Then plot the velocity as arrows
    q = ax.quiver(lon,lat, u=u[:,0], v=v[:,0], scale=5., transform=data_proj, color = 'b')
    # Put a legend for the arrows
    plt.quiverkey(q,0.1,0.95,0.5,'0.5 m/s',color='blue',labelcolor='black',
                      transform=plt.gca().transAxes)
    
    # Indicate on the map, where the section starts, and where it ends
    ax.text(lon[0],lat[0],'A',transform=data_proj,fontweight='bold',
           horizontalalignment='center',verticalalignment='top',
           fontsize=15)
    ax.text(lon[-1],lat[-1],'B',transform=data_proj,fontweight='bold',
           horizontalalignment='center',verticalalignment='top',
           fontsize=15)
    
    
    
    
    
    # Plot the section over distance
    
    # Get the distance from the start of the section
    distance = np.insert(np.cumsum(gsw.distance(lon,lat)/1000),0,0)
    
    plt.figure(figsize=(10,6))
    
    plt.contourf(distance, depth, -crossvel.transpose(), cmap="RdBu", levels=np.arange(-.5, .5, 0.05), extend="both")
    
    # Add information and make the map a bit nicer
    # plt.ylim(0,300)
    plt.gca().invert_yaxis()
    plt.colorbar(label='cross-velocity (m/s)')
    plt.text(distance[0],-0.5,'A',fontweight='bold')
    plt.text(distance[-1],-0.5,'B',fontweight='bold')
    plt.xlabel('Distance (km)')
    plt.ylabel('Depth (m)');
    plt.title('Negative cross velocity = velocity to the left (in the direction of the section)')
    
    plt.show()
    
    return
        
        


    
    
###############################################################################
###############################################################################



        
    
    
def example_meteorological_timeseries():
    
    plt.close("all")
    
    global path_data
    
    
    # radiation
    df_rad = Met.read_Campbell_radiation(f"{path_data}TOA5_19688.rad_2022_02_11_1145.dat")
    
    fig, ax = plt.subplots(1,1)
    with pd.plotting.plot_params.use('x_compat', True):
        df_rad.plot(y="CM3Up_Avg", ax=ax, c="b")
        df_rad.plot(y="CM3Dn_Avg", ax=ax, c="g")
        df_rad.plot(y="CG3UpCo_Avg", ax=ax, c="r")
        df_rad.plot(y="CG3DnCo_Avg", ax=ax, c="orange")
    ax.set_xlabel(None)
    ax.set_ylabel("irradiance [W/m^2]")
    ax.grid("both")
    ax.xaxis.set_major_formatter(mpl.dates.DateFormatter('%d.%b'))
    
    
    
    # TinyTags
    df_TT = Met.read_Tinytag(f"{path_data}20220318_TT4.txt", "TT")
    df_TH = Met.read_Tinytag(f"{path_data}20220318_TH3.txt", "TH")
    df_CEB = Met.read_Tinytag(f"{path_data}20210911_CEB1.txt", "CEB")
    
    fig, (ax_T, ax_CEB) = plt.subplots(2,1)
    ax_RH = ax_T.twinx()
    df_TT.plot(y="T_black_degC", ax=ax_T, c="darkblue")
    df_TT.plot(y="T_white_degC", ax=ax_T, c="lightblue")
    df_TH.plot(y="T_degC", ax=ax_T, c="b")
    df_TH.plot(y="RH_%RH", ax=ax_RH, c="r")
    ax_T.set_xlabel(None)
    ax_T.set_ylabel("temperature [degC]", c="b")
    ax_RH.set_ylabel("relative humidity [%]", c="r")
    ax_RH.spines['left'].set_color('b')
    ax_RH.spines['right'].set_color('r')
    ax_T.tick_params(axis='y', colors="b", labelcolor="b")
    ax_RH.tick_params(axis='y', colors="r", labelcolor="r")
    ax_T.grid("both")
    ax_T.legend(loc="lower right")
    ax_RH.legend(loc="upper left")
    
    df_CEB.plot(y="T_degC", ax=ax_CEB, c="g")
    ax_CEB.set_xlabel(None)
    ax_CEB.set_ylabel("soil temperature [degC]")
    ax_CEB.grid()
    
    plt.show()
    
    return





    
    
###############################################################################
###############################################################################







def example_meteorological_map():
    
    plt.close("all")
    
    global path_data
    
    aa = Met.read_AROME(f"{path_data}Arome_Arctic/AA_nearsurface_*.nc")
    
    fig, ax = Met.initialize_empty_map(lat_limits=[78., 79.], lon_limits=[13., 18.])
    fig, ax = Met.map_add_coastline(fig, ax, option=1, color="k", lat_limits=[78., 79.], lon_limits=[13., 18.], path_mapdata=f"{path_data}Svalbard_map_data/")
    
    pic = ax.contourf(aa.lon, aa.lat, aa.T.isel(time=3), levels=100, cmap=cmo.cm.thermal, transform=ccrs.PlateCarree())
    cbar = plt.colorbar(pic, ax=ax)
    cbar.ax.set_xlabel("2m temperature")
    
    plt.show()
    
    return
    
    
    



    
    
    