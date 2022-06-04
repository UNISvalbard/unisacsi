# -*- coding: utf-8 -*-
"""
This module contains scripts to download data from the weather models
of the Norwegian Meteorological Institute, AROME-Arctic and MetCoOp.
The code is optimized for the use in the UNIS courses.
"""

from netCDF4 import Dataset
import numpy as np
import pandas as pd
import yaml
import sys
import xarray as xr
import MET_model_tools as Mmt


def set_parameters():

    int_h = 1                        # Time interval between data time steps in each data file (1 hour)
    int_f = 3                        # Time interval in hours between each data file (3 hours)
    
    variables_names_units =    {'T': ['temperature', "degC"],
                                'T_pot': ["potential_temperature", "degC"],
                                'RH': ['relative_humidity', "%"],
                                'u': ['x_wind', "m/s"],
                                'v': ['y_wind', "m/s"],
                                "WS": ["wind_speed", "m/s"],
                                "WD": ["wind_direction", "deg"],
                                "q": ["specific_humidity", "g/kg"],
                                'p_surf': ['surface_pressure', "hPa"],
                                'T_surf': ['surface_temperature', "degC"],
                                'MSLP': ["air_pressure_at_sea_level", "hPa"],
                                'tau_x': ["downward_eastward_momentum_flux", "N/m^2"],
                                'tau_y': ["downward_northward_momentum_flux", "N/m^2"],
                                'SW_up': ["upwelling_shortwave_radiation", "W/m^2"],
                                'SW_down': ["downwelling_shortwave_radiation", "W/m^2"],
                                'LW_up': ["upwelling_longwave_radiation", "W/m^2"],
                                'LW_down': ["downwelling_longwave_radiation", "W/m^2"],
                                'LHF': ["downward_latent_heat_flux", "W/m^2"],
                                'SHF': ["downward_sensible_heat_flux", "W/m^2"],
                                'cloud_cover': ['cloud_area_fraction', "%"],
                                'ABL_height': ['atmosphere_boundary_layer_thickness', "m"],
                                "p": ["pressure", "hPa"],
                                "z": ["height", "m"],
                                'time': ["time", "seconds since 1970-01-01 00:00:00"]}
    
    return [int_h, int_f, variables_names_units]


def get_download_config(start_time, end_time, start_h, num_h, int_f, int_h, static_file, vari_config_file, model):
    
    time_vec = pd.date_range(start_time, end_time, freq=f"{int_f}H", closed="left")
    time_ind = np.arange(start_h, start_h+num_h, int_h, dtype=int)
    
    static_fields = {}
    with Dataset(static_file, 'r') as f:
        static_fields["lon"] = f.variables["lon"][:]
        static_fields["lat"] = f.variables["lat"][:]
        static_fields["orog"] = f.variables["orog"][:]
        static_fields["lsm"] = f.variables["lsm"][:]
    
    
    fileurls = []
    if model == "AA":
        full_model_name = "AROME_Arctic"
        for t in time_vec:
            fileurls.append(f'https://thredds.met.no/thredds/dodsC/aromearcticarchive/{t.strftime("%Y/%m/%d")}/arome_arctic_det_2_5km_{t.strftime("%Y%m%d")}T{t.strftime("%H")}Z.nc')
    elif model == "MC":
        full_model_name= "METCoOp"
        for t in time_vec:
            fileurls.append(f'https://thredds.met.no/thredds/dodsC/meps25epsarchive/{t.strftime("%Y/%m/%d")}/meps_det_2_5km_{t.strftime("%Y%m%d")}T{t.strftime("%H")}Z.nc')
    else:
        assert False, "Model name not recognized, specify either 'AA' or 'MC'."
        
    
    with open(vari_config_file, "r") as stream:
        varis = [k for k, s in yaml.safe_load(stream).items() if s  == 1]
    if "wind" in varis:
        varis.remove("wind")
        varis += ["ur", "vr"]
    if "momentum_flux" in varis:
        varis.remove("momentum_flux")
        varis += ["tau_x", "tau_y"]
    if "radiation" in varis:
        varis.remove("radiation")
        varis += ['SW_net', 'SW_down', 'LW_net', 'LW_down']
    if "turb_fluxes" in varis:
        varis.remove("turb_fluxes")
        varis += ["LHF", "SHF"]
        
    return [time_vec, time_ind, fileurls, varis, static_fields, full_model_name]





def download_static_fields(out_path, model="AA"):
    """
    Function to download and save the static fields (lon, lat, orog and lsm)
    of one of the models of the Norwegian Meteorological Institute.

    Parameters
    ----------
    out_path : str
        Path specifying the location and file name where the static fields get saved
    model: str, optional
        Name of the model, either 'AA' for Arome-Arctic or 'MC' for MetCoOp

    Returns
    -------
    None.

    """

    if model == "AA":
        file = 'https://thredds.met.no/thredds/dodsC/aromearcticarchive/2022/06/03/arome_arctic_det_2_5km_20220603T00Z.nc'
        comment = "AROME-Arctic static fields of full data files with 2.5 km horizontal resolution"
    elif model == "MC":
        file = 'https://thredds.met.no/thredds/dodsC/meps25epsarchive/2022/02/20/meps_det_2_5km_20220220T00Z.nc'
        comment = "MetCoOp static fields of full data files with 2.5 km horizontal resolution"
        
    with Dataset(file) as f:
        x = f.variables['x'][:]
        y = f.variables['y'][:]
        AA_longitude = f.variables['longitude'][:]
        AA_latitude = f.variables['latitude'][:]
        AA_topo_height = np.squeeze(f.variables['surface_geopotential'][0,:,:])
        AA_lsm = np.squeeze(f.variables['land_area_fraction'][0,:,:])
    
    with Dataset(out_path, 'w', format="NETCDF4") as f:
        f.Comments = comment
        f.createDimension('x', len(x))
        f.createDimension('y', len(y))
    
        var = f.createVariable('lon', 'f4', ('x', 'y',))
        var.units = 'degree_north'
        var.long_name = 'longitude'
        var[:] = np.transpose(AA_longitude)
    
        var = f.createVariable('lat', 'f4', ('x', 'y',))
        var.units = 'degree_east'
        var.long_name = 'latitude'
        var[:] = np.transpose(AA_latitude)
    
        var = f.createVariable('orog', 'f4', ('x', 'y',))
        var.units = 'm'
        var.long_name = 'orography'
        var[:] = np.transpose(AA_topo_height) / 9.81
    
        var = f.createVariable('lsm', 'f4', ('x', 'y',))
        var.units = '1'
        var.long_name = 'land-sea-mask'
        var[:] = np.transpose(AA_lsm)
        
    print(f"Static fields were successfully saved into {out_path}.")
        
    return






def download_nearsurface_timeseries(start_time, end_time, stt_name, stt_lon, stt_lat, start_h, num_h, out_path, static_file, vari_config_file, model="AA"):

    int_h, int_f, variables_names_units = set_parameters()
    
    time_vec, time_ind, fileurls, varis, static_fields, full_model_name = \
        get_download_config(start_time, end_time, start_h, num_h, int_f, int_h, static_file, vari_config_file, model)
        
    if "Humidity" in varis:
        varis.remove("Humidity")
        varis += ["RH", "q"]
        
    model_varnames =    {'T': 'air_temperature_2m',
                         'RH': 'relative_humidity_2m',
                         'q': 'specific_humidity_2m',
                         'ur': 'x_wind_10m',
                         'vr': 'y_wind_10m',
                         'p_surf': 'surface_air_pressure',
                         'T_surf': 'air_temperature_0m',
                         'MSLP': "air_pressure_at_sea_level",
                         'tau_x': "downward_eastward_momentum_flux_in_air",
                         'tau_y': "downward_northward_momentum_flux_in_air",
                         'SW_net': "integral_of_surface_net_downward_shortwave_flux_wrt_time",
                         'SW_down': "integral_of_surface_downwelling_shortwave_flux_in_air_wrt_time",
                         'LW_net': "integral_of_surface_net_downward_longwave_flux_wrt_time",
                         'LW_down': "integral_of_surface_downwelling_longwave_flux_in_air_wrt_time",
                         'LHF': "integral_of_surface_downward_latent_heat_flux_wrt_time",
                         'SHF': "integral_of_surface_downward_sensible_heat_flux_wrt_time",
                         'cloud_cover': 'cloud_area_fraction',
                         'ABL_height': 'atmosphere_boundary_layer_thickness'}
    
    varis_to_load = {}
    for v in varis:
        try:
            varis_to_load[v] = model_varnames[v]
        except KeyError:
            print(f"The variable {v} is unfortunately not available as nearsurface time series, please de-select in the configuration file and try again.")
            sys.exit(1)
            
    # Finding nearest model grid points (x,y) to the longitude and latitude coordinates of the stations in the station list
    coords_xx = np.zeros_like(stt_lon, dtype=int)
    coords_yy = np.zeros_like(stt_lon, dtype=int)
    model_lon = np.zeros_like(stt_lon)
    model_lat = np.zeros_like(stt_lon)
    model_orog = np.zeros_like(stt_lon)
    model_lsm = np.zeros_like(stt_lon)
    for i, (stat, lon, lat) in enumerate(zip(stt_name, stt_lon, stt_lat)):
        coords_xx[i], coords_yy[i], model_lon[i], model_lat[i] = Mmt.lonlat2xy(lon, lat, static_fields['lon'], static_fields['lat'], 1)
        
        model_orog[i] = static_fields['orog'][coords_xx[i], coords_yy[i]]
        model_lsm[i] = static_fields['lsm'][coords_xx[i], coords_yy[i]]

    time = []
    data = {}
    
    # determine overall size of time dimension
    len_time = 0                                          # overall time index
    for filename in fileurls:                 # loop over files --> time in the order of days
        for a in range(num_h):                     # loop over timesteps --> time in the order of hours
            len_time += 1
            
    for qr, stat in enumerate(stt_name):           # loop over stations
        data[stat] = {}

        for vari in varis_to_load.keys():
            data[stat][vari] = np.zeros((len_time))

        nn = 0                                          # overall time index
        for filename in fileurls:                 # loop over files --> time in the order of days

            with Dataset(filename, 'r') as f:
                for qrr in time_ind:                      # loop over timesteps --> time in the order of hours
                    if qr == 0:
                        time.append(f.variables['time'][qrr])

                    # Retrieving variables from MET Norway server thredds.met.no
                    for vari, vari_met in varis_to_load.items():
                        if vari_met[:8] == "integral":
                            data[stat][vari][nn] = np.squeeze(f.variables[vari_met][qrr,0,coords_yy[qr],coords_xx[qr]]) / (qrr*3600.)
                        else:
                            data[stat][vari][nn] = np.squeeze(f.variables[vari_met][qrr,0,coords_yy[qr],coords_xx[qr]])

                        print(f'Done reading variable {vari_met} from file {filename} on thredds server')

                    nn += 1
                
        if "SW_net" in varis_to_load:
            # Converting radiative fluxes from net into up
            data[stat]["SW_up"] = data[stat]["SW_down"] - data[stat]["SW_net"]
            data[stat]["LW_up"] = data[stat]["LW_down"] - data[stat]["LW_net"]
            
            del data[stat]['SW_net']
            del data[stat]['LW_net']

        if "ur" in varis_to_load:
            # Wind u and v components in the original data are grid-related.
            # Therefore, we rotate here the wind components from grid- to earth-related coordinates.
            data[stat]['u'], data[stat]['v'] = Mmt.Rotate_uv_components(data[stat]['ur'], data[stat]['vr'],coords_xx[qr], coords_yy[qr], static_fields["lon"],1,model)
    
            # Calculating wind direction
            data[stat]['WD'] = (np.rad2deg(np.arctan2(-data[stat]['u'], -data[stat]['v']))+360.) % 360.
    
            # Calculating wind speed
            data[stat]['WS'] = np.sqrt((data[stat]['u']**2.) + (data[stat]['v']**2.))
            
            del data[stat]['ur']
            del data[stat]['vr']

        if (("T" in varis_to_load) & ("p_surf" in varis_to_load)):
            # Calculating potential temperature
            data[stat]['T_pot'] = ((data[stat]['T'])*((1.e5/(data[stat]['p_surf']))**(287./1005.))) - 273.15

        # Converting units
        if "p_surf" in data[stat].keys():
            data[stat]['p_surf'] /= 100.
        if "MSLP" in data[stat].keys():
            data[stat]['MSLP'] /= 100.
        if "T" in data[stat].keys():
            data[stat]['T'] -= 273.15
        if "T_surf" in data[stat].keys():
            data[stat]['T_surf'] -= 273.15
        if "RH" in data[stat].keys():
            data[stat]['RH'] *= 100.
            data[stat]['q'] *= 1000.
        if "cloud_cover" in data[stat].keys():
            data[stat]['cloud_cover'] *= 100.


    time = np.array(time)
    
    data_reorg = {vari: np.array([data[s][vari] for s in stt_name]) for vari in data[stt_name[0]].keys()}
    
    d_dict = {
        "coords": {
            "time": {"dims": "time", "data": time, "attrs": {"units": "seconds since 1970-01-01 00:00:00"}},
            "station_name": {"dims": "station", "data": stt_name, "attrs": {"units": "1"}},
            "station_lat": {"dims": "station", "data": stt_lat, "attrs": {"units": "degN", "long_name": "latitude_of_station"}},
            "station_lon": {"dims": "station", "data": stt_lon, "attrs": {"units": "degE", "long_name": "longitude_of_station"}},
            "model_lat": {"dims": "station", "data": model_lat, "attrs": {"units": "degN", "long_name": "latitude_of_model_gridpoint"}},
            "model_lon": {"dims": "station", "data": model_lon, "attrs": {"units": "degE", "long_name": "longitude_of_model_gridpoint"}},
            "model_lsm": {"dims": "station", "data": model_lsm, "attrs": {"units": "1", "long_name": "land_sea_mask_of_model_gridpoint"}},
            "model_orog": {"dims": "station", "data": model_orog, "attrs": {"units": "m", "long_name": "elevation_of_model_gridpoint"}},
            },
        "attrs": {"Comments": f"Near-surface time series data extracted from {full_model_name} simulations."},
        "dims": ["station", "time"],
        "data_vars": {
            vari: {"dims": ["station", "time"],
                   "data": d,
                   "attrs": {"units": variables_names_units[vari][1],
                             "long_name": variables_names_units[vari][0]}} for vari, d in data_reorg.items()}}
    
    ds = xr.Dataset.from_dict(d_dict)
    ds = ds.set_index({"station": "station_name"}, append=True)
    
    ds.to_netcdf(out_path)
    
    return
    
    
    
    
def download_nearsurface_fields(start_time, end_time, lon_lims, lat_lims, start_h, num_h, out_path, static_file, vari_config_file, model="AA", int_x=1, int_y=1):

    int_h, int_f, variables_names_units = set_parameters()
    
    time_vec, time_ind, fileurls, varis, static_fields, full_model_name = \
        get_download_config(start_time, end_time, start_h, num_h, int_f, int_h, static_file, vari_config_file, model)
        
    if "Humidity" in varis:
        varis.remove("Humidity")
        varis += ["RH", "q"]
        
    model_varnames =    {'T': 'air_temperature_2m',
                         'RH': 'relative_humidity_2m',
                         'q': 'specific_humidity_2m',
                         'ur': 'x_wind_10m',
                         'vr': 'y_wind_10m',
                         'p_surf': 'surface_air_pressure',
                         'T_surf': 'air_temperature_0m',
                         'MSLP': "air_pressure_at_sea_level",
                         'tau_x': "downward_eastward_momentum_flux_in_air",
                         'tau_y': "downward_northward_momentum_flux_in_air",
                         'SW_net': "integral_of_surface_net_downward_shortwave_flux_wrt_time",
                         'SW_down': "integral_of_surface_downwelling_shortwave_flux_in_air_wrt_time",
                         'LW_net': "integral_of_surface_net_downward_longwave_flux_wrt_time",
                         'LW_down': "integral_of_surface_downwelling_longwave_flux_in_air_wrt_time",
                         'LHF': "integral_of_surface_downward_latent_heat_flux_wrt_time",
                         'SHF': "integral_of_surface_downward_sensible_heat_flux_wrt_time",
                         'cloud_cover': 'cloud_area_fraction',
                         'ABL_height': 'atmosphere_boundary_layer_thickness'}
    
    varis_to_load = {}
    for v in varis:
        try:
            varis_to_load[v] = model_varnames[v]
        except KeyError:
            print(f"The variable {v} is unfortunately not available as nearsurface time series, please de-select in the configuration file and try again.")
            sys.exit(1)
        

    start_lonlat, count_lonlat, _, _ = Mmt.lonlat2xy(lon_lims, lat_lims, static_fields["lon"], static_fields["lat"], 2)
    
    idx = np.arange(start_lonlat[0], (start_lonlat[0]+count_lonlat[0]+1))
    idy = np.arange(start_lonlat[1], (start_lonlat[1]+count_lonlat[1]+1))
    idxx = idx[::int_x]
    idyy = idy[::int_y]
    
    model_lon = static_fields['lon'][idxx,:][:,idyy]
    model_lat = static_fields['lat'][idxx,:][:,idyy]
    model_orog = static_fields['orog'][idxx,:][:,idyy]
    model_lsm = static_fields['lsm'][idxx,:][:,idyy]

    time = []
    data = {}
    
    # determine overall size of time dimension
    len_time = 0                                          # overall time index
    for filename in fileurls:                 # loop over files --> time in the order of days
        for a in range(num_h):                     # loop over timesteps --> time in the order of hours
            len_time += 1
            

    for vari in varis_to_load.keys():
        data[vari] = np.zeros((model_lon.shape[0], model_lon.shape[1], len_time))

    nn = 0                                          # overall time index
    for filename in fileurls:                 # loop over files --> time in the order of days

        with Dataset(filename, 'r') as f:
            for qrr in time_ind:                      # loop over timesteps --> time in the order of hours
                time.append(f.variables['time'][qrr])

                # Retrieving variables from MET Norway server thredds.met.no
                for vari, vari_met in varis_to_load.items():
                    if vari_met[:8] == "integral":
                        data[vari][:,:,nn] = np.transpose(np.squeeze(f.variables[vari_met][qrr,0,idyy,idxx]) / (qrr*3600.))
                    else:
                        data[vari][:,:,nn] = np.transpose(np.squeeze(f.variables[vari_met][qrr,0,idyy,idxx]))

                    print(f'Done reading variable {vari_met} from file {filename} on thredds server')

                nn += 1
            
    if "SW_net" in varis_to_load:
        # Converting radiative fluxes from net into up
        data["SW_up"] = data["SW_down"] - data["SW_net"]
        data["LW_up"] = data["LW_down"] - data["LW_net"]
        
        del data['SW_net']
        del data['LW_net']

    if "ur" in varis_to_load:
        # Wind u and v components in the original data are grid-related.
        # Therefore, we rotate here the wind components from grid- to earth-related coordinates.
        data['u'], data['v'] = Mmt.Rotate_uv_components(data['ur'], data['vr'],idxx, idyy, static_fields["lon"],2,model)

        # Calculating wind direction
        data['WD'] = (np.rad2deg(np.arctan2(-data['u'], -data['v']))+360.) % 360.

        # Calculating wind speed
        data['WS'] = np.sqrt((data['u']**2.) + (data['v']**2.))
        
        del data['ur']
        del data['vr']

    if (("T" in varis_to_load) & ("p_surf" in varis_to_load)):
        # Calculating potential temperature
        data['T_pot'] = ((data['T'])*((1.e5/(data['p_surf']))**(287./1005.))) - 273.15

    # Converting units
    if "p_surf" in data.keys():
        data['p_surf'] /= 100.
    if "MSLP" in data.keys():
        data['MSLP'] /= 100.
    if "T" in data.keys():
        data['T'] -= 273.15
    if "T_surf" in data.keys():
        data['T_surf'] -= 273.15
    if "RH" in data.keys():
        data['RH'] *= 100.
        data['q'] *= 1000.
    if "cloud_cover" in data.keys():
        data['cloud_cover'] *= 100.


    time = np.array(time)
    
    d_dict = {
        "coords": {
            "time": {"dims": "time", "data": time, "attrs": {"units": "seconds since 1970-01-01 00:00:00"}},
            "lat": {"dims": ["x", "y"], "data": model_lat, "attrs": {"units": "degN", "long_name": "latitude"}},
            "lon": {"dims": ["x", "y"], "data": model_lon, "attrs": {"units": "degE", "long_name": "longitude"}},
            "lsm": {"dims": ["x", "y"], "data": model_lsm, "attrs": {"units": "1", "long_name": "land_sea_mask"}},
            "orog": {"dims": ["x", "y"], "data": model_orog, "attrs": {"units": "m", "long_name": "orography"}},
            },
        "attrs": {"Comments": f"Near-surface fields extracted from {full_model_name} simulations."},
        "dims": ["x", "y", "time"],
        "data_vars": {
            vari: {"dims": ["x", "y", "time"],
                   "data": d,
                   "attrs": {"units": variables_names_units[vari][1],
                             "long_name": variables_names_units[vari][0]}} for vari, d in data.items()}}
    
    ds = xr.Dataset.from_dict(d_dict)
    
    ds.to_netcdf(out_path)
    
    return
    
    
def download_3D_data_fields(start_time, end_time, lon_lims, lat_lims, model_levels, start_h, num_h, out_path, static_file, vari_config_file, model="AA", int_x=1, int_y=1):
    
    int_h, int_f, variables_names_units = set_parameters()
    
    time_vec, time_ind, fileurls, varis, static_fields, full_model_name = \
        get_download_config(start_time, end_time, start_h, num_h, int_f, int_h, static_file, vari_config_file, model)
        
    if "Humidity" in varis:
        varis.remove("Humidity")
        varis += ["q"]
        
    model_varnames =    {'T': 'air_temperature_ml',
                         'q': 'specific_humidity_ml',
                         'ur': 'x_wind_ml',
                         'vr': 'y_wind_ml',
                         'p_surf': 'surface_air_pressure',
                         'T_surf': 'air_temperature_0m'}
    
    varis_to_load = {}
    for v in varis:
        try:
            varis_to_load[v] = model_varnames[v]
        except KeyError:
            print(f"The variable {v} is unfortunately not available as nearsurface time series, please de-select in the configuration file and try again.")
            sys.exit(1)
    if "p_surf" not in varis_to_load.keys():
        varis_to_load["p_surf"] = model_varnames["p_surf"]
    if "T_surf" not in varis_to_load.keys():
        varis_to_load["T_surf"] = model_varnames["T_surf"]
    if "T" not in varis_to_load.keys():
        varis_to_load["T"] = model_varnames["T"]
        

    start_lonlat, count_lonlat, _, _ = Mmt.lonlat2xy(lon_lims, lat_lims, static_fields["lon"], static_fields["lat"], 2)
    
    idx = np.arange(start_lonlat[0], (start_lonlat[0]+count_lonlat[0]+1))
    idy = np.arange(start_lonlat[1], (start_lonlat[1]+count_lonlat[1]+1))
    idxx = idx[::int_x]
    idyy = idy[::int_y]
    
    model_lon = static_fields['lon'][idxx,:][:,idyy]
    model_lat = static_fields['lat'][idxx,:][:,idyy]
    model_orog = static_fields['orog'][idxx,:][:,idyy]
    model_lsm = static_fields['lsm'][idxx,:][:,idyy]

    time = []
    data = {}
    
    # determine overall size of time dimension
    len_time = 0                                          # overall time index
    for filename in fileurls:                 # loop over files --> time in the order of days
        for a in range(num_h):                     # loop over timesteps --> time in the order of hours
            len_time += 1
            
    for vari in varis_to_load.keys():
        if vari in ["p_surf", "T_surf"]:
            data[vari] = np.zeros((model_lon.shape[0], model_lon.shape[1], len_time))
        else:
            data[vari] = np.zeros((model_lon.shape[0], model_lon.shape[1], model_levels, len_time))
    data["p"] = np.zeros((model_lon.shape[0], model_lon.shape[1], model_levels, len_time))
    data["z"] = np.zeros((model_lon.shape[0], model_lon.shape[1], model_levels, len_time))
    
    with Dataset(fileurls[0], 'r') as f:
        hybrid = f.variables['hybrid'][-model_levels:]
        ap = f.variables['ap'][-model_levels:]
        b = f.variables['b'][-model_levels:]

    nn = 0                                          # overall time index
    for filename in fileurls:                 # loop over files --> time in the order of days
        with Dataset(filename, 'r') as f:
            for qrr in time_ind:                      # loop over timesteps --> time in the order of hours
                time.append(f.variables['time'][qrr])

                # Retrieving variables from MET Norway server thredds.met.no
                for vari, vari_met in varis_to_load.items():
                    if vari in ["p_surf", "T_surf"]:
                        data[vari][:,:,nn] = np.transpose(np.squeeze(f.variables[vari_met][qrr,0,idyy,idxx]))
                    else:
                        data[vari][:,:,:,nn] = np.transpose(np.squeeze(f.variables[vari_met][qrr,-model_levels:,idyy,idxx]), (2,1,0))

                    print(f'Done reading variable {vari_met} from file {filename} on thredds server')
                    
                
                    
                data['z'][:,:,:,nn], data['p'][:,:,:,nn] = Mmt.Calculate_height_levels_and_pressure(hybrid,ap,b, data['T_surf'][:,:,nn], data['p_surf'][:,:,nn], data['T'][:,:,:,nn],3)

                nn += 1
                
    if "ur" in varis_to_load:
        # Wind u and v components in the original data are grid-related.
        # Therefore, we rotate here the wind components from grid- to earth-related coordinates.
        data['u'], data['v'] = Mmt.Rotate_uv_components(data['ur'], data['vr'],idxx, idyy, static_fields["lon"],2,model)

        # Calculating wind direction
        data['WD'] = (np.rad2deg(np.arctan2(-data['u'], -data['v']))+360.) % 360.

        # Calculating wind speed
        data['WS'] = np.sqrt((data['u']**2.) + (data['v']**2.))
        
        del data['ur']
        del data['vr']
                
                
    # Calculating potential temperature
    data['T_pot'] = ((data['T'])*((1.e5/(data['p']))**(287./1005.))) - 273.15
    if ("q" in varis_to_load):
        T = data["T"] - 273.15
        e = (data['q']*data['p'])/(0.622 + 0.378*data['q'])
        data['RH'] = (e/(611.2 * np.exp((17.62*T)/(243.12+T)))) * 100.
        data['q'] *= 1000.
        
    # Converting units
    data['p'] /= 100.
    data['T'] -= 273.15
    
    del data["p_surf"]
    del data["T_surf"]
    
    
    time = np.array(time)
    
    d_dict = {
        "coords": {
            "time": {"dims": "time", "data": time, "attrs": {"units": "seconds since 1970-01-01 00:00:00"}},
            "ml": {"dims": "ml", "data": np.arange(model_levels,0,-1), "attrs": {"units": "1", "long_name": "model_level"}},
            "lat": {"dims": ["x", "y"], "data": model_lat, "attrs": {"units": "degN", "long_name": "latitude"}},
            "lon": {"dims": ["x", "y"], "data": model_lon, "attrs": {"units": "degE", "long_name": "longitude"}},
            "lsm": {"dims": ["x", "y"], "data": model_lsm, "attrs": {"units": "1", "long_name": "land_sea_mask"}},
            "orog": {"dims": ["x", "y"], "data": model_orog, "attrs": {"units": "m", "long_name": "orography"}},
            },
        "attrs": {"Comments": f"3D data fields extracted from {full_model_name} simulations."},
        "dims": ["x", "y", "ml", "time"],
        "data_vars": {
            vari: {"dims": ["x", "y", "ml", "time"],
                   "data": d,
                   "attrs": {"units": variables_names_units[vari][1],
                             "long_name": variables_names_units[vari][0]}} for vari, d in data.items()}}
    
    ds = xr.Dataset.from_dict(d_dict)
    
    ds.to_netcdf(out_path)
    
    return
    
    
def download_profiles(start_time, end_time, stt_name, stt_lon, stt_lat, model_levels, start_h, num_h, out_path, static_file, vari_config_file, model="AA"):

    int_h, int_f, variables_names_units = set_parameters()
    
    time_vec, time_ind, fileurls, varis, static_fields, full_model_name = \
        get_download_config(start_time, end_time, start_h, num_h, int_f, int_h, static_file, vari_config_file, model)
    
    if "Humidity" in varis:
        varis.remove("Humidity")
        varis += ["q"]
        
    model_varnames =    {'T': 'air_temperature_ml',
                         'q': 'specific_humidity_ml',
                         'ur': 'x_wind_ml',
                         'vr': 'y_wind_ml',
                         'p_surf': 'surface_air_pressure',
                         'T_surf': 'air_temperature_0m'}
    
    varis_to_load = {}
    for v in varis:
        try:
            varis_to_load[v] = model_varnames[v]
        except KeyError:
            print(f"The variable {v} is unfortunately not available as nearsurface time series, please de-select in the configuration file and try again.")
            sys.exit(1)     
    if "p_surf" not in varis_to_load.keys():
        varis_to_load["p_surf"] = model_varnames["p_surf"]
    if "T_surf" not in varis_to_load.keys():
        varis_to_load["T_surf"] = model_varnames["T_surf"]
    if "T" not in varis_to_load.keys():
        varis_to_load["T"] = model_varnames["T"]
        
    # Finding nearest model grid points (x,y) to the longitude and latitude coordinates of the stations in the station list
    coords_xx = np.zeros_like(stt_lon, dtype=int)
    coords_yy = np.zeros_like(stt_lon, dtype=int)
    model_lon = np.zeros_like(stt_lon)
    model_lat = np.zeros_like(stt_lon)
    model_orog = np.zeros_like(stt_lon)
    model_lsm = np.zeros_like(stt_lon)
    for i, (stat, lon, lat) in enumerate(zip(stt_name, stt_lon, stt_lat)):
        coords_xx[i], coords_yy[i], model_lon[i], model_lat[i] = Mmt.lonlat2xy(lon, lat, static_fields['lon'], static_fields['lat'], 1)
        
        model_orog[i] = static_fields['orog'][coords_xx[i], coords_yy[i]]
        model_lsm[i] = static_fields['lsm'][coords_xx[i], coords_yy[i]]

    time = []
    data = {}
    
    # determine overall size of time dimension
    len_time = 0                                          # overall time index
    for filename in fileurls:                 # loop over files --> time in the order of days
        for a in range(num_h):                     # loop over timesteps --> time in the order of hours
            len_time += 1
            
            
    for qr, stat in enumerate(stt_name):           # loop over stations
        data[stat] = {}

        for vari in varis_to_load.keys():
            if vari in ["p_surf", "T_surf"]:
                data[stat][vari] = np.zeros((len_time))
            else:
                data[stat][vari] = np.zeros((model_levels, len_time))
        data[stat]["p"] = np.zeros((model_levels, len_time))
        data[stat]["z"] = np.zeros((model_levels, len_time))
        
        with Dataset(fileurls[0], 'r') as f:
            hybrid = f.variables['hybrid'][-model_levels:]
            ap = f.variables['ap'][-model_levels:]
            b = f.variables['b'][-model_levels:]

        nn = 0                                       # overall time index
        for filename in fileurls:                    # loop over files --> time in the order of days
            with Dataset(filename, 'r') as f:
                for qrr in time_ind:                      # loop over timesteps --> time in the order of hours
                    if qr == 0:
                        time.append(f.variables['time'][qrr])

                    # Retrieving variables from MET Norway server thredds.met.no
                    for vari, vari_met in varis_to_load.items():
                        if vari in ["p_surf", "T_surf"]:
                            data[stat][vari][nn] = np.squeeze(f.variables[vari_met][qrr,0,coords_yy[qr],coords_xx[qr]])
                        else:
                            data[stat][vari][:,nn] = np.squeeze(f.variables[vari_met][qrr,-model_levels:,coords_yy[qr],coords_xx[qr]])

                        print(f'Done reading variable {vari_met} from file {filename} on thredds server')
                        
                        
                    data[stat]['z'][:,nn], data[stat]['p'][:,nn] = Mmt.Calculate_height_levels_and_pressure(hybrid,ap,b, data[stat]['T_surf'][nn], data[stat]['p_surf'][nn], data[stat]['T'][:,nn],1)

                    nn += 1

        if "ur" in varis_to_load:
            # Wind u and v components in the original data are grid-related.
            # Therefore, we rotate here the wind components from grid- to earth-related coordinates.
            data[stat]['u'], data[stat]['v'] = Mmt.Rotate_uv_components(data[stat]['ur'], data[stat]['vr'],coords_xx[qr], coords_yy[qr], static_fields["lon"],1,model)
    
            # Calculating wind direction
            data[stat]['WD'] = (np.rad2deg(np.arctan2(-data[stat]['u'], -data[stat]['v']))+360.) % 360.
    
            # Calculating wind speed
            data[stat]['WS'] = np.sqrt((data[stat]['u']**2.) + (data[stat]['v']**2.))
            
            del data[stat]['ur']
            del data[stat]['vr']


        # Calculating potential temperature
        data[stat]['T_pot'] = ((data[stat]['T'])*((1.e5/(data[stat]['p']))**(287./1005.))) - 273.15
        if ("q" in varis_to_load):
            T = data[stat]["T"] - 273.15
            e = (data[stat]['q']*data[stat]['p'])/(0.622 + 0.378*data[stat]['q'])
            data[stat]['RH'] = (e/(611.2 * np.exp((17.62*T)/(243.12+T)))) * 100.
            data[stat]['q'] *= 1000.
            
        # Converting units
        data[stat]['p'] /= 100.
        data[stat]['T'] -= 273.15
        
        del data[stat]["p_surf"]
        del data[stat]["T_surf"]
        

    time = np.array(time)
    
    
    data_reorg = {vari: np.array([data[s][vari] for s in stt_name]) for vari in data[stt_name[0]].keys()}
    
    d_dict = {
        "coords": {
            "time": {"dims": "time", "data": time, "attrs": {"units": "seconds since 1970-01-01 00:00:00"}},
            "ml": {"dims": "ml", "data": np.arange(model_levels,0,-1), "attrs": {"units": "1", "long_name": "model_level"}},
            "station_name": {"dims": "station", "data": stt_name, "attrs": {"units": "1"}},
            "station_lat": {"dims": "station", "data": stt_lat, "attrs": {"units": "degN", "long_name": "latitude_of_station"}},
            "station_lon": {"dims": "station", "data": stt_lon, "attrs": {"units": "degE", "long_name": "longitude_of_station"}},
            "model_lat": {"dims": "station", "data": model_lat, "attrs": {"units": "degN", "long_name": "latitude_of_model_gridpoint"}},
            "model_lon": {"dims": "station", "data": model_lon, "attrs": {"units": "degE", "long_name": "longitude_of_model_gridpoint"}},
            "model_lsm": {"dims": "station", "data": model_lsm, "attrs": {"units": "1", "long_name": "land_sea_mask_of_model_gridpoint"}},
            "model_orog": {"dims": "station", "data": model_orog, "attrs": {"units": "m", "long_name": "elevation_of_model_gridpoint"}},
            },
        "attrs": {"Comments": f"Profile data extracted from {full_model_name} simulations."},
        "dims": ["station", "time", "ml"],
        "data_vars": {
            vari: {"dims": ["station", "ml", "time"],
                    "data": d,
                    "attrs": {"units": variables_names_units[vari][1],
                              "long_name": variables_names_units[vari][0]}} for vari, d in data_reorg.items()}}
    
    ds = xr.Dataset.from_dict(d_dict)
    ds = ds.set_index({"station": "station_name"}, append=True)
    
    ds.to_netcdf(out_path)
    
    return






if __name__ == "__main__":
    
    download_3D_data_fields("2022-05-05 00:00:00", "2022-05-06 00:00:00", 
                                    [10., 15.], [78., 80.], 10, 0, 3, 
                                    "C:/Users/lukasf/Desktop/test.nc",
                                    "C:/Users/lukasf/Desktop/AA_static_fields.nc", 
                                    "C:/Users/lukasf/Github/unisacsi/unisacsi/config_model_variables.yml", 
                                    model="AA", int_x=2, int_y=2)
