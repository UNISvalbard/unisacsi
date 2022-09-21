# -*- coding: utf-8 -*-
"""
This module contains scripts to download data from the weather models
of the Norwegian Meteorological Institute, AROME-Arctic and MetCoOp.
The code is optimized for the use in the UNIS courses.
"""

import unisacsi
from netCDF4 import Dataset
import numpy as np
import pandas as pd
import yaml
import sys
import copy
import xarray as xr
import matplotlib as mpl
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
import utm
from scipy import interpolate


def download_MET_model_data(config_file):
    """
    Function to download and save model data
    of one of the models of the Norwegian Meteorological Institute.
    The download configuration (period to download, variables, type of data, ...)
    is done in a config file.

    Parameters
    ----------
    config_file : str
        Full path to the yaml file with the configuration settings

    Returns
    -------
    None.

    """

    with open(config_file, "r") as f:
        config_settings = yaml.safe_load(f)
        
    if config_settings["model"] == "MC":
        config_settings["resolution"] = "2p5km"

    if config_settings["save_daily_files"]:
        days = pd.date_range(config_settings["start_day"], config_settings["end_day"], freq="1D")
        daily_config = copy.copy(config_settings)
        for d in days:
            daily_config["start_day"] = d.strftime("%Y-%m-%d")
            daily_config["end_day"] = (d+pd.Timedelta(days=1)).strftime("%Y-%m-%d")
            path = config_settings["out_path"].split("/")[:-1]
            filename = config_settings["out_path"].split("/")[-1]
            daily_config["out_path"] = f"{'/'.join(path)}/{filename}_{config_settings['resolution']}_{d.strftime('%Y%m%d')}.nc"
            print("############################################################")
            print(f"start downloading data from {d.strftime('%Y-%m-%d')}")
            print("############################################################")
            MET_model_download_class(daily_config)
    else:
        config_settings["out_path"] = f"{config_settings['out_path']}_{config_settings['resolution']}.nc"
        MET_model_download_class(config_settings)


    return



def download_MET_model_static_fields(config_file):
    """
    Function to download and save the static fields (lon, lat, orog and lsm)
    of one of the models of the Norwegian Meteorological Institute.

    Parameters
    ----------
    config_file : str
        Full path to the yaml file with the configuration settings (same as for the data download)

    Returns
    -------
    None.

    """

    with open(config_file, "r") as f:
        config_settings = yaml.safe_load(f)

    model = config_settings["model"]
    resolution = config_settings["resolution"]

    if model == "AA":
        if resolution == "2p5km":
            file = 'https://thredds.met.no/thredds/dodsC/aromearcticarchive/2022/06/03/arome_arctic_det_2_5km_20220603T00Z.nc'
        elif resolution == "500m":
            file = 'https://thredds.met.no/thredds/dodsC/metusers/yuriib/N-FORCES/AS500_2022090200.nc'
    elif model == "MC":
        file = 'https://thredds.met.no/thredds/dodsC/meps25epsarchive/2022/02/20/meps_det_2_5km_20220220T00Z.nc'
        resolution = "2p5km"

    path = config_settings["static_file"].split("/")[:-1]
    filename = config_settings["static_file"].split("/")[-1]
    out_path = f"{'/'.join(path)}/{filename}_{resolution}.nc"
    
    with xr.open_dataset(file) as static_fields:
        static_fields.isel(time=1)[["x", "y", "longitude", "latitude", "projection_lambert", "surface_geopotential", "land_area_fraction"]].squeeze().to_netcdf(out_path)


    print(f"Static fields were successfully saved into {out_path}.")

    return



class MET_model_download_class():
    """
    Class handling the download of MET model data.

    Parameters
    ----------
    config_settings : dict
        Download configuration settings read from the configuration file.

"""

    def __init__(self, config_settings):

        self.start_time = config_settings["start_day"]
        self.end_time = config_settings["end_day"]
        self.int_h = config_settings["int_h"]
        self.int_f = config_settings["int_f"]
        self.start_h = config_settings["start_h"]
        self.num_h = config_settings["num_h"]
        self.model = config_settings["model"]
        self.check_plot = config_settings["check_plot"]
        self.out_path = config_settings["out_path"]
        self.stt_name = list(config_settings["stations"].keys())
        self.stt_lon = [config_settings["stations"][s]["lon"] for s in self.stt_name]
        self.stt_lat = [config_settings["stations"][s]["lat"] for s in self.stt_name]
        self.lon_lims = [config_settings["area"]["lon_min"], config_settings["area"]["lon_max"]]
        self.lat_lims = [config_settings["area"]["lat_min"], config_settings["area"]["lat_max"]]
        self.int_x = config_settings["int_x"]
        self.int_y = config_settings["int_y"]
        self.resolution = config_settings["resolution"]
        self.start_point = [config_settings["crosssection"]["start_lat"], config_settings["crosssection"]["start_lon"]]
        self.end_point = [config_settings["crosssection"]["end_lat"], config_settings["crosssection"]["end_lon"]]
        self.model_levels = config_settings["model_levels"]
        self.p_levels = config_settings["pressure_levels"]

        if self.model == "AA":
            if self.resolution == "2p5km":
                self.time_vec = pd.date_range(self.start_time, self.end_time, freq=f"{self.int_f}H", closed="left")
            elif self.resolution == "500m":
                self.time_vec = pd.date_range(self.start_time, self.end_time, freq="1D", closed="left")
            else:
                assert False, "Resolution not valid, specify either '2p5km' or '500m'."
        elif self.model == "MC":
            self.time_vec = pd.date_range(self.start_time, self.end_time, freq=f"{self.int_f}H", closed="left")
        else:
            assert False, "Model name not recognized, specify either 'AA' or 'MC'."
        self.time_ind = np.arange(self.start_h, self.start_h+self.num_h, self.int_h, dtype=int)


        path = config_settings["static_file"].split("/")[:-1]
        filename = config_settings["static_file"].split("/")[-1]
        self.static_file = f"{'/'.join(path)}/{filename}_{self.resolution}.nc"
        self.static_fields = {}
        with Dataset(self.static_file, 'r') as f:
            self.static_fields["x"] = f.variables["x"][:]
            self.static_fields["y"] = f.variables["y"][:]
            self.static_fields["lon"] = f.variables["longitude"][:]
            self.static_fields["lat"] = f.variables["latitude"][:]
            self.static_fields["orog"] = f.variables["surface_geopotential"][:] / 9.81
            self.static_fields["lsm"] = f.variables["land_area_fraction"][:]


        self.fileurls = []
        if self.model == "AA":
            self.full_model_name = "AROME_Arctic"
            if self.resolution == "2p5km":
                for t in self.time_vec:
                    if t < pd.Timestamp("2022-02-01"):
                        if config_settings["data_format"] == 5:
                            self.fileurls.append(f'https://thredds.met.no/thredds/dodsC/aromearcticarchive/{t.strftime("%Y/%m/%d")}/arome_arctic_extracted_2_5km_{t.strftime("%Y%m%d")}T{t.strftime("%H")}Z.nc')
                        else:
                            self.fileurls.append(f'https://thredds.met.no/thredds/dodsC/aromearcticarchive/{t.strftime("%Y/%m/%d")}/arome_arctic_full_2_5km_{t.strftime("%Y%m%d")}T{t.strftime("%H")}Z.nc')
                    else:
                        self.fileurls.append(f'https://thredds.met.no/thredds/dodsC/aromearcticarchive/{t.strftime("%Y/%m/%d")}/arome_arctic_det_2_5km_{t.strftime("%Y%m%d")}T{t.strftime("%H")}Z.nc')
            elif self.resolution == "500m":
                for t in self.time_vec:
                    self.fileurls.append(f'https://thredds.met.no/thredds/dodsC/metusers/yuriib/N-FORCES/AS500_{t.strftime("%Y%m%d")}00.nc')
            else:
                assert False, "Resolution not valid, specify either '2p5km' or '500m'."
        elif self.model == "MC":
            self.full_model_name= "METCoOp"
            for t in self.time_vec:
                self.fileurls.append(f'https://thredds.met.no/thredds/dodsC/meps25epsarchive/{t.strftime("%Y/%m/%d")}/meps_det_2_5km_{t.strftime("%Y%m%d")}T{t.strftime("%H")}Z.nc')
        else:
            assert False, "Model name not recognized, specify either 'AA' or 'MC'."




        self.varis = [k for k, s in config_settings["variables"].items() if s == 1]
        if "wind" in self.varis:
            self.varis.remove("wind")
            self.varis += ["ur", "vr"]
        if "momentum_flux" in self.varis:
            self.varis.remove("momentum_flux")
            self.varis += ["tau_x", "tau_y"]
        if "radiation" in self.varis:
            self.varis.remove("radiation")
            self.varis += ['SW_net', 'SW_down', 'LW_net', 'LW_down']
        if "turb_fluxes" in self.varis:
            self.varis.remove("turb_fluxes")
            self.varis += ["LHF", "SHF"]

        self.variables_names_units =    {'T': ['temperature', "degC"],
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
                                        'precip': ["precipitation", "mm"],
                                        "p": ["pressure", "hPa"],
                                        "z": ["height", "m"],
                                        "z_asl": ["height_above_sea_level", "m"],
                                        'time': ["time", "seconds since 1970-01-01 00:00:00"]}


        if "data_format" in config_settings.keys():
            if config_settings["data_format"] == 0:
                self.download_profiles()
            elif config_settings["data_format"] == 1:
                self.download_nearsurface_timeseries()
            elif config_settings["data_format"] == 2:
                self.download_nearsurface_fields()
            elif config_settings["data_format"] == 3:
                self.new()
                # self.download_3D_data_fields()
            elif config_settings["data_format"] == 4:
                self.download_cross_section()
            elif config_settings["data_format"] == 5:
                self.download_pressure_levels()
            else:
                assert False, "Data format not a valid option, please change in the config file."
        else:
            assert False, "Please specify the type of data you want to download."






    def download_nearsurface_timeseries(self):
        """
        Method to download time series of near-surface data from one or several locations.

        Returns
        -------
        None.

        """


        if "Humidity" in self.varis:
            self.varis.remove("Humidity")
            self.varis += ["RH", "q"]

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
                            'ABL_height': 'atmosphere_boundary_layer_thickness',
                            'precip': "precipitation_amount_acc"}

        varis_to_load = {}
        for v in self.varis:
            try:
                varis_to_load[v] = model_varnames[v]
            except KeyError:
                print(f"The variable {v} is unfortunately not available as nearsurface time series, please de-select in the configuration file and try again.")
                sys.exit(1)

        # Finding nearest model grid points (x,y) to the longitude and latitude coordinates of the stations in the station list
        coords_xx = np.zeros_like(self.stt_lon, dtype=int)
        coords_yy = np.zeros_like(self.stt_lon, dtype=int)
        model_lon = np.zeros_like(self.stt_lon)
        model_lat = np.zeros_like(self.stt_lon)
        model_orog = np.zeros_like(self.stt_lon)
        model_lsm = np.zeros_like(self.stt_lon)
        for i, (stat, lon, lat) in enumerate(zip(self.stt_name, self.stt_lon, self.stt_lat)):
            coords_xx[i], coords_yy[i], model_lon[i], model_lat[i] = unisacsi.lonlat2xy(lon, lat, self.static_fields['lon'], self.static_fields['lat'], 1)

            model_orog[i] = self.static_fields['orog'][coords_xx[i], coords_yy[i]]
            model_lsm[i] = self.static_fields['lsm'][coords_xx[i], coords_yy[i]]


        if self.check_plot:
            # figure to check if the right area was selected
            lon_min = np.nanmin([model_lon[i] for i in range(len(self.stt_name))])-1.
            lon_max = np.nanmax([model_lon[i] for i in range(len(self.stt_name))])+1.
            lat_min = np.nanmin([model_lat[i] for i in range(len(self.stt_name))])-0.5
            lat_max = np.nanmax([model_lat[i] for i in range(len(self.stt_name))])+0.5
            plt.close("all")
            plt.ion()
            fig, ax = plt.subplots(1,1,subplot_kw={'projection': ccrs.Mercator()})
            ax.set_extent([lon_min, lon_max, lat_min, lat_max], crs=ccrs.PlateCarree())
            ax.coastlines(resolution="10m", linewidth=0.5, zorder=30)
            gl = ax.gridlines(draw_labels=False)
            gl.left_labels = True
            gl.bottom_labels = True
            gl.xformatter = LONGITUDE_FORMATTER
            gl.yformatter = LATITUDE_FORMATTER
            ax.set_title(None)
            ax.set_xlabel(None)
            ax.set_ylabel(None)
            ax.set_title(None)
            for i, (lon, lat) in enumerate(zip(self.stt_lon, self.stt_lat)):
                ax.scatter(lon, lat, s=10., c="r", marker="x", transform=ccrs.PlateCarree())
                ax.scatter(model_lon[i], model_lat[i], s=10., c="k", marker="o", transform=ccrs.PlateCarree())
            plt.show(block=False)
            plt.pause(5.)

            print("Type 'Y' to proceed to the download. Any other input will terminate the script immediately.")
            print("")
            switch_continue = input("Your input:  ")

            plt.close("all")

            if switch_continue != "Y":
                sys.exit(1)

        time = []
        data = {}

        # determine overall size of time dimension
        len_time = 0                                          # overall time index
        for filename in self.fileurls:                 # loop over files --> time in the order of days
            for a in range(self.num_h):                     # loop over timesteps --> time in the order of hours
                len_time += 1

        for qr, stat in enumerate(self.stt_name):           # loop over stations
            data[stat] = {}

            for vari in varis_to_load.keys():
                data[stat][vari] = np.zeros((len_time))

            nn = 0                                          # overall time index
            for filename in self.fileurls:                 # loop over files --> time in the order of days

                with Dataset(filename, 'r') as f:
                    for qrr in self.time_ind:                      # loop over timesteps --> time in the order of hours
                        if qr == 0:
                            time.append(f.variables['time'][qrr])

                        # Retrieving variables from MET Norway server thredds.met.no
                        for vari, vari_met in varis_to_load.items():
                            data[stat][vari][nn] = np.squeeze(f.variables[vari_met][qrr,0,coords_yy[qr],coords_xx[qr]])

                            print(f'Done reading variable {vari_met} from file {filename} on thredds server')

                        nn += 1

            for vari, vari_met in varis_to_load.items():
                if vari_met[:8] == "integral":
                    data[stat][vari][1:] -= np.diff(data[stat][vari])
                    data[stat][vari][0] /= self.start_h
                    data[stat][vari] /= (3600.*self.int_h)
                elif vari_met[-3:] == "acc":
                    data[stat][vari][1:] -= np.diff(data[stat][vari])
                    data[stat][vari][0] /= self.start_h

            if "SW_net" in varis_to_load:
                # Converting radiative fluxes from net into up
                data[stat]["SW_up"] = data[stat]["SW_down"] - data[stat]["SW_net"]
                data[stat]["LW_up"] = data[stat]["LW_down"] - data[stat]["LW_net"]

                del data[stat]['SW_net']
                del data[stat]['LW_net']

            if "ur" in varis_to_load:
                # Wind u and v components in the original data are grid-related.
                # Therefore, we rotate here the wind components from grid- to earth-related coordinates.
                data[stat]['u'], data[stat]['v'] = unisacsi.Rotate_uv_components(data[stat]['ur'], data[stat]['vr'],coords_xx[qr], coords_yy[qr], self.static_fields["lon"],1,self.model)

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

        data_reorg = {vari: np.array([data[s][vari] for s in self.stt_name]) for vari in data[self.stt_name[0]].keys()}

        d_dict = {
            "coords": {
                "time": {"dims": "time", "data": time, "attrs": {"units": "seconds since 1970-01-01 00:00:00"}},
                "station_name": {"dims": "station", "data": self.stt_name, "attrs": {"units": "1"}},
                "station_lat": {"dims": "station", "data": self.stt_lat, "attrs": {"units": "degN", "long_name": "latitude_of_station"}},
                "station_lon": {"dims": "station", "data": self.stt_lon, "attrs": {"units": "degE", "long_name": "longitude_of_station"}},
                "model_lat": {"dims": "station", "data": model_lat, "attrs": {"units": "degN", "long_name": "latitude_of_model_gridpoint"}},
                "model_lon": {"dims": "station", "data": model_lon, "attrs": {"units": "degE", "long_name": "longitude_of_model_gridpoint"}},
                "model_lsm": {"dims": "station", "data": model_lsm, "attrs": {"units": "1", "long_name": "land_sea_mask_of_model_gridpoint"}},
                "model_orog": {"dims": "station", "data": model_orog, "attrs": {"units": "m", "long_name": "elevation_of_model_gridpoint"}},
                "model_x": {"dims": "station", "data": coords_xx, "attrs": {"units": "m", "long_name": "x-coordinate in model Cartesian system"}},
                "model_y": {"dims": "station", "data": coords_yy, "attrs": {"units": "m", "long_name": "y-coordinate in model Cartesian system"}},
                },
            "attrs": {"Comments": f"Near-surface time series data extracted from {self.full_model_name} simulations.",
                      "Conventions": "CF-1.6, ACDD"},
            "dims": ["station", "time"],
            "data_vars": {
                vari: {"dims": ["station", "time"],
                       "data": d,
                       "attrs": {"units": self.variables_names_units[vari][1],
                                 "long_name": self.variables_names_units[vari][0]}} for vari, d in data_reorg.items()}}

        ds = xr.Dataset.from_dict(d_dict)
        ds = ds.set_index({"station": "station_name"}, append=True)
        
        ds.rio.write_crs("+proj=lcc +lat_0=77.5 +lon_0=-25 +lat_1=77.5 +lat_2=77.5 +no_defs +R=6.371e+06", inplace=True)

        ds = xr.decode_cf(ds)        

        ds.to_netcdf(self.out_path)

        return




    def download_nearsurface_fields(self):
        """
        Method to download 2D fields of near-surface variables.

        Returns
        -------
        None.

        """


        if "Humidity" in self.varis:
            self.varis.remove("Humidity")
            self.varis += ["RH", "q"]

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
                             'ABL_height': 'atmosphere_boundary_layer_thickness',
                             'precip': "precipitation_amount_acc"}

        varis_to_load = {}
        for v in self.varis:
            try:
                varis_to_load[v] = model_varnames[v]
            except KeyError:
                print(f"The variable {v} is unfortunately not available as nearsurface time series, please de-select in the configuration file and try again.")
                sys.exit(1)


        start_lonlat, count_lonlat, _, _ = unisacsi.lonlat2xy(self.lon_lims, self.lat_lims, self.static_fields["lon"], self.static_fields["lat"], 2)

        idx = np.arange(start_lonlat[0], (start_lonlat[0]+count_lonlat[0]+1))
        idy = np.arange(start_lonlat[1], (start_lonlat[1]+count_lonlat[1]+1))
        idxx = idx[::self.int_x]
        idyy = idy[::self.int_y]

        model_lon = self.static_fields['lon'][idxx,:][:,idyy]
        model_lat = self.static_fields['lat'][idxx,:][:,idyy]
        model_orog = self.static_fields['orog'][idxx,:][:,idyy]
        model_lsm = self.static_fields['lsm'][idxx,:][:,idyy]
        model_x = self.static_fields['x'][idxx]
        model_y = self.static_fields['y'][idyy]

        if self.check_plot:
            # figure to check if the right area was selected
            plt.close("all")
            plt.ion()
            fig, ax = plt.subplots(1,1,subplot_kw={'projection': ccrs.Mercator()})
            ax.set_extent([self.lon_lims[0]-2., self.lon_lims[1]+2., self.lat_lims[0]-.5, self.lat_lims[1]+.5], crs=ccrs.PlateCarree())
            ax.coastlines(resolution="10m", linewidth=0.5, zorder=30)
            gl = ax.gridlines(draw_labels=False)
            gl.left_labels = True
            gl.bottom_labels = True
            gl.xformatter = LONGITUDE_FORMATTER
            gl.yformatter = LATITUDE_FORMATTER
            ax.set_title(None)
            ax.set_xlabel(None)
            ax.set_ylabel(None)
            ax.set_title(None)
            ax.pcolormesh(model_lon, model_lat, model_orog, cmap=mpl.cm.terrain, shading="nearest", transform=ccrs.PlateCarree())
            plt.show(block=False)
            plt.pause(5.)

            print("Type 'Y' to proceed to the download. Any other input will terminate the script immediately.")
            print("")
            switch_continue = input("Your input:  ")

            plt.close("all")

            if switch_continue != "Y":
                sys.exit(1)

        time = []
        data = {}

        # determine overall size of time dimension
        len_time = 0                                          # overall time index
        for filename in self.fileurls:                 # loop over files --> time in the order of days
            for a in range(self.num_h):                     # loop over timesteps --> time in the order of hours
                len_time += 1


        for vari in varis_to_load.keys():
            data[vari] = np.zeros((model_lon.shape[0], model_lon.shape[1], len_time))

        nn = 0                                          # overall time index
        for filename in self.fileurls:                 # loop over files --> time in the order of days

            with Dataset(filename, 'r') as f:
                for qrr in self.time_ind:                      # loop over timesteps --> time in the order of hours
                    time.append(f.variables['time'][qrr])

                    # Retrieving variables from MET Norway server thredds.met.no
                    for vari, vari_met in varis_to_load.items():
                        data[vari][:,:,nn] = np.transpose(np.squeeze(f.variables[vari_met][qrr,0,idyy,idxx]))

                        print(f'Done reading variable {vari_met} from file {filename} on thredds server')

                    nn += 1

        for vari, vari_met in varis_to_load.items():
            if vari_met[:8] == "integral":
                data[vari][:,:,1:] -= np.diff(data[vari], axis=-1)
                data[vari][:,:,0] /= self.start_h
                data[vari] /= (3600.*self.int_h)
            elif vari_met[-3:] == "acc":
                data[vari][:,:,1:] -= np.diff(data[vari], axis=-1)
                data[vari][:,:,0] /= self.start_h

        if "SW_net" in varis_to_load:
            # Converting radiative fluxes from net into up
            data["SW_up"] = data["SW_down"] - data["SW_net"]
            data["LW_up"] = data["LW_down"] - data["LW_net"]

            del data['SW_net']
            del data['LW_net']

        if "ur" in varis_to_load:
            # Wind u and v components in the original data are grid-related.
            # Therefore, we rotate here the wind components from grid- to earth-related coordinates.
            data["u"] = np.zeros((model_lon.shape[0], model_lon.shape[1], len_time))
            data["v"] = np.zeros((model_lon.shape[0], model_lon.shape[1], len_time))
            for nn in range(len(time)):
                data['u'][:,:,nn], data['v'][:,:,nn] = unisacsi.Rotate_uv_components(data['ur'][:,:,nn], data['vr'][:,:,nn],idxx, idyy, self.static_fields["lon"],2,self.model)

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
                "x": {"dims": ["x"], "data": model_x, "attrs": {"units": "m", "long_name": "x-coordinate in model Cartesian system"}},
                "y": {"dims": ["y"], "data": model_y, "attrs": {"units": "m", "long_name": "y-coordinate in model Cartesian system"}}
                },
            "attrs": {"Comments": f"Near-surface fields extracted from {self.full_model_name} simulations.",
                      "Conventions": "CF-1.6, ACDD"},
            "dims": ["x", "y", "time"],
            "data_vars": {
                vari: {"dims": ["x", "y", "time"],
                       "data": d,
                       "attrs": {"units": self.variables_names_units[vari][1],
                                 "long_name": self.variables_names_units[vari][0]}} for vari, d in data.items()}}

        ds = xr.Dataset.from_dict(d_dict)

        ds.rio.write_crs("+proj=lcc +lat_0=77.5 +lon_0=-25 +lat_1=77.5 +lat_2=77.5 +no_defs +R=6.371e+06", inplace=True)
        
        ds = xr.decode_cf(ds)

        ds.to_netcdf(self.out_path)

        return




    def download_cross_section(self):
        """
        Method to download a (vertical )cross section.

        Returns
        -------
        None.

        """


        if "Humidity" in self.varis:
            self.varis.remove("Humidity")
            self.varis += ["q"]

        model_varnames =    {'T': 'air_temperature_ml',
                             'q': 'specific_humidity_ml',
                             'ur': 'x_wind_ml',
                             'vr': 'y_wind_ml',
                             'p_surf': 'surface_air_pressure',
                             'T_surf': 'air_temperature_0m'}

        varis_to_load = {}
        for v in self.varis:
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

        lat_lims = [np.min([self.start_point[0], self.end_point[0]]), np.max([self.start_point[0], self.end_point[0]])]
        lon_lims = [np.min([self.start_point[1], self.end_point[1]]), np.max([self.start_point[1], self.end_point[1]])]

        start_lonlat, count_lonlat, _, _ = unisacsi.lonlat2xy(lon_lims, lat_lims, self.static_fields["lon"], self.static_fields["lat"], 2)
        start_lonlat -= 2
        count_lonlat += 5

        idx = np.arange(start_lonlat[0], (start_lonlat[0]+count_lonlat[0]+1))
        idy = np.arange(start_lonlat[1], (start_lonlat[1]+count_lonlat[1]+1))
        idxx = idx[::self.int_x]
        idyy = idy[::self.int_y]

        model_lon = self.static_fields['lon'][idxx,:][:,idyy]
        model_lat = self.static_fields['lat'][idxx,:][:,idyy]
        model_orog = self.static_fields['orog'][idxx,:][:,idyy]
        model_x = self.static_fields['x'][idxx]
        model_y = self.static_fields['y'][idyy]

        if self.check_plot:
            # figure to check if the right area was selected
            plt.close("all")
            plt.ion()
            fig, ax = plt.subplots(1,1,subplot_kw={'projection': ccrs.Mercator()})
            ax.set_extent([lon_lims[0]-2., lon_lims[1]+2., lat_lims[0]-.5, lat_lims[1]+.5], crs=ccrs.PlateCarree())
            ax.coastlines(resolution="10m", linewidth=0.5, zorder=30)
            gl = ax.gridlines(draw_labels=False)
            gl.left_labels = True
            gl.bottom_labels = True
            gl.xformatter = LONGITUDE_FORMATTER
            gl.yformatter = LATITUDE_FORMATTER
            ax.set_title(None)
            ax.set_xlabel(None)
            ax.set_ylabel(None)
            ax.set_title(None)
            ax.pcolormesh(model_lon, model_lat, model_orog, cmap=mpl.cm.terrain, shading="nearest", transform=ccrs.PlateCarree())
            ax.plot([self.start_point[1], self.end_point[1]], [self.start_point[0], self.end_point[0]], 'r-', transform=ccrs.PlateCarree())
            plt.show(block=False)
            plt.pause(3.)

            print("Type 'Y' to proceed to the download. Any other input will terminate the script immediately.")
            print("")
            switch_continue = input("Your input:  ")

            plt.close("all")

            if switch_continue != "Y":
                sys.exit(1)

        time = []
        data = {}

        # determine overall size of time dimension
        len_time = 0                                          # overall time index
        for filename in self.fileurls:                 # loop over files --> time in the order of days
            for a in range(self.num_h):                     # loop over timesteps --> time in the order of hours
                len_time += 1

        for vari in varis_to_load.keys():
            if vari in ["p_surf", "T_surf"]:
                data[vari] = np.zeros((model_lon.shape[0], model_lon.shape[1], len_time))
            else:
                data[vari] = np.zeros((model_lon.shape[0], model_lon.shape[1], self.model_levels, len_time))
        data["p"] = np.zeros((model_lon.shape[0], model_lon.shape[1], self.model_levels, len_time))
        data["z"] = np.zeros((model_lon.shape[0], model_lon.shape[1], self.model_levels, len_time))

        with Dataset(self.fileurls[0], 'r') as f:
            hybrid = f.variables['hybrid'][-self.model_levels:]
            ap = f.variables['ap'][-self.model_levels:]
            b = f.variables['b'][-self.model_levels:]

        nn = 0                                          # overall time index
        for filename in self.fileurls:                 # loop over files --> time in the order of days
            with Dataset(filename, 'r') as f:
                for qrr in self.time_ind:                      # loop over timesteps --> time in the order of hours
                    time.append(f.variables['time'][qrr])

                    # Retrieving variables from MET Norway server thredds.met.no
                    for vari, vari_met in varis_to_load.items():
                        if vari in ["p_surf", "T_surf"]:
                            data[vari][:,:,nn] = np.transpose(np.squeeze(f.variables[vari_met][qrr,0,idyy,idxx]))
                        else:
                            data[vari][:,:,:,nn] = np.transpose(np.squeeze(f.variables[vari_met][qrr,-self.model_levels:,idyy,idxx]), (2,1,0))

                        print(f'Done reading variable {vari_met} from file {filename} on thredds server')



                    data['z'][:,:,:,nn], data['p'][:,:,:,nn] = unisacsi.Calculate_height_levels_and_pressure(hybrid,ap,b, data['T_surf'][:,:,nn], data['p_surf'][:,:,nn], data['T'][:,:,:,nn],3)

                    nn += 1

        if "ur" in varis_to_load:
            # Wind u and v components in the original data are grid-related.
            # Therefore, we rotate here the wind components from grid- to earth-related coordinates.
            data["u"] = np.zeros((model_lon.shape[0], model_lon.shape[1], self.model_levels, len_time))
            data["v"] = np.zeros((model_lon.shape[0], model_lon.shape[1], self.model_levels, len_time))
            for ml in range(self.model_levels):
                for nn in range(len(time)):
                    data['u'][:,:,ml,nn], data['v'][:,:,ml,nn] = unisacsi.Rotate_uv_components(data['ur'][:,:,ml,nn], data['vr'][:,:,ml,nn],idxx, idyy, self.static_fields["lon"],2,self.model)

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


        # Calculating distance between end and start points
        # using that (d2km) to find out how many points to use for the
        # interpolation

        radius = 6371.
        lat1 = self.start_point[0] * (np.pi/180)
        lat2 = self.end_point[0] * (np.pi/180)
        lon1 = self.start_point[1] * (np.pi/180)
        lon2 = self.end_point[1] * (np.pi/180)
        deltaLat = lat2 - lat1
        deltaLon = lon2 - lon1

        x = deltaLon * np.cos((lat1+lat2)/2.)
        y = deltaLat
        d2km = radius * np.sqrt(x**2. + y**2.) # Pythagoran distance

        dd = int(d2km/2.5)

        lons = np.linspace(self.start_point[1], self.end_point[1], dd)
        lats = np.linspace(self.start_point[0], self.end_point[0], dd)

        # infer new x-axis as distances between the new lat-lon-points
        basepoints_x, basepoints_y, _, _ = utm.from_latlon(lats, lons)
        delta_x = basepoints_x - basepoints_x[0]
        delta_y = basepoints_y - basepoints_y[0]
        xx = np.sqrt(delta_x**2. + delta_y**2.) / 1000.

        # Interpolating the surface height on to the cross section
        hgs = interpolate.interp2d(model_lon, model_lat, model_orog, kind='linear')
        hgss = np.diagonal(np.transpose(hgs(lons, lats)))



        data_cross = {"z_asl": np.zeros((len(lons), len(lats), self.model_levels, len_time))}
        for vari, d in data.items():
            data_cross[vari] = np.zeros((len(lons), self.model_levels, len_time))

        for nn in range(len(time)):
            # Interpolating the atmospheric data on to the cross section

            zz = 0
            for i in range(self.model_levels):

                s = interpolate.interp2d(model_lon, model_lat, np.squeeze(data['z'][:,:,i,nn]), kind='linear')
                data_cross["z_asl"][:,:,zz,nn] = s(lons, lats)

                for vari in data.keys():
                    data_cross[vari][:,zz,nn] = np.diagonal(interpolate.griddata((model_lon.flatten(), model_lat.flatten(), np.squeeze(data['z'][:,:,i,nn]).flatten()), np.squeeze(data[vari][:,:,i,nn]).flatten(), (lons, lats, data_cross["z_asl"][:,:,zz,nn]), method="linear"), axis1=0, axis2=1)

                zz = zz+1;

        data_cross["z_asl"] = np.transpose(np.diagonal(data_cross["z_asl"], axis1=0, axis2=1), (2,0,1))

        hg = np.repeat(hgss[:,np.newaxis], data_cross["z_asl"].shape[1], axis=1)
        hg = np.repeat(hg[:,:,np.newaxis], data_cross["z_asl"].shape[2], axis=2)

        data_cross["z"] = data_cross["z_asl"]
        data_cross["z_asl"] = data_cross["z_asl"] + hg

        d_dict = {
            "coords": {
                "time": {"dims": "time", "data": time, "attrs": {"units": "seconds since 1970-01-01 00:00:00"}},
                "ml": {"dims": "ml", "data": np.arange(self.model_levels,0,-1), "attrs": {"units": "1", "long_name": "model_level"}},
                "lat": {"dims": "x", "data": lats, "attrs": {"units": "degN", "long_name": "latitude"}},
                "lon": {"dims": "x", "data": lons, "attrs": {"units": "degE", "long_name": "longitude"}},
                "x": {"dims": "x", "data": xx, "attrs": {"units": "m", "long_name": "section_distance"}}
                },
            "attrs": {"Comments": f"Cross section data extracted from {self.full_model_name} simulations.",
                      "Conventions": "CF-1.6, ACDD"},
            "dims": ["x", "ml", "time"],
            "data_vars": {
                vari: {"dims": ["x", "ml", "time"],
                       "data": d,
                       "attrs": {"units": self.variables_names_units[vari][1],
                                 "long_name": self.variables_names_units[vari][0]}} for vari, d in data_cross.items()}}

        ds = xr.Dataset.from_dict(d_dict)
        
        ds.rio.write_crs("+proj=lcc +lat_0=77.5 +lon_0=-25 +lat_1=77.5 +lat_2=77.5 +no_defs +R=6.371e+06", inplace=True)

        ds = xr.decode_cf(ds)

        ds.to_netcdf(self.out_path)


        return






    def download_3D_data_fields(self):
        """
        Method to download the full 3D model output.

        Returns
        -------
        None.

        """


        if "Humidity" in self.varis:
            self.varis.remove("Humidity")
            self.varis += ["q"]

        model_varnames =    {'T': 'air_temperature_ml',
                             'q': 'specific_humidity_ml',
                             'ur': 'x_wind_ml',
                             'vr': 'y_wind_ml',
                             'p_surf': 'surface_air_pressure',
                             'T_surf': 'air_temperature_0m'}

        varis_to_load = {}
        for v in self.varis:
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


        start_lonlat, count_lonlat, _, _ = unisacsi.lonlat2xy(self.lon_lims, self.lat_lims, self.static_fields["lon"], self.static_fields["lat"], 2)

        idx = np.arange(start_lonlat[0], (start_lonlat[0]+count_lonlat[0]+1))
        idy = np.arange(start_lonlat[1], (start_lonlat[1]+count_lonlat[1]+1))
        idxx = idx[::self.int_x]
        idyy = idy[::self.int_y]

        model_lon = self.static_fields['lon'][idxx,:][:,idyy]
        model_lat = self.static_fields['lat'][idxx,:][:,idyy]
        model_orog = self.static_fields['orog'][idxx,:][:,idyy]
        model_lsm = self.static_fields['lsm'][idxx,:][:,idyy]
        model_x = self.static_fields['x'][idxx]
        model_y = self.static_fields['y'][idyy]

        if self.check_plot:
            # figure to check if the right area was selected
            plt.close("all")
            plt.ion()
            fig, ax = plt.subplots(1,1,subplot_kw={'projection': ccrs.Mercator()})
            ax.set_extent([self.lon_lims[0]-2., self.lon_lims[1]+2., self.lat_lims[0]-.5, self.lat_lims[1]+.5], crs=ccrs.PlateCarree())
            ax.coastlines(resolution="10m", linewidth=0.5, zorder=30)
            gl = ax.gridlines(draw_labels=False)
            gl.left_labels = True
            gl.bottom_labels = True
            gl.xformatter = LONGITUDE_FORMATTER
            gl.yformatter = LATITUDE_FORMATTER
            ax.set_title(None)
            ax.set_xlabel(None)
            ax.set_ylabel(None)
            ax.set_title(None)
            ax.pcolormesh(model_lon, model_lat, model_orog, cmap=mpl.cm.terrain, shading="nearest", transform=ccrs.PlateCarree())
            plt.show(block=False)
            plt.pause(5.)

            print("Type 'Y' to proceed to the download. Any other input will terminate the script immediately.")
            print("")
            switch_continue = input("Your input:  ")

            plt.close("all")

            if switch_continue != "Y":
                sys.exit(1)

        time = []
        data = {}
        
        # determine overall size of time dimension
        len_time = 0                                          # overall time index
        for filename in self.fileurls:                 # loop over files --> time in the order of days
            for a in range(self.num_h):                     # loop over timesteps --> time in the order of hours
                len_time += 1

        for vari in varis_to_load.keys():
            if vari in ["p_surf", "T_surf"]:
                data[vari] = np.zeros((model_lon.shape[0], model_lon.shape[1], len_time))
            else:
                data[vari] = np.zeros((model_lon.shape[0], model_lon.shape[1], self.model_levels, len_time))
        data["p"] = np.zeros((model_lon.shape[0], model_lon.shape[1], self.model_levels, len_time))
        data["z"] = np.zeros((model_lon.shape[0], model_lon.shape[1], self.model_levels, len_time))

        with Dataset(self.fileurls[0], 'r') as f:
            hybrid = f.variables['hybrid'][-self.model_levels:]
            ap = f.variables['ap'][-self.model_levels:]
            b = f.variables['b'][-self.model_levels:]

        nn = 0                                          # overall time index
        for filename in self.fileurls:                 # loop over files --> time in the order of days
            with Dataset(filename, 'r') as f:
                for qrr in self.time_ind:                      # loop over timesteps --> time in the order of hours
                    time.append(f.variables['time'][qrr])

                    # Retrieving variables from MET Norway server thredds.met.no
                    for vari, vari_met in varis_to_load.items():
                        if vari in ["p_surf", "T_surf"]:
                            data[vari][:,:,nn] = np.transpose(np.squeeze(f.variables[vari_met][qrr,0,idyy,idxx]))
                        else:
                            data[vari][:,:,:,nn] = np.transpose(np.squeeze(f.variables[vari_met][qrr,-self.model_levels:,idyy,idxx]), (2,1,0))

                        print(f'Done reading variable {vari_met} from file {filename} on thredds server')



                    data['z'][:,:,:,nn], data['p'][:,:,:,nn] = unisacsi.Calculate_height_levels_and_pressure(hybrid,ap,b, data['T_surf'][:,:,nn], data['p_surf'][:,:,nn], data['T'][:,:,:,nn],3)

                    nn += 1

        if "ur" in varis_to_load:
            # Wind u and v components in the original data are grid-related.
            # Therefore, we rotate here the wind components from grid- to earth-related coordinates.
            data["u"] = np.zeros((model_lon.shape[0], model_lon.shape[1], self.model_levels, len_time))
            data["v"] = np.zeros((model_lon.shape[0], model_lon.shape[1], self.model_levels, len_time))
            for ml in range(self.model_levels):
                for nn in range(len(time)):
                    data['u'][:,:,ml,nn], data['v'][:,:,ml,nn] = unisacsi.Rotate_uv_components(data['ur'][:,:,ml,nn], data['vr'][:,:,ml,nn],idxx, idyy, self.static_fields["lon"],2,self.model)


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
                "ml": {"dims": "ml", "data": np.arange(self.model_levels,0,-1), "attrs": {"units": "1", "long_name": "model_level"}},
                "lat": {"dims": ["x", "y"], "data": model_lat, "attrs": {"units": "degN", "long_name": "latitude"}},
                "lon": {"dims": ["x", "y"], "data": model_lon, "attrs": {"units": "degE", "long_name": "longitude"}},
                "lsm": {"dims": ["x", "y"], "data": model_lsm, "attrs": {"units": "1", "long_name": "land_sea_mask"}},
                "orog": {"dims": ["x", "y"], "data": model_orog, "attrs": {"units": "m", "long_name": "orography"}},
                "x": {"dims": ["x"], "data": model_x, "attrs": {"units": "m", "long_name": "x-coordinate in model Cartesian system"}},
                "y": {"dims": ["y"], "data": model_y, "attrs": {"units": "m", "long_name": "y-coordinate in model Cartesian system"}}
                },
            "attrs": {"Comments": f"3D data fields extracted from {self.full_model_name} simulations.",
                      "Conventions": "CF-1.6, ACDD"},
            "dims": ["x", "y", "ml", "time"],
            "data_vars": {
                vari: {"dims": ["x", "y", "ml", "time"],
                       "data": d,
                       "attrs": {"units": self.variables_names_units[vari][1],
                                 "long_name": self.variables_names_units[vari][0]}} for vari, d in data.items()}}

        ds = xr.Dataset.from_dict(d_dict)
        
        ds.rio.write_crs("+proj=lcc +lat_0=77.5 +lon_0=-25 +lat_1=77.5 +lat_2=77.5 +no_defs +R=6.371e+06", inplace=True)

        ds = xr.decode_cf(ds)

        ds.to_netcdf(self.out_path)

        return




    def download_pressure_levels(self):
        """
        Method to download 2D fields from one or more pressure levels.

        Returns
        -------
        None.

        """


        if "Humidity" in self.varis:
            self.varis.remove("Humidity")
            self.varis += ["q"]

        model_varnames =    {'T': 'air_temperature_pl',
                             'q': 'specific_humidity_pl',
                             'ur': 'x_wind_pl',
                             'vr': 'y_wind_pl'}

        varis_to_load = {}
        for v in self.varis:
            try:
                varis_to_load[v] = model_varnames[v]
            except KeyError:
                print(f"The variable {v} is unfortunately not available as nearsurface time series, please de-select in the configuration file and try again.")
                sys.exit(1)

        model_p_levels = np.array(['50','100','150','200','250','300','400','500','700','800','850','925','1000'])
        plevels = [l.split("_")[0] for l in self.p_levels]
        ind_p_levels = [np.where(model_p_levels == l)[0][0] for l in plevels]


        start_lonlat, count_lonlat, _, _ = unisacsi.lonlat2xy(self.lon_lims, self.lat_lims, self.static_fields["lon"], self.static_fields["lat"], 2)

        idx = np.arange(start_lonlat[0], (start_lonlat[0]+count_lonlat[0]+1))
        idy = np.arange(start_lonlat[1], (start_lonlat[1]+count_lonlat[1]+1))
        idxx = idx[::self.int_x]
        idyy = idy[::self.int_y]

        model_lon = self.static_fields['lon'][idxx,:][:,idyy]
        model_lat = self.static_fields['lat'][idxx,:][:,idyy]
        model_orog = self.static_fields['orog'][idxx,:][:,idyy]
        model_lsm = self.static_fields['lsm'][idxx,:][:,idyy]
        model_x = self.static_fields['x'][idxx]
        model_y = self.static_fields['y'][idyy]


        if self.check_plot:
            # figure to check if the right area was selected
            plt.close("all")
            plt.ion()
            fig, ax = plt.subplots(1,1,subplot_kw={'projection': ccrs.Mercator()})
            ax.set_extent([self.lon_lims[0]-2., self.lon_lims[1]+2., self.lat_lims[0]-.5, self.lat_lims[1]+.5], crs=ccrs.PlateCarree())
            ax.coastlines(resolution="10m", linewidth=0.5, zorder=30)
            gl = ax.gridlines(draw_labels=False)
            gl.left_labels = True
            gl.bottom_labels = True
            gl.xformatter = LONGITUDE_FORMATTER
            gl.yformatter = LATITUDE_FORMATTER
            ax.set_title(None)
            ax.set_xlabel(None)
            ax.set_ylabel(None)
            ax.set_title(None)
            ax.pcolormesh(model_lon, model_lat, model_orog, cmap=mpl.cm.terrain, shading="nearest", transform=ccrs.PlateCarree())
            plt.show(block=False)
            plt.pause(5.)

            print("Type 'Y' to proceed to the download. Any other input will terminate the script immediately.")
            print("")
            switch_continue = input("Your input:  ")

            plt.close("all")

            if switch_continue != "Y":
                sys.exit(1)

        time = []
        data = {}

        # determine overall size of time dimension
        len_time = 0                                          # overall time index
        for filename in self.fileurls:                 # loop over files --> time in the order of days
            for a in range(self.num_h):                     # loop over timesteps --> time in the order of hours
                len_time += 1

        for vari in varis_to_load.keys():
            data[vari] = np.zeros((model_lon.shape[0], model_lon.shape[1], len(ind_p_levels), len_time))

        nn = 0                                          # overall time index
        for filename in self.fileurls:                 # loop over files --> time in the order of days
            with Dataset(filename, 'r') as f:
                for qrr in self.time_ind:                      # loop over timesteps --> time in the order of hours
                    for lc, l in enumerate(ind_p_levels):
                        if lc == 0:
                            time.append(f.variables['time'][qrr])

                        # Retrieving variables from MET Norway server thredds.met.no
                        for vari, vari_met in varis_to_load.items():
                            data[vari][:,:,lc,nn] = np.transpose(np.squeeze(f.variables[vari_met][qrr,l,idyy,idxx]))

                            print(f'Done reading variable {vari_met} at p-level {self.p_levels[lc]} from file {filename} on thredds server')

                    nn += 1

        if "ur" in varis_to_load:
            # Wind u and v components in the original data are grid-related.
            # Therefore, we rotate here the wind components from grid- to earth-related coordinates.
            data["u"] = np.zeros((model_lon.shape[0], model_lon.shape[1], len(ind_p_levels), len_time))
            data["v"] = np.zeros((model_lon.shape[0], model_lon.shape[1], len(ind_p_levels), len_time))
            for ml in range(len(ind_p_levels)):
                for nn in range(len(time)):
                    data['u'][:,:,ml,nn], data['v'][:,:,ml,nn] = unisacsi.Rotate_uv_components(data['ur'][:,:,ml,nn], data['vr'][:,:,ml,nn],idxx, idyy, self.static_fields["lon"],2,self.model)


            # Calculating wind direction
            data['WD'] = (np.rad2deg(np.arctan2(-data['u'], -data['v']))+360.) % 360.

            # Calculating wind speed
            data['WS'] = np.sqrt((data['u']**2.) + (data['v']**2.))

            del data['ur']
            del data['vr']


        # Calculating potential temperature and relative humidity
        plevels = np.array([float(l) for l in plevels])
        if "T" in varis_to_load:
            data["T_pot"] = np.zeros((model_lon.shape[0], model_lon.shape[1], len(ind_p_levels), len_time))
        if (("q" in varis_to_load) & ("T" in varis_to_load)):
            data["RH"] = np.zeros((model_lon.shape[0], model_lon.shape[1], len(ind_p_levels), len_time))


        if ("T" in varis_to_load):
            for lc, l in enumerate(ind_p_levels):
                data['T_pot'][:,lc,:,:] = ((data['T'][:,lc,:,:])*((1.e3/(plevels[lc]))**(287./1005.))) - 273.15
                if ("q" in varis_to_load):
                    T = data["T"][:,lc,:,:] - 273.15
                    e = (data['q'][:,lc,:,:]*plevels[lc])/(0.622 + 0.378*data['q'][:,lc,:,:])
                    data['RH'][:,lc,:,:] = (e/(611.2 * np.exp((17.62*T)/(243.12+T)))) * 100.

        # Converting units
        if ("q" in varis_to_load):
            data['q'] *= 1000.
        if ("T" in varis_to_load):
            data['T'] -= 273.15


        time = np.array(time)

        d_dict = {
            "coords": {
                "time": {"dims": "time", "data": time, "attrs": {"units": "seconds since 1970-01-01 00:00:00"}},
                "pl": {"dims": "pl", "data": plevels, "attrs": {"units": "hPa", "long_name": "pressure_level"}},
                "lat": {"dims": ["x", "y"], "data": model_lat, "attrs": {"units": "degN", "long_name": "latitude"}},
                "lon": {"dims": ["x", "y"], "data": model_lon, "attrs": {"units": "degE", "long_name": "longitude"}},
                "lsm": {"dims": ["x", "y"], "data": model_lsm, "attrs": {"units": "1", "long_name": "land_sea_mask"}},
                "orog": {"dims": ["x", "y"], "data": model_orog, "attrs": {"units": "m", "long_name": "orography"}},
                "x": {"dims": ["x"], "data": model_x, "attrs": {"units": "m", "long_name": "x-coordinate in model Cartesian system"}},
                "y": {"dims": ["y"], "data": model_y, "attrs": {"units": "m", "long_name": "y-coordinate in model Cartesian system"}}
                },
            "attrs": {"Comments": f"Pressure level data fields extracted from {self.full_model_name} simulations.",
                      "Conventions": "CF-1.6, ACDD"},
            "dims": ["x", "y", "pl", "time"],
            "data_vars": {
                vari: {"dims": ["x", "y", "pl", "time"],
                       "data": d,
                       "attrs": {"units": self.variables_names_units[vari][1],
                                 "long_name": self.variables_names_units[vari][0]}} for vari, d in data.items()}}

        ds = xr.Dataset.from_dict(d_dict)
        
        ds.rio.write_crs("+proj=lcc +lat_0=77.5 +lon_0=-25 +lat_1=77.5 +lat_2=77.5 +no_defs +R=6.371e+06", inplace=True)

        ds = xr.decode_cf(ds)

        ds.to_netcdf(self.out_path)

        return







    def download_profiles(self):
        """
        Method to download profiles of atmospheric variables at one or several locations.

        Returns
        -------
        None.

        """


        if "Humidity" in self.varis:
            self.varis.remove("Humidity")
            self.varis += ["q"]

        model_varnames =    {'T': 'air_temperature_ml',
                             'q': 'specific_humidity_ml',
                             'ur': 'x_wind_ml',
                             'vr': 'y_wind_ml',
                             'p_surf': 'surface_air_pressure',
                             'T_surf': 'air_temperature_0m'}

        varis_to_load = {}
        for v in self.varis:
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
        coords_xx = np.zeros_like(self.stt_lon, dtype=int)
        coords_yy = np.zeros_like(self.stt_lon, dtype=int)
        model_lon = np.zeros_like(self.stt_lon)
        model_lat = np.zeros_like(self.stt_lon)
        model_orog = np.zeros_like(self.stt_lon)
        model_lsm = np.zeros_like(self.stt_lon)
        for i, (stat, lon, lat) in enumerate(zip(self.stt_name, self.stt_lon, self.stt_lat)):
            coords_xx[i], coords_yy[i], model_lon[i], model_lat[i] = unisacsi.lonlat2xy(lon, lat, self.static_fields['lon'], self.static_fields['lat'], 1)

            model_orog[i] = self.static_fields['orog'][coords_xx[i], coords_yy[i]]
            model_lsm[i] = self.static_fields['lsm'][coords_xx[i], coords_yy[i]]

        if self.check_plot:
            # figure to check if the right area was selected
            lon_min = np.nanmin([model_lon[i] for i in range(len(self.stt_name))])-1.
            lon_max = np.nanmax([model_lon[i] for i in range(len(self.stt_name))])+1.
            lat_min = np.nanmin([model_lat[i] for i in range(len(self.stt_name))])-0.5
            lat_max = np.nanmax([model_lat[i] for i in range(len(self.stt_name))])+0.5
            plt.close("all")
            plt.ion()
            fig, ax = plt.subplots(1,1,subplot_kw={'projection': ccrs.Mercator()})
            ax.set_extent([lon_min, lon_max, lat_min, lat_max], crs=ccrs.PlateCarree())
            ax.coastlines(resolution="10m", linewidth=0.5, zorder=30)
            gl = ax.gridlines(draw_labels=False)
            gl.left_labels = True
            gl.bottom_labels = True
            gl.xformatter = LONGITUDE_FORMATTER
            gl.yformatter = LATITUDE_FORMATTER
            ax.set_title(None)
            ax.set_xlabel(None)
            ax.set_ylabel(None)
            ax.set_title(None)
            for i, (lon, lat) in enumerate(zip(self.stt_lon, self.stt_lat)):
                ax.scatter(lon, lat, s=10., c="r", marker="x", transform=ccrs.PlateCarree())
                ax.scatter(model_lon[i], model_lat[i], s=10., c="k", marker="o", transform=ccrs.PlateCarree())
            plt.show(block=False)
            plt.pause(5.)

            print("Type 'Y' to proceed to the download. Any other input will terminate the script immediately.")
            print("")
            switch_continue = input("Your input:  ")

            plt.close("all")

            if switch_continue != "Y":
                sys.exit(1)

        time = []
        data = {}

        # determine overall size of time dimension
        len_time = 0                                          # overall time index
        for filename in self.fileurls:                 # loop over files --> time in the order of days
            for a in range(self.num_h):                     # loop over timesteps --> time in the order of hours
                len_time += 1


        for qr, stat in enumerate(self.stt_name):           # loop over stations
            data[stat] = {}

            for vari in varis_to_load.keys():
                if vari in ["p_surf", "T_surf"]:
                    data[stat][vari] = np.zeros((len_time))
                else:
                    data[stat][vari] = np.zeros((self.model_levels, len_time))
            data[stat]["p"] = np.zeros((self.model_levels, len_time))
            data[stat]["z"] = np.zeros((self.model_levels, len_time))

            with Dataset(self.fileurls[0], 'r') as f:
                hybrid = f.variables['hybrid'][-self.model_levels:]
                ap = f.variables['ap'][-self.model_levels:]
                b = f.variables['b'][-self.model_levels:]

            nn = 0                                       # overall time index
            for filename in self.fileurls:                    # loop over files --> time in the order of days
                with Dataset(filename, 'r') as f:
                    for qrr in self.time_ind:                      # loop over timesteps --> time in the order of hours
                        if qr == 0:
                            time.append(f.variables['time'][qrr])

                        # Retrieving variables from MET Norway server thredds.met.no
                        for vari, vari_met in varis_to_load.items():
                            if vari in ["p_surf", "T_surf"]:
                                data[stat][vari][nn] = np.squeeze(f.variables[vari_met][qrr,0,coords_yy[qr],coords_xx[qr]])
                            else:
                                data[stat][vari][:,nn] = np.squeeze(f.variables[vari_met][qrr,-self.model_levels:,coords_yy[qr],coords_xx[qr]])

                            print(f'Done reading variable {vari_met} from file {filename} on thredds server')


                        data[stat]['z'][:,nn], data[stat]['p'][:,nn] = unisacsi.Calculate_height_levels_and_pressure(hybrid,ap,b, data[stat]['T_surf'][nn], data[stat]['p_surf'][nn], data[stat]['T'][:,nn],1)

                        nn += 1

            if "ur" in varis_to_load:
                # Wind u and v components in the original data are grid-related.
                # Therefore, we rotate here the wind components from grid- to earth-related coordinates.
                data[stat]['u'], data[stat]['v'] = unisacsi.Rotate_uv_components(data[stat]['ur'], data[stat]['vr'],coords_xx[qr], coords_yy[qr], self.static_fields["lon"],1,self.model)

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


        data_reorg = {vari: np.array([data[s][vari] for s in self.stt_name]) for vari in data[self.stt_name[0]].keys()}

        d_dict = {
            "coords": {
                "time": {"dims": "time", "data": time, "attrs": {"units": "seconds since 1970-01-01 00:00:00"}},
                "ml": {"dims": "ml", "data": np.arange(self.model_levels,0,-1), "attrs": {"units": "1", "long_name": "model_level"}},
                "station_name": {"dims": "station", "data": self.stt_name, "attrs": {"units": "1"}},
                "station_lat": {"dims": "station", "data": self.stt_lat, "attrs": {"units": "degN", "long_name": "latitude_of_station"}},
                "station_lon": {"dims": "station", "data": self.stt_lon, "attrs": {"units": "degE", "long_name": "longitude_of_station"}},
                "model_lat": {"dims": "station", "data": model_lat, "attrs": {"units": "degN", "long_name": "latitude_of_model_gridpoint"}},
                "model_lon": {"dims": "station", "data": model_lon, "attrs": {"units": "degE", "long_name": "longitude_of_model_gridpoint"}},
                "model_lsm": {"dims": "station", "data": model_lsm, "attrs": {"units": "1", "long_name": "land_sea_mask_of_model_gridpoint"}},
                "model_orog": {"dims": "station", "data": model_orog, "attrs": {"units": "m", "long_name": "elevation_of_model_gridpoint"}},
                "model_x": {"dims": "station", "data": coords_xx, "attrs": {"units": "m", "long_name": "x-coordinate in model Cartesian system"}},
                "model_y": {"dims": "station", "data": coords_yy, "attrs": {"units": "m", "long_name": "y-coordinate in model Cartesian system"}}
                },
            "attrs": {"Comments": f"Profile data extracted from {self.full_model_name} simulations.",
                      "Conventions": "CF-1.6, ACDD"},
            "dims": ["station", "time", "ml"],
            "data_vars": {
                vari: {"dims": ["station", "ml", "time"],
                        "data": d,
                        "attrs": {"units": self.variables_names_units[vari][1],
                                  "long_name": self.variables_names_units[vari][0]}} for vari, d in data_reorg.items()}}

        ds = xr.Dataset.from_dict(d_dict)
        ds = ds.set_index({"station": "station_name"}, append=True)
        
        ds.rio.write_crs("+proj=lcc +lat_0=77.5 +lon_0=-25 +lat_1=77.5 +lat_2=77.5 +no_defs +R=6.371e+06", inplace=True)

        ds = xr.decode_cf(ds)

        ds.to_netcdf(self.out_path)

        return
    
    
    def new(self):
        
        if "Humidity" in self.varis:
            self.varis.remove("Humidity")
            self.varis += ["q"]

        model_varnames =    {'T': 'air_temperature_ml',
                             'q': 'specific_humidity_ml',
                             'ur': 'x_wind_ml',
                             'vr': 'y_wind_ml',
                             'p_surf': 'surface_air_pressure',
                             'T_surf': 'air_temperature_0m'}

        varis_to_load = {}
        for v in self.varis:
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
            
        model_varis = list(varis_to_load.values()) + ["x", "y", "longitude", "latitude", "projection_lambert", "surface_geopotential", "land_area_fraction", "ap", "b"]
            
            
        start_lonlat, count_lonlat, _, _ = unisacsi.lonlat2xy(self.lon_lims, self.lat_lims, self.static_fields["lon"], self.static_fields["lat"], 2)

        idx = np.arange(start_lonlat[0], (start_lonlat[0]+count_lonlat[0]+1))
        idy = np.arange(start_lonlat[1], (start_lonlat[1]+count_lonlat[1]+1))
        idxx = idx[::self.int_x]
        idyy = idy[::self.int_y]
                        
        for filename in self.fileurls:
            with xr.open_dataset(filename) as full_file:
                data = full_file.isel(time=self.time_ind, x=idxx, y=idyy, hybrid=np.arange(-self.model_levels,0,1))[model_varis].squeeze()
                
            ap, b, sp = xr.broadcast(data["ap"], data["b"], data["surface_air_pressure"])
            data["air_pressure_ml"] = ap + b*sp
            
            data["z"] = xr.zeros_like(data["air_pressure_ml"])
            
            for c, n in enumerate(range(1,len(data["hybrid"])+1)):
                if c == 0:
                    pd = data["surface_air_pressure"].values/data["air_pressure_ml"].isel(hybrid=-n).values
                    Tm = np.nanmean([data["air_temperature_0m"].values, data["air_temperature_ml"].isel(hybrid=-n).values])
                else:
                    pd = data["air_pressure_ml"].isel(hybrid=-n+1).values/data["air_pressure_ml"].isel(hybrid=-n).values
                    Tm = 0.5 * (data["air_temperature_ml"].isel(hybrid=-n+1).values + data["air_temperature_ml"].isel(hybrid=-n).values)
                data["z"][dict(hybrid=-n)] = data["z"].isel(hybrid=-n+1).values + 287.*Tm/9.81*np.log(pd)
                
            for vari, vari_met in varis_to_load.items():
                if vari_met[:8] == "integral":
                    data[vari][dict(time=range(1,len(data["time"])))] -= data[vari].diff(dim="time").values
                    data[vari][dict(time=0)] /= self.start_h
                    data[vari] /= (3600.*self.int_h)
                elif vari_met[-3:] == "acc":
                    data[vari][dict(time=range(1,len(data["time"])))] -= data[vari].diff(dim="time").values
                    data[vari][dict(time=0)] /= self.start_h

            if "SW_net" in varis_to_load:
                # Converting radiative fluxes from net into up
                data["SW_up"] = data["SW_down"] - data["SW_net"]
                data["LW_up"] = data["LW_down"] - data["LW_net"]

                del data['SW_net']
                del data['LW_net']
                
            if "x_wind_ml" in model_varis:
                # Wind u and v components in the original data are grid-related.
                # Therefore, we rotate here the wind components from grid- to earth-related coordinates.
                cone = np.sin(np.abs(np.deg2rad(data.projection_lambert.attrs["latitude_of_projection_origin"]))) # cone factor

                data["diffn"] = data.projection_lambert.attrs["longitude_of_central_meridian"] - data.longitude
                data["diffn"].values[data["diffn"].values > 180.] -= 360
                data["diffn"].values[data["diffn"].values < -180.] += 360

                data["alpha"]  = np.deg2rad(data["diffn"]) * cone

                data["u"] = data['x_wind_ml'] * np.cos(data["alpha"]) - data['y_wind_ml'] * np.sin(data["alpha"])
                data["v"] = data['y_wind_ml'] * np.cos(data["alpha"]) + data['x_wind_ml'] * np.sin(data["alpha"])

                # Calculating wind direction
                data['WD'] = (np.rad2deg(np.arctan2(-data['u'], -data['v']))+360.) % 360.

                # Calculating wind speed
                data['WS'] = np.sqrt((data['u']**2.) + (data['v']**2.))

                del data['x_wind_ml']
                del data['y_wind_ml']
                del data['diffn']
                del data['alpha']
                
                
            # Calculating potential temperature
            data['air_potential_temperature_ml'] = ((data['air_temperature_ml'])*((1.e5/(data['air_pressure_ml']))**(287./1005.))) - 273.15

            if ("q" in varis_to_load):
                T = data["air_temperature_ml"] - 273.15
                e = (data['specific_humidity_ml']*data['air_pressure'])/(0.622 + 0.378*data['specific_humidity_ml'])
                data['relative_humidty_ml'] = (e/(611.2 * np.exp((17.62*T)/(243.12+T)))) * 100.
                data['specific_humidity_ml'] *= 1000.
            
            # Converting units
            data['air_pressure_ml'] /= 100.
            data['air_temperature_ml'] -= 273.15

            print(data)
            
        return
    
    
    
