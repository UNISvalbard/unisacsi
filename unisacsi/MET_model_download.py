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

        data_formats = {0: "profile data",
                        1: "near-surface time series data",
                        2: "2D near surface field data",
                        3: "3D data",
                        4: "cross section data",
                        5: "pressure level data"}




        if "data_format" in config_settings.keys():
            if config_settings["data_format"] == 0:
                self.data_format = data_formats[0]
                self.download_profile_data()

            elif config_settings["data_format"] == 1:
                self.data_format = data_formats[1]
                self.download_near_surface_time_series_data()
                    
            elif config_settings["data_format"] == 2:
                self.data_format = data_formats[2]
                self.download_near_surface_field_data()
                
            elif config_settings["data_format"] == 3:
                self.data_format = data_formats[3]
                self.download_3D_data()
                
            elif config_settings["data_format"] == 4:
                self.data_format = data_formats[4]
                self.download_cross_section_data()
                
            elif config_settings["data_format"] == 5:
                self.data_format = data_formats[5]
                self.download_pressure_level_data()
                
            else:
                assert False, "Data format not a valid option, please change in the config file."
        else:
            assert False, "Please specify the type of data you want to download."
            


        

            
            
    def download_profile_data(self):
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
                print(f"The variable {v} is unfortunately not available as {self.data_format}, please de-select in the configuration file and try again.")
                sys.exit(1)
        if "p_surf" not in varis_to_load.keys():
            varis_to_load["p_surf"] = model_varnames["p_surf"]
        if "T_surf" not in varis_to_load.keys():
            varis_to_load["T_surf"] = model_varnames["T_surf"]
        if "T" not in varis_to_load.keys():
            varis_to_load["T"] = model_varnames["T"]
            
        model_varis = list(varis_to_load.values()) + ["x", "y", "longitude", "latitude", "projection_lambert", "surface_geopotential", "land_area_fraction", "ap", "b"]

        idxx = []
        idyy = []
        for i, (stat, lon, lat) in enumerate(zip(self.stt_name, self.stt_lon, self.stt_lat)):
            
            radius = 6378.137
            lat1 = lat * (np.pi/180.)
            lat2 = self.static_fields['lat'] * (np.pi/180.)
            lon1 = lon * (np.pi/180.)
            lon2 = self.static_fields['lon'] * (np.pi/180.)
            deltaLat = lat2 - lat1
            deltaLon = lon2 - lon1

            x = deltaLon * np.cos((lat1+lat2)/2.)
            y = deltaLat
            d2km = radius * np.sqrt(x**2. + y**2.)

            xx, yy = np.unravel_index(d2km.argmin(), d2km.shape)
            
            idxx.append(xx)
            idyy.append(yy)
              
        if self.check_plot:
            # figure to check if the right area was selected
            lon_min = np.nanmin([self.static_fields["lon"][idxx[i], idyy[i]] for i in range(len(self.stt_name))])-1.
            lon_max = np.nanmax([self.static_fields["lon"][idxx[i], idyy[i]] for i in range(len(self.stt_name))])+1.
            lat_min = np.nanmin([self.static_fields["lat"][idxx[i], idyy[i]] for i in range(len(self.stt_name))])-0.5
            lat_max = np.nanmax([self.static_fields["lat"][idxx[i], idyy[i]] for i in range(len(self.stt_name))])+0.5
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
                ax.scatter(self.static_fields["lon"][idxx[i], idyy[i]], self.static_fields["lat"][idxx[i], idyy[i]], s=10., c="k", marker="o", transform=ccrs.PlateCarree())
            plt.show(block=False)
            plt.pause(5.)

            print("Type 'Y' to proceed to the download. Any other input will terminate the script immediately.")
            print("")
            switch_continue = input("Your input:  ")

            plt.close("all")

            if switch_continue != "Y":
                sys.exit(1)
        
        
        chunks = []
        for filename in self.fileurls:
            with xr.open_dataset(filename) as full_file:
                data = full_file.isel(time=self.time_ind, x=idxx, y=idyy, hybrid=np.arange(-self.model_levels,0,1))[model_varis].squeeze()
                
            ap, b, sp = xr.broadcast(data["ap"], data["b"], data["surface_air_pressure"])
            data["air_pressure_ml"] = ap + b*sp
            
            data["z"] = xr.zeros_like(data["air_pressure_ml"])
            
            for c, n in enumerate(range(1,len(data["hybrid"])+1)):
                if c == 0:
                    p_d = data["surface_air_pressure"].values/data["air_pressure_ml"].isel(hybrid=-n).values
                    Tm = np.nanmean([data["air_temperature_0m"].values, data["air_temperature_ml"].isel(hybrid=-n).values])
                else:
                    p_d = data["air_pressure_ml"].isel(hybrid=-n+1).values/data["air_pressure_ml"].isel(hybrid=-n).values
                    Tm = 0.5 * (data["air_temperature_ml"].isel(hybrid=-n+1).values + data["air_temperature_ml"].isel(hybrid=-n).values)
                data["z"][dict(hybrid=-n)] = data["z"].isel(hybrid=-n+1).values + 287.*Tm/9.81*np.log(p_d)

                
            if "x_wind_ml" in model_varis:
                # Wind u and v components in the original data are grid-related.
                # Therefore, we rotate here the wind components from grid- to earth-related coordinates.
                cone = np.sin(np.abs(np.deg2rad(data.projection_lambert.attrs["latitude_of_projection_origin"]))) # cone factor

                data["diffn"] = data.projection_lambert.attrs["longitude_of_central_meridian"] - data.longitude
                data["diffn"].values[data["diffn"].values > 180.] -= 360
                data["diffn"].values[data["diffn"].values < -180.] += 360

                data["alpha"]  = np.deg2rad(data["diffn"]) * cone

                data["eastward_wind_ml"] = data['x_wind_ml'] * np.cos(data["alpha"]) - data['y_wind_ml'] * np.sin(data["alpha"])
                data["northward_wind_ml"] = data['y_wind_ml'] * np.cos(data["alpha"]) + data['x_wind_ml'] * np.sin(data["alpha"])

                # Calculating wind direction
                data['wind_direction_ml'] = (np.rad2deg(np.arctan2(-data['eastward_wind_ml'], -data['northward_wind_ml']))+360.) % 360.

                # Calculating wind speed
                data['wind_speed_ml'] = np.sqrt((data['eastward_wind_ml']**2.) + (data['northward_wind_ml']**2.))

                del data['x_wind_ml']
                del data['y_wind_ml']
                del data['diffn']
                del data['alpha']
                
                
            # Calculating potential temperature
            data['air_potential_temperature_ml'] = (data['air_temperature_ml'])*((1.e5/(data['air_pressure_ml']))**(287./1005.))

            if 'specific_humidity_ml' in model_varis:
                T = data["air_temperature_ml"] - 273.15
                e = (data['specific_humidity_ml']*data['air_pressure'])/(0.622 + 0.378*data['specific_humidity_ml'])
                data['relative_humidty_ml'] = e/(611.2 * np.exp((17.62*T)/(243.12+T)))
                
            
            del data["surface_air_pressure"]
            del data["air_temperature_0m"]
            
            chunks.append(data)
        
        ds = xr.concat(chunks, dim="time")
        ds.to_netcdf(self.out_path)

        return
    
    
    
    
            
            
    def download_near_surface_time_series_data(self):
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
                print(f"The variable {v} is unfortunately not available as {self.data_format}, please de-select in the configuration file and try again.")
                sys.exit(1)
                
        model_varis = list(varis_to_load.values()) + ["x", "y", "longitude", "latitude", "projection_lambert", "surface_geopotential", "land_area_fraction"]

        idxx = []
        idyy = []
        for i, (stat, lon, lat) in enumerate(zip(self.stt_name, self.stt_lon, self.stt_lat)):
            coords_xx, coords_yy, _, _ = unisacsi.lonlat2xy(lon, lat, self.static_fields['lon'], self.static_fields['lat'], 1)
            idxx.append(coords_xx)
            idyy.append(coords_yy)
            
        if self.check_plot:
            # figure to check if the right area was selected
            lon_min = np.nanmin([self.static_fields["lon"][idxx[i], idyy[i]] for i in range(len(self.stt_name))])-1.
            lon_max = np.nanmax([self.static_fields["lon"][idxx[i], idyy[i]] for i in range(len(self.stt_name))])+1.
            lat_min = np.nanmin([self.static_fields["lat"][idxx[i], idyy[i]] for i in range(len(self.stt_name))])-0.5
            lat_max = np.nanmax([self.static_fields["lat"][idxx[i], idyy[i]] for i in range(len(self.stt_name))])+0.5
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
                ax.scatter(self.static_fields["lon"][idxx[i], idyy[i]], self.static_fields["lat"][idxx[i], idyy[i]], s=10., c="k", marker="o", transform=ccrs.PlateCarree())
            plt.show(block=False)
            plt.pause(5.)

            print("Type 'Y' to proceed to the download. Any other input will terminate the script immediately.")
            print("")
            switch_continue = input("Your input:  ")

            plt.close("all")

            if switch_continue != "Y":
                sys.exit(1)
            
            
        chunks = []
        for filename in self.fileurls:
            with xr.open_dataset(filename) as full_file:
                data = full_file.isel(time=self.time_ind, x=idxx, y=idyy)[model_varis].squeeze()
                
            for vari in model_varis:
                if vari[:8] == "integral":
                    data[vari][dict(time=range(1,len(data["time"])))] -= data[vari].diff(dim="time").values
                    data[vari][dict(time=0)] /= self.start_h
                    data[vari] /= (3600.*self.int_h)
                    data = data.rename({vari: vari[12:-9]})
                elif vari[-3:] == "acc":
                    data[vari][dict(time=range(1,len(data["time"])))] -= data[vari].diff(dim="time").values
                    data[vari][dict(time=0)] /= self.start_h
                    data = data.rename({vari: vari[:-4]})

            if "integral_of_surface_net_downward_shortwave_flux_wrt_time" in model_varis:
                # Converting radiative fluxes from net into up
                data["surface_upwelling_shortwave_flux_in_air"] = data["surface_downwelling_shortwave_flux_in_air"] - data["surface_net_downward_shortwave_flux"]
                data["surface_upwelling_longwave_flux_in_air"] = data["surface_downwelling_longwave_flux_in_air"] - data["surface_net_downward_longwave_flux"]

                del data['surface_net_downward_shortwave_flux']
                del data['surface_net_downward_longwave_flux']
                
            if "x_wind_10m" in model_varis:
                # Wind u and v components in the original data are grid-related.
                # Therefore, we rotate here the wind components from grid- to earth-related coordinates.
                cone = np.sin(np.abs(np.deg2rad(data.projection_lambert.attrs["latitude_of_projection_origin"]))) # cone factor

                data["diffn"] = data.projection_lambert.attrs["longitude_of_central_meridian"] - data.longitude
                data["diffn"].values[data["diffn"].values > 180.] -= 360
                data["diffn"].values[data["diffn"].values < -180.] += 360

                data["alpha"]  = np.deg2rad(data["diffn"]) * cone

                data["eastward_wind_10m"] = data['x_wind_10m'] * np.cos(data["alpha"]) - data['y_wind_10m'] * np.sin(data["alpha"])
                data["northward_wind_10m"] = data['y_wind_10m'] * np.cos(data["alpha"]) + data['x_wind_10m'] * np.sin(data["alpha"])

                # Calculating wind direction
                data['wind_direction_10m'] = (np.rad2deg(np.arctan2(-data['eastward_wind_10m'], -data['northward_wind_10m']))+360.) % 360.

                # Calculating wind speed
                data['wind_speed_10m'] = np.sqrt((data['eastward_wind_10m']**2.) + (data['northward_wind_10m']**2.))

                del data['x_wind_10m']
                del data['y_wind_10m']
                del data['diffn']
                del data['alpha']
                
                
            # Calculating potential temperature
            if (('air_temperature_2m' in model_varis) & ("surface_air_pressure" in model_varis)):
                data['air_potential_temperature_2m'] = (data['air_temperature_2m'])*((1.e5/(data['surface_air_pressure']))**(287./1005.))

            if 'specific_humidity_2m' in model_varis:
                T = data["air_temperature_2m"] - 273.15
                e = (data['specific_humidity_2m']*data['surface_air_pressure'])/(0.622 + 0.378*data['specific_humidity_2m'])
                data['relative_humidty_2m'] = (e/(611.2 * np.exp((17.62*T)/(243.12+T))))
            
                
            chunks.append(data)
        
        ds = xr.concat(chunks, dim="time")
        ds.to_netcdf(self.out_path)
        
        
        return
    
    
    
    def download_near_surface_field_data(self):
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
                print(f"The variable {v} is unfortunately not available as {self.data_format}, please de-select in the configuration file and try again.")
                sys.exit(1)
                
        model_varis = list(varis_to_load.values()) + ["x", "y", "longitude", "latitude", "projection_lambert", "surface_geopotential", "land_area_fraction"]

        lon_corners  = [self.lon_lims[0], self.lon_lims[1], self.lon_lims[1], self.lon_lims[0]]
        lat_corners  = [self.lat_lims[0], self.lat_lims[0], self.lat_lims[1], self.lat_lims[1]]

        coords_xx = np.zeros(len(lon_corners))
        coords_yy = np.zeros(len(lon_corners))

        for qq in range(len(lon_corners)):
            lonn1 = lon_corners[qq]
            latt1 = lat_corners[qq]

            radius = 6378.137
            lat1 = latt1 * (np.pi/180.)
            lat2 = self.static_fields["lat"] * (np.pi/180.)
            lon1 = lonn1 * (np.pi/180.)
            lon2 = self.static_fields["lon"] * (np.pi/180.)
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

        start_lonlat = np.array([lonmin_id, latmin_id])
        count_lonlat = np.array([np.abs(lonmax_id-lonmin_id), np.abs(latmax_id-latmin_id)])
        
        idx = np.arange(start_lonlat[0], (start_lonlat[0]+count_lonlat[0]+1))
        idy = np.arange(start_lonlat[1], (start_lonlat[1]+count_lonlat[1]+1))
        idxx = idx[::self.int_x]
        idyy = idy[::self.int_y]
        
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
            ax.pcolormesh(self.static_fields['lon'][idxx,:][:,idyy], self.static_fields['lat'][idxx,:][:,idyy], self.static_fields['orog'][idxx,:][:,idyy], cmap=mpl.cm.terrain, shading="nearest", transform=ccrs.PlateCarree())
            plt.show(block=False)
            plt.pause(5.)

            print("Type 'Y' to proceed to the download. Any other input will terminate the script immediately.")
            print("")
            switch_continue = input("Your input:  ")

            plt.close("all")

            if switch_continue != "Y":
                sys.exit(1)
            
            
        chunks = []
        for filename in self.fileurls:
            with xr.open_dataset(filename) as full_file:
                data = full_file.isel(time=self.time_ind, x=idxx, y=idyy)[model_varis].squeeze()
                
            for vari in model_varis:
                if vari[:8] == "integral":
                    data[vari][dict(time=range(1,len(data["time"])))] -= data[vari].diff(dim="time").values
                    data[vari][dict(time=0)] /= self.start_h
                    data[vari] /= (3600.*self.int_h)
                    data = data.rename({vari: vari[12:-9]})
                elif vari[-3:] == "acc":
                    data[vari][dict(time=range(1,len(data["time"])))] -= data[vari].diff(dim="time").values
                    data[vari][dict(time=0)] /= self.start_h
                    data = data.rename({vari: vari[:-4]})

            if "integral_of_surface_net_downward_shortwave_flux_wrt_time" in model_varis:
                # Converting radiative fluxes from net into up
                data["surface_upwelling_shortwave_flux_in_air"] = data["surface_downwelling_shortwave_flux_in_air"] - data["surface_net_downward_shortwave_flux"]
                data["surface_upwelling_longwave_flux_in_air"] = data["surface_downwelling_longwave_flux_in_air"] - data["surface_net_downward_longwave_flux"]

                del data['surface_net_downward_shortwave_flux']
                del data['surface_net_downward_longwave_flux']
                
            if "x_wind_10m" in model_varis:
                # Wind u and v components in the original data are grid-related.
                # Therefore, we rotate here the wind components from grid- to earth-related coordinates.
                cone = np.sin(np.abs(np.deg2rad(data.projection_lambert.attrs["latitude_of_projection_origin"]))) # cone factor

                data["diffn"] = data.projection_lambert.attrs["longitude_of_central_meridian"] - data.longitude
                data["diffn"].values[data["diffn"].values > 180.] -= 360
                data["diffn"].values[data["diffn"].values < -180.] += 360

                data["alpha"]  = np.deg2rad(data["diffn"]) * cone

                data["eastward_wind_10m"] = data['x_wind_10m'] * np.cos(data["alpha"]) - data['y_wind_10m'] * np.sin(data["alpha"])
                data["northward_wind_10m"] = data['y_wind_10m'] * np.cos(data["alpha"]) + data['x_wind_10m'] * np.sin(data["alpha"])

                # Calculating wind direction
                data['wind_direction_10m'] = (np.rad2deg(np.arctan2(-data['eastward_wind_10m'], -data['northward_wind_10m']))+360.) % 360.

                # Calculating wind speed
                data['wind_speed_10m'] = np.sqrt((data['eastward_wind_10m']**2.) + (data['northward_wind_10m']**2.))

                del data['x_wind_10m']
                del data['y_wind_10m']
                del data['diffn']
                del data['alpha']
                
                
            # Calculating potential temperature
            if (("air_temperature_2m" in model_varis) & ("surface_air_pressure" in model_varis)):
                data['air_potential_temperature_2m'] = (data['air_temperature_2m'])*((1.e5/(data['surface_air_pressure']))**(287./1005.))

            if 'specific_humidity_2m' in model_varis:
                T = data["air_temperature_2m"] - 273.15
                e = (data['specific_humidity_2m']*data['surface_air_pressure'])/(0.622 + 0.378*data['specific_humidity_2m'])
                data['relative_humidty_2m'] = (e/(611.2 * np.exp((17.62*T)/(243.12+T))))
            
                
            chunks.append(data)
        
        ds = xr.concat(chunks, dim="time")
        ds.to_netcdf(self.out_path)
        
        
        return
    
    
    
    def download_3D_data(self):
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
                print(f"The variable {v} is unfortunately not available as {self.data_format}, please de-select in the configuration file and try again.")
                sys.exit(1)
        if "p_surf" not in varis_to_load.keys():
            varis_to_load["p_surf"] = model_varnames["p_surf"]
        if "T_surf" not in varis_to_load.keys():
            varis_to_load["T_surf"] = model_varnames["T_surf"]
        if "T" not in varis_to_load.keys():
            varis_to_load["T"] = model_varnames["T"]
            
        model_varis = list(varis_to_load.values()) + ["x", "y", "longitude", "latitude", "projection_lambert", "surface_geopotential", "land_area_fraction", "ap", "b"]

        lon_corners  = [self.lon_lims[0], self.lon_lims[1], self.lon_lims[1], self.lon_lims[0]]
        lat_corners  = [self.lat_lims[0], self.lat_lims[0], self.lat_lims[1], self.lat_lims[1]]

        coords_xx = np.zeros(len(lon_corners))
        coords_yy = np.zeros(len(lon_corners))

        for qq in range(len(lon_corners)):
            lonn1 = lon_corners[qq]
            latt1 = lat_corners[qq]

            radius = 6378.137
            lat1 = latt1 * (np.pi/180.)
            lat2 = self.static_fields["lat"] * (np.pi/180.)
            lon1 = lonn1 * (np.pi/180.)
            lon2 = self.static_fields["lon"] * (np.pi/180.)
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

        start_lonlat = np.array([lonmin_id, latmin_id])
        count_lonlat = np.array([np.abs(lonmax_id-lonmin_id), np.abs(latmax_id-latmin_id)])
        
        idx = np.arange(start_lonlat[0], (start_lonlat[0]+count_lonlat[0]+1))
        idy = np.arange(start_lonlat[1], (start_lonlat[1]+count_lonlat[1]+1))
        idxx = idx[::self.int_x]
        idyy = idy[::self.int_y]
        
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
            ax.pcolormesh(self.static_fields['lon'][idxx,:][:,idyy], self.static_fields['lat'][idxx,:][:,idyy], self.static_fields['orog'][idxx,:][:,idyy], cmap=mpl.cm.terrain, shading="nearest", transform=ccrs.PlateCarree())
            plt.show(block=False)
            plt.pause(5.)

            print("Type 'Y' to proceed to the download. Any other input will terminate the script immediately.")
            print("")
            switch_continue = input("Your input:  ")

            plt.close("all")

            if switch_continue != "Y":
                sys.exit(1)
            
            
        chunks = []
        for filename in self.fileurls:
            with xr.open_dataset(filename) as full_file:
                data = full_file.isel(time=self.time_ind, x=idxx, y=idyy, hybrid=np.arange(-self.model_levels,0,1))[model_varis].squeeze()
                
            ap, b, sp = xr.broadcast(data["ap"], data["b"], data["surface_air_pressure"])
            data["air_pressure_ml"] = ap + b*sp
            
            data["z"] = xr.zeros_like(data["air_pressure_ml"])
            
            for c, n in enumerate(range(1,len(data["hybrid"])+1)):
                if c == 0:
                    p_d = data["surface_air_pressure"].values/data["air_pressure_ml"].isel(hybrid=-n).values
                    Tm = np.nanmean([data["air_temperature_0m"].values, data["air_temperature_ml"].isel(hybrid=-n).values])
                else:
                    p_d = data["air_pressure_ml"].isel(hybrid=-n+1).values/data["air_pressure_ml"].isel(hybrid=-n).values
                    Tm = 0.5 * (data["air_temperature_ml"].isel(hybrid=-n+1).values + data["air_temperature_ml"].isel(hybrid=-n).values)
                data["z"][dict(hybrid=-n)] = data["z"].isel(hybrid=-n+1).values + 287.*Tm/9.81*np.log(p_d)

                
                
            if "x_wind_ml" in model_varis:
                # Wind u and v components in the original data are grid-related.
                # Therefore, we rotate here the wind components from grid- to earth-related coordinates.
                cone = np.sin(np.abs(np.deg2rad(data.projection_lambert.attrs["latitude_of_projection_origin"]))) # cone factor

                data["diffn"] = data.projection_lambert.attrs["longitude_of_central_meridian"] - data.longitude
                data["diffn"].values[data["diffn"].values > 180.] -= 360
                data["diffn"].values[data["diffn"].values < -180.] += 360

                data["alpha"]  = np.deg2rad(data["diffn"]) * cone

                data["eastward_wind_ml"] = data['x_wind_ml'] * np.cos(data["alpha"]) - data['y_wind_ml'] * np.sin(data["alpha"])
                data["northward_wind_ml"] = data['y_wind_ml'] * np.cos(data["alpha"]) + data['x_wind_ml'] * np.sin(data["alpha"])

                # Calculating wind direction
                data['wind_direction_ml'] = (np.rad2deg(np.arctan2(-data['eastward_wind_ml'], -data['northward_wind_ml']))+360.) % 360.

                # Calculating wind speed
                data['wind_speed_ml'] = np.sqrt((data['eastward_wind_ml']**2.) + (data['northward_wind_ml']**2.))

                del data['x_wind_ml']
                del data['y_wind_ml']
                del data['diffn']
                del data['alpha']
                
                
            # Calculating potential temperature
            if 'air_temperature_ml' in model_varis:
                data['air_potential_temperature_ml'] = (data['air_temperature_ml'])*((1.e5/(data['air_pressure_ml']))**(287./1005.))

            if 'specific_humidity_ml' in model_varis:
                T = data["air_temperature_ml"] - 273.15
                e = (data['specific_humidity_ml']*data['air_pressure_ml'])/(0.622 + 0.378*data['specific_humidity_ml'])
                data['relative_humidty_ml'] = (e/(611.2 * np.exp((17.62*T)/(243.12+T))))
            
                
            chunks.append(data)
        
        ds = xr.concat(chunks, dim="time")
        ds.to_netcdf(self.out_path)
        
        
        
        return
    
    
    
    def download_cross_section_data(self):
        """
        Method to download a (vertical) cross section.

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
                print(f"The variable {v} is unfortunately not available as {self.data_format}, please de-select in the configuration file and try again.")
                sys.exit(1)
        if "p_surf" not in varis_to_load.keys():
            varis_to_load["p_surf"] = model_varnames["p_surf"]
        if "T_surf" not in varis_to_load.keys():
            varis_to_load["T_surf"] = model_varnames["T_surf"]
        if "T" not in varis_to_load.keys():
            varis_to_load["T"] = model_varnames["T"]
            
        model_varis = list(varis_to_load.values()) + ["x", "y", "longitude", "latitude", "projection_lambert", "surface_geopotential", "land_area_fraction", "ap", "b"]

        
        lat_lims = [np.min([self.start_point[0], self.end_point[0]]), np.max([self.start_point[0], self.end_point[0]])]
        lon_lims = [np.min([self.start_point[1], self.end_point[1]]), np.max([self.start_point[1], self.end_point[1]])]
        start_lonlat, count_lonlat, _, _ = unisacsi.lonlat2xy(lon_lims, lat_lims, self.static_fields["lon"], self.static_fields["lat"], 2)
        start_lonlat -= 2
        count_lonlat += 5
        idx = np.arange(start_lonlat[0], (start_lonlat[0]+count_lonlat[0]+1))
        idy = np.arange(start_lonlat[1], (start_lonlat[1]+count_lonlat[1]+1))
        idxx = idx[::self.int_x]
        idyy = idy[::self.int_y]
        
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
            ax.pcolormesh(self.static_fields['lon'][idxx,:][:,idyy], self.static_fields['lat'][idxx,:][:,idyy], self.static_fields['orog'][idxx,:][:,idyy], cmap=mpl.cm.terrain, shading="nearest", transform=ccrs.PlateCarree())
            ax.plot([self.start_point[1], self.end_point[1]], [self.start_point[0], self.end_point[0]], 'r-', transform=ccrs.PlateCarree())
            plt.show(block=False)
            plt.pause(5.)

            print("Type 'Y' to proceed to the download. Any other input will terminate the script immediately.")
            print("")
            switch_continue = input("Your input:  ")

            plt.close("all")

            if switch_continue != "Y":
                sys.exit(1)
            
            
        chunks = []
        for filename in self.fileurls:
            with xr.open_dataset(filename) as full_file:
                data = full_file.isel(time=self.time_ind, x=idxx, y=idyy, hybrid=np.arange(-self.model_levels,0,1))[model_varis].squeeze()
                
            ap, b, sp = xr.broadcast(data["ap"], data["b"], data["surface_air_pressure"])
            data["air_pressure_ml"] = ap + b*sp
            
            data["z"] = xr.zeros_like(data["air_pressure_ml"])
            
            for c, n in enumerate(range(1,len(data["hybrid"])+1)):
                if c == 0:
                    p_d = data["surface_air_pressure"].values/data["air_pressure_ml"].isel(hybrid=-n).values
                    Tm = np.nanmean([data["air_temperature_0m"].values, data["air_temperature_ml"].isel(hybrid=-n).values])
                else:
                    p_d = data["air_pressure_ml"].isel(hybrid=-n+1).values/data["air_pressure_ml"].isel(hybrid=-n).values
                    Tm = 0.5 * (data["air_temperature_ml"].isel(hybrid=-n+1).values + data["air_temperature_ml"].isel(hybrid=-n).values)
                data["z"][dict(hybrid=-n)] = data["z"].isel(hybrid=-n+1).values + 287.*Tm/9.81*np.log(p_d)

                
                
            if "x_wind_ml" in model_varis:
                # Wind u and v components in the original data are grid-related.
                # Therefore, we rotate here the wind components from grid- to earth-related coordinates.
                cone = np.sin(np.abs(np.deg2rad(data.projection_lambert.attrs["latitude_of_projection_origin"]))) # cone factor

                data["diffn"] = data.projection_lambert.attrs["longitude_of_central_meridian"] - data.longitude
                data["diffn"].values[data["diffn"].values > 180.] -= 360
                data["diffn"].values[data["diffn"].values < -180.] += 360

                data["alpha"]  = np.deg2rad(data["diffn"]) * cone

                data["eastward_wind_ml"] = data['x_wind_ml'] * np.cos(data["alpha"]) - data['y_wind_ml'] * np.sin(data["alpha"])
                data["northward_wind_ml"] = data['y_wind_ml'] * np.cos(data["alpha"]) + data['x_wind_ml'] * np.sin(data["alpha"])

                # Calculating wind direction
                data['wind_direction_ml'] = (np.rad2deg(np.arctan2(-data['eastward_wind_ml'], -data['northward_wind_ml']))+360.) % 360.

                # Calculating wind speed
                data['wind_speed_ml'] = np.sqrt((data['eastward_wind_ml']**2.) + (data['northward_wind_ml']**2.))

                del data['x_wind_ml']
                del data['y_wind_ml']
                del data['diffn']
                del data['alpha']
                
                
            # Calculating potential temperature
            if 'air_temperature_ml' in model_varis:
                data['air_potential_temperature_ml'] = (data['air_temperature_ml'])*((1.e5/(data['air_pressure_ml']))**(287./1005.))

            if 'specific_humidity_ml' in model_varis:
                T = data["air_temperature_ml"] - 273.15
                e = (data['specific_humidity_ml']*data['air_pressure_ml'])/(0.622 + 0.378*data['specific_humidity_ml'])
                data['relative_humidty_ml'] = (e/(611.2 * np.exp((17.62*T)/(243.12+T))))
            
                
            chunks.append(data)
        
        ds = xr.concat(chunks, dim="time")
        ds.to_netcdf(self.out_path)

        return
    
    
    
    
    
    
    
    def download_pressure_level_data(self):
        """
        Method to download 2D fields from one or more pressure levels.

        Returns
        -------
        None.

        """
        
        if "Humidity" in self.varis:
            self.varis.remove("Humidity")
            self.varis += ["q", "RH"]
            
        model_varnames =    {'T': 'air_temperature_pl',
                             'q': 'specific_humidity_pl',
                             "RH": "relative_humidity_pl",
                             'ur': 'x_wind_pl',
                             'vr': 'y_wind_pl'}
        
        varis_to_load = {}
        for v in self.varis:
            try:
                varis_to_load[v] = model_varnames[v]
            except KeyError:
                print(f"The variable {v} is unfortunately not available as {self.data_format}, please de-select in the configuration file and try again.")
                sys.exit(1)
                
        model_varis = list(varis_to_load.values()) + ["x", "y", "longitude", "latitude", "projection_lambert", "surface_geopotential", "land_area_fraction", "geopotential_pl"]


        model_p_levels = np.array(['50','100','150','200','250','300','400','500','700','800','850','925','1000'])
        plevels = [l.split("_")[0] for l in self.p_levels]
        ind_p_levels = [np.where(model_p_levels == l)[0][0] for l in plevels]

        lon_corners  = [self.lon_lims[0], self.lon_lims[1], self.lon_lims[1], self.lon_lims[0]]
        lat_corners  = [self.lat_lims[0], self.lat_lims[0], self.lat_lims[1], self.lat_lims[1]]

        coords_xx = np.zeros(len(lon_corners))
        coords_yy = np.zeros(len(lon_corners))

        for qq in range(len(lon_corners)):
            lonn1 = lon_corners[qq]
            latt1 = lat_corners[qq]

            radius = 6378.137
            lat1 = latt1 * (np.pi/180.)
            lat2 = self.static_fields["lat"] * (np.pi/180.)
            lon1 = lonn1 * (np.pi/180.)
            lon2 = self.static_fields["lon"] * (np.pi/180.)
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

        start_lonlat = np.array([lonmin_id, latmin_id])
        count_lonlat = np.array([np.abs(lonmax_id-lonmin_id), np.abs(latmax_id-latmin_id)])
        
        idx = np.arange(start_lonlat[0], (start_lonlat[0]+count_lonlat[0]+1))
        idy = np.arange(start_lonlat[1], (start_lonlat[1]+count_lonlat[1]+1))
        idxx = idx[::self.int_x]
        idyy = idy[::self.int_y]
        
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
            ax.pcolormesh(self.static_fields['lon'][idxx,:][:,idyy], self.static_fields['lat'][idxx,:][:,idyy], self.static_fields['orog'][idxx,:][:,idyy], cmap=mpl.cm.terrain, shading="nearest", transform=ccrs.PlateCarree())
            plt.show(block=False)
            plt.pause(5.)

            print("Type 'Y' to proceed to the download. Any other input will terminate the script immediately.")
            print("")
            switch_continue = input("Your input:  ")

            plt.close("all")

            if switch_continue != "Y":
                sys.exit(1)
                
                
        chunks = []
        for filename in self.fileurls:
            with xr.open_dataset(filename) as full_file:
                data = full_file.isel(time=self.time_ind, x=idxx, y=idyy, pressure=ind_p_levels)[model_varis].squeeze()
                
                
            if "x_wind_pl" in model_varis:
                # Wind u and v components in the original data are grid-related.
                # Therefore, we rotate here the wind components from grid- to earth-related coordinates.
                cone = np.sin(np.abs(np.deg2rad(data.projection_lambert.attrs["latitude_of_projection_origin"]))) # cone factor

                data["diffn"] = data.projection_lambert.attrs["longitude_of_central_meridian"] - data.longitude
                data["diffn"].values[data["diffn"].values > 180.] -= 360
                data["diffn"].values[data["diffn"].values < -180.] += 360

                data["alpha"]  = np.deg2rad(data["diffn"]) * cone

                data["eastward_wind_pl"] = data['x_wind_pl'] * np.cos(data["alpha"]) - data['y_wind_pl'] * np.sin(data["alpha"])
                data["northward_wind_pl"] = data['y_wind_pl'] * np.cos(data["alpha"]) + data['x_wind_pl'] * np.sin(data["alpha"])

                # Calculating wind direction
                data['wind_direction_pl'] = (np.rad2deg(np.arctan2(-data['eastward_wind_pl'], -data['northward_wind_pl']))+360.) % 360.

                # Calculating wind speed
                data['wind_speed_pl'] = np.sqrt((data['eastward_wind_pl']**2.) + (data['northward_wind_pl']**2.))

                del data['x_wind_pl']
                del data['y_wind_pl']
                del data['diffn']
                del data['alpha']
        
        ds = xr.concat(chunks, dim="time")
        ds.to_netcdf(self.out_path)
        
        

        
        return
    
    
    
        
        
        
        

        

    