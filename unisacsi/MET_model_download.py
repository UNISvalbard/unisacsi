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
import metpy


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
        
        
    if config_settings["latest"]:
        config_settings["out_path"] = f"{config_settings['out_path']}_{config_settings['resolution']}.nc"
        print("############################################################")
        print("start downloading latest data")
        print("############################################################")
        try:
            MET_model_download_class(config_settings)
        except:
            print("ERROR! download failed, skipping file")
    elif config_settings["save_daily_files"]:
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
                try:
                    MET_model_download_class(daily_config)
                except:
                    print("ERROR! download failed, skipping file")
    else:
        config_settings["out_path"] = f"{config_settings['out_path']}_{config_settings['resolution']}.nc"
        try:
            MET_model_download_class(config_settings)
        except:
            print("ERROR! download failed, skipping file")


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

        self.latest = config_settings["latest"]
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
        self.start_point = [config_settings["section"]["start_lat"], config_settings["section"]["start_lon"]]
        self.end_point = [config_settings["section"]["end_lat"], config_settings["section"]["end_lon"]]
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
        self.static_fields = xr.open_dataset(self.static_file)
        self.dxdy = np.nanmean(np.diff(self.static_fields["x"]))
        

        self.fileurls = []
        if self.model == "AA":
            self.full_model_name = "AROME_Arctic"
            if self.resolution == "2p5km":
                if self.latest:
                    self.fileurls.append("https://thredds.met.no/thredds/dodsC/aromearcticlatest/archive/arome_arctic_det_2_5km_latest.nc")
                else:
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
                        4: "vertical cross section data",
                        5: "horizontal near-surface section",
                        6: "pressure level data"}




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
                self.download_vertical_cross_section_data()
                
            elif config_settings["data_format"] == 5:
                self.data_format = data_formats[5]
                self.download_horizontal_near_surface_section_data()
                
            elif config_settings["data_format"] == 6:
                self.data_format = data_formats[6]
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

        data_crs = ccrs.CRS(self.static_fields.projection_lambert.attrs["proj4"])
        xx = []
        yy = []
        for i in range(len(self.stt_lon)):
            x, y = data_crs.transform_point(self.stt_lon[i], self.stt_lat[i], src_crs=ccrs.PlateCarree())
            xx.append(x)
            yy.append(y)
              
        if self.check_plot:
            # figure to check if the right area was selected
            lon_min = np.nanmin([self.static_fields.sel(x=xx[i], y=yy[i], method='nearest')["longitude"] for i in range(len(self.stt_name))])-1.
            lon_max = np.nanmax([self.static_fields.sel(x=xx[i], y=yy[i], method='nearest')["longitude"] for i in range(len(self.stt_name))])+1.
            lat_min = np.nanmin([self.static_fields.sel(x=xx[i], y=yy[i], method='nearest')["latitude"] for i in range(len(self.stt_name))])-0.5
            lat_max = np.nanmax([self.static_fields.sel(x=xx[i], y=yy[i], method='nearest')["latitude"] for i in range(len(self.stt_name))])+0.5
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
                ax.scatter(self.static_fields.sel(x=xx[i], y=yy[i], method='nearest')["longitude"], self.static_fields.sel(x=xx[i], y=yy[i], method='nearest')["latitude"], s=10., c="k", marker="o", transform=ccrs.PlateCarree())
            plt.show(block=False)
            plt.pause(5.)

            print("Type 'Y' to proceed to the download. Any other input will terminate the script immediately.")
            print("")
            switch_continue = input("Your input:  ")

            plt.close("all")

            if switch_continue != "Y":
                sys.exit(1)
                
            print("Starting data download...")
        
        
        chunks = []
        for filename in self.fileurls:
            stations = []
            with xr.open_dataset(filename) as full_file:
                for i in range(len(self.stt_lon)):
                    stations.append(full_file.isel(time=self.time_ind, hybrid=np.arange(-self.model_levels,0,1)).sel(x=xx[i], y=yy[i], method='nearest')[model_varis].squeeze())
                    
            data = xr.concat(stations, dim="station")
            
            chunks.append(data)
            
            print(f"Done downloading {filename}.")
        
        ds = xr.concat(chunks, dim="time")
        ds["stations"] = xr.DataArray(self.stt_name, dims=['station'])
                                
        ap, b, sp = xr.broadcast(ds["ap"], ds["b"], ds["surface_air_pressure"])
        ds["air_pressure_ml"] = ap + b*sp
        
        ds["z"] = xr.zeros_like(ds["air_pressure_ml"])
        
        for c, n in enumerate(range(1,len(ds["hybrid"])+1)):
            if c == 0:
                p_d = (ds["surface_air_pressure"]/ds["air_pressure_ml"].isel(hybrid=-n))
                Tm = (0.5 * (ds["air_temperature_0m"] + ds["air_temperature_ml"].isel(hybrid=-n)))
            else:
                p_d = (ds["air_pressure_ml"].isel(hybrid=-n+1)/ds["air_pressure_ml"].isel(hybrid=-n))
                Tm = (0.5 * (ds["air_temperature_ml"].isel(hybrid=-n+1) + ds["air_temperature_ml"].isel(hybrid=-n)))
            ds["z"][dict(hybrid=-n)] = ds["z"].isel(hybrid=-n+1) + 287.*Tm/9.81*np.log(p_d)

            
        if "x_wind_ml" in model_varis:
            # Wind u and v components in the original ds are grid-related.
            # Therefore, we rotate here the wind components from grid- to earth-related coordinates.
            cone = np.sin(np.abs(np.deg2rad(ds.projection_lambert.attrs["latitude_of_projection_origin"]))) # cone factor

            ds["diffn"] = ds.projection_lambert.attrs["longitude_of_central_meridian"] - ds.longitude
            ds["diffn"].values[ds["diffn"].values > 180.] -= 360
            ds["diffn"].values[ds["diffn"].values < -180.] += 360

            ds["alpha"]  = np.deg2rad(ds["diffn"]) * cone

            ds["eastward_wind_ml"] = ds['x_wind_ml'] * np.cos(ds["alpha"]) - ds['y_wind_ml'] * np.sin(ds["alpha"])
            ds["northward_wind_ml"] = ds['y_wind_ml'] * np.cos(ds["alpha"]) + ds['x_wind_ml'] * np.sin(ds["alpha"])

            # Calculating wind direction
            ds['wind_direction_ml'] = (np.rad2deg(np.arctan2(-ds['eastward_wind_ml'], -ds['northward_wind_ml']))+360.) % 360.

            # Calculating wind speed
            ds['wind_speed_ml'] = np.sqrt((ds['eastward_wind_ml']**2.) + (ds['northward_wind_ml']**2.))

            del ds['x_wind_ml']
            del ds['y_wind_ml']
            del ds['diffn']
            del ds['alpha']
            
            
        # Calculating potential temperature
        ds['air_potential_temperature_ml'] = (ds['air_temperature_ml'])*((1.e5/(ds['air_pressure_ml']))**(287./1005.))

        if 'specific_humidity_ml' in model_varis:
            T = ds["air_temperature_ml"] - 273.15
            e = (ds['specific_humidity_ml']*ds['air_pressure_ml'])/(0.622 + 0.378*ds['specific_humidity_ml'])
            ds['relative_humidty_ml'] = e/(611.2 * np.exp((17.62*T)/(243.12+T)))
            
        del ds["ap"]
        del ds["b"]
            
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

        data_crs = ccrs.CRS(self.static_fields.projection_lambert.attrs["proj4"])
        xx = []
        yy = []
        for i in range(len(self.stt_lon)):
            x, y = data_crs.transform_point(self.stt_lon[i], self.stt_lat[i], src_crs=ccrs.PlateCarree())
            xx.append(x)
            yy.append(y)
            
        if self.check_plot:
            # figure to check if the right area was selected
            lon_min = np.nanmin([self.static_fields.sel(x=xx[i], y=yy[i], method='nearest')["longitude"] for i in range(len(self.stt_name))])-1.
            lon_max = np.nanmax([self.static_fields.sel(x=xx[i], y=yy[i], method='nearest')["longitude"] for i in range(len(self.stt_name))])+1.
            lat_min = np.nanmin([self.static_fields.sel(x=xx[i], y=yy[i], method='nearest')["latitude"] for i in range(len(self.stt_name))])-0.5
            lat_max = np.nanmax([self.static_fields.sel(x=xx[i], y=yy[i], method='nearest')["latitude"] for i in range(len(self.stt_name))])+0.5
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
                ax.scatter(self.static_fields.sel(x=xx[i], y=yy[i], method='nearest')["longitude"], self.static_fields.sel(x=xx[i], y=yy[i], method='nearest')["latitude"], s=10., c="k", marker="o", transform=ccrs.PlateCarree())
            plt.show(block=False)
            plt.pause(5.)

            print("Type 'Y' to proceed to the download. Any other input will terminate the script immediately.")
            print("")
            switch_continue = input("Your input:  ")

            plt.close("all")

            if switch_continue != "Y":
                sys.exit(1)
                
            print("Starting data download...")
            
            
        chunks = []
        for filename in self.fileurls:
            stations = []
            with xr.open_dataset(filename) as full_file:
                for i in range(len(self.stt_lon)):
                    stations.append(full_file.isel(time=self.time_ind, hybrid=np.arange(-self.model_levels,0,1)).sel(x=xx[i], y=yy[i], method='nearest')[model_varis].squeeze())
                    
            data = xr.concat(stations, dim="station")
                
            for vari in model_varis:
                if vari[:8] == "integral":
                    data[vari][dict(time=range(1,len(data["time"])))] -= data[vari][dict(time=range(0,len(data["time"])-1))].values
                    data[vari][dict(time=0)] /= self.start_h
                    data[vari] /= (3600.*self.int_h)
                    data[vari].attrs["standard_name"] = data[vari].attrs["standard_name"][12:-9]
                    data[vari].attrs["units"] = "W/m^2"
                    data[vari].attrs["long_name"] = data[vari].attrs["long_name"][12:]
                    data = data.rename({vari: vari[12:-9]})
                    
                elif vari[-3:] == "acc":
                    data[vari][dict(time=range(1,len(data["time"])))] -= data[vari][dict(time=range(0,len(data["time"])-1))].values
                    data[vari][dict(time=0)] /= self.start_h
                    data[vari].attrs["long_name"] = data[vari].attrs["long_name"][12:]
                    data = data.rename({vari: vari[:-4]})
                    
            chunks.append(data)
            
            print(f"Done downloading {filename}.")
        
        ds = xr.concat(chunks, dim="time")
        
        ds["stations"] = xr.DataArray(self.stt_name, dims=['station'])

        if "integral_of_surface_net_downward_shortwave_flux_wrt_time" in model_varis:
            # Converting radiative fluxes from net into up
            ds["surface_upwelling_shortwave_flux_in_air"] = ds["surface_downwelling_shortwave_flux_in_air"] - ds["surface_net_downward_shortwave_flux"]
            ds["surface_upwelling_longwave_flux_in_air"] = ds["surface_downwelling_longwave_flux_in_air"] - ds["surface_net_downward_longwave_flux"]

            del ds['surface_net_downward_shortwave_flux']
            del ds['surface_net_downward_longwave_flux']
            
        if "x_wind_10m" in model_varis:
            # Wind u and v components in the original ds are grid-related.
            # Therefore, we rotate here the wind components from grid- to earth-related coordinates.
            cone = np.sin(np.abs(np.deg2rad(ds.projection_lambert.attrs["latitude_of_projection_origin"]))) # cone factor

            ds["diffn"] = ds.projection_lambert.attrs["longitude_of_central_meridian"] - ds.longitude
            ds["diffn"].values[ds["diffn"].values > 180.] -= 360
            ds["diffn"].values[ds["diffn"].values < -180.] += 360

            ds["alpha"]  = np.deg2rad(ds["diffn"]) * cone

            ds["eastward_wind_10m"] = ds['x_wind_10m'] * np.cos(ds["alpha"]) - ds['y_wind_10m'] * np.sin(ds["alpha"])
            ds["northward_wind_10m"] = ds['y_wind_10m'] * np.cos(ds["alpha"]) + ds['x_wind_10m'] * np.sin(ds["alpha"])

            # Calculating wind direction
            ds['wind_direction_10m'] = (np.rad2deg(np.arctan2(-ds['eastward_wind_10m'], -ds['northward_wind_10m']))+360.) % 360.

            # Calculating wind speed
            ds['wind_speed_10m'] = np.sqrt((ds['eastward_wind_10m']**2.) + (ds['northward_wind_10m']**2.))

            del ds['x_wind_10m']
            del ds['y_wind_10m']
            del ds['diffn']
            del ds['alpha']
            
        # Calculating potential temperature
        if (('air_temperature_2m' in model_varis) & ("surface_air_pressure" in model_varis)):
            ds['air_potential_temperature_2m'] = (ds['air_temperature_2m'])*((1.e5/(ds['surface_air_pressure']))**(287./1005.))

            if 'specific_humidity_2m' in model_varis:
                T = ds["air_temperature_2m"] - 273.15
                e = (ds['specific_humidity_2m']*ds['surface_air_pressure'])/(0.622 + 0.378*ds['specific_humidity_2m'])
                ds['relative_humidty_2m'] = (e/(611.2 * np.exp((17.62*T)/(243.12+T))))
            
            
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
        
        data_crs = ccrs.CRS(self.static_fields.projection_lambert.attrs["proj4"])
        xx = []
        yy = []
        x, y = data_crs.transform_point(self.lon_lims[0], self.lat_lims[0], src_crs=ccrs.PlateCarree())
        xx.append(self.static_fields['x'].sel(x=x, method="nearest"))
        yy.append(self.static_fields['y'].sel(y=y, method="nearest"))
        x, y = data_crs.transform_point(self.lon_lims[1], self.lat_lims[0], src_crs=ccrs.PlateCarree())
        xx.append(self.static_fields['x'].sel(x=x, method="nearest"))
        yy.append(self.static_fields['y'].sel(y=y, method="nearest"))
        x, y = data_crs.transform_point(self.lon_lims[1], self.lat_lims[1], src_crs=ccrs.PlateCarree())
        xx.append(self.static_fields['x'].sel(x=x, method="nearest"))
        yy.append(self.static_fields['y'].sel(y=y, method="nearest"))
        x, y = data_crs.transform_point(self.lon_lims[0], self.lat_lims[1], src_crs=ccrs.PlateCarree())
        xx.append(self.static_fields['x'].sel(x=x, method="nearest"))
        yy.append(self.static_fields['y'].sel(y=y, method="nearest"))

        x = np.arange(np.nanmin(xx), (np.nanmax(xx)+1), self.dxdy)
        y = np.arange(np.nanmin(yy), (np.nanmax(yy)+1), self.dxdy)
        x = x[::self.int_x]
        y = y[::self.int_y]
                
        
        if self.check_plot:
            # figure to check if the right area was selected
            plt.close("all")
            plt.ion()
            fig, ax = plt.subplots(1,1,subplot_kw={'projection': ccrs.Mercator()})
            ax.set_extent([x[0]-0.1*abs(x[-1]-x[0]), x[-1]+0.1*abs(x[-1]-x[0]), y[0]-0.1*abs(y[-1]-y[0]), y[-1]+0.1*abs(y[-1]-y[0])], crs=data_crs)
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
            pic = ax.pcolormesh(self.static_fields['longitude'].sel(x=x, y=y), self.static_fields['latitude'].sel(x=x, y=y), (self.static_fields['surface_geopotential'].sel(x=x, y=y))/9.81, cmap=mpl.cm.terrain, shading="nearest", transform=ccrs.PlateCarree())
            cbar = plt.colorbar(pic, ax=ax)
            cbar.ax.set_ylabel("model grid elevation [m]")
            plt.show(block=False)
            plt.pause(5.)

            print("Type 'Y' to proceed to the download. Any other input will terminate the script immediately.")
            print("")
            switch_continue = input("Your input:  ")

            plt.close("all")

            if switch_continue != "Y":
                sys.exit(1)
                
            print("Starting data download...")
            
            
        chunks = []
        for filename in self.fileurls:
            with xr.open_dataset(filename) as full_file:
                data = full_file.isel(time=self.time_ind).sel(x=x, y=y)[model_varis].squeeze()
                
            for vari in model_varis:
                if vari[:8] == "integral":
                    data[vari][dict(time=range(1,len(data["time"])))] -= data[vari][dict(time=range(0,len(data["time"])-1))].values
                    data[vari][dict(time=0)] /= self.start_h
                    data[vari] /= (3600.*self.int_h)
                    data[vari].attrs["standard_name"] = data[vari].attrs["standard_name"][12:-9]
                    data[vari].attrs["units"] = "W/m^2"
                    data[vari].attrs["long_name"] = data[vari].attrs["long_name"][12:]
                    data = data.rename({vari: vari[12:-9]})
                elif vari[-3:] == "acc":
                    data[vari][dict(time=range(1,len(data["time"])))] -= data[vari][dict(time=range(0,len(data["time"])-1))].values
                    data[vari][dict(time=0)] /= self.start_h
                    data[vari].attrs["long_name"] = data[vari].attrs["long_name"][12:]
                    data = data.rename({vari: vari[:-4]})
                        
            chunks.append(data)
            
            print(f"Done downloading {filename}.")
        
        ds = xr.concat(chunks, dim="time")

        if "integral_of_surface_net_downward_shortwave_flux_wrt_time" in model_varis:
            # Converting radiative fluxes from net into up
            ds["surface_upwelling_shortwave_flux_in_air"] = ds["surface_downwelling_shortwave_flux_in_air"] - ds["surface_net_downward_shortwave_flux"]
            ds["surface_upwelling_longwave_flux_in_air"] = ds["surface_downwelling_longwave_flux_in_air"] - ds["surface_net_downward_longwave_flux"]

            del ds['surface_net_downward_shortwave_flux']
            del ds['surface_net_downward_longwave_flux']
            
        if "x_wind_10m" in model_varis:
            # Wind u and v components in the original ds are grid-related.
            # Therefore, we rotate here the wind components from grid- to earth-related coordinates.
            cone = np.sin(np.abs(np.deg2rad(ds.projection_lambert.attrs["latitude_of_projection_origin"]))) # cone factor

            ds["diffn"] = ds.projection_lambert.attrs["longitude_of_central_meridian"] - ds.longitude
            ds["diffn"].values[ds["diffn"].values > 180.] -= 360
            ds["diffn"].values[ds["diffn"].values < -180.] += 360

            ds["alpha"]  = np.deg2rad(ds["diffn"]) * cone

            ds["eastward_wind_10m"] = ds['x_wind_10m'] * np.cos(ds["alpha"]) - ds['y_wind_10m'] * np.sin(ds["alpha"])
            ds["northward_wind_10m"] = ds['y_wind_10m'] * np.cos(ds["alpha"]) + ds['x_wind_10m'] * np.sin(ds["alpha"])

            # Calculating wind direction
            ds['wind_direction_10m'] = (np.rad2deg(np.arctan2(-ds['eastward_wind_10m'], -ds['northward_wind_10m']))+360.) % 360.

            # Calculating wind speed
            ds['wind_speed_10m'] = np.sqrt((ds['eastward_wind_10m']**2.) + (ds['northward_wind_10m']**2.))

            del ds['x_wind_10m']
            del ds['y_wind_10m']
            del ds['diffn']
            del ds['alpha']
            
        # Calculating potential temperature
        if (("air_temperature_2m" in model_varis) & ("surface_air_pressure" in model_varis)):
            ds['air_potential_temperature_2m'] = (ds['air_temperature_2m'])*((1.e5/(ds['surface_air_pressure']))**(287./1005.))

            if 'specific_humidity_2m' in model_varis:
                T = ds["air_temperature_2m"] - 273.15
                e = (ds['specific_humidity_2m']*ds['surface_air_pressure'])/(0.622 + 0.378*ds['specific_humidity_2m'])
                ds['relative_humidty_2m'] = (e/(611.2 * np.exp((17.62*T)/(243.12+T))))
        
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

        data_crs = ccrs.CRS(self.static_fields.projection_lambert.attrs["proj4"])
        xx = []
        yy = []
        x, y = data_crs.transform_point(self.lon_lims[0], self.lat_lims[0], src_crs=ccrs.PlateCarree())
        xx.append(self.static_fields['x'].sel(x=x, method="nearest"))
        yy.append(self.static_fields['y'].sel(y=y, method="nearest"))
        x, y = data_crs.transform_point(self.lon_lims[1], self.lat_lims[0], src_crs=ccrs.PlateCarree())
        xx.append(self.static_fields['x'].sel(x=x, method="nearest"))
        yy.append(self.static_fields['y'].sel(y=y, method="nearest"))
        x, y = data_crs.transform_point(self.lon_lims[1], self.lat_lims[1], src_crs=ccrs.PlateCarree())
        xx.append(self.static_fields['x'].sel(x=x, method="nearest"))
        yy.append(self.static_fields['y'].sel(y=y, method="nearest"))
        x, y = data_crs.transform_point(self.lon_lims[0], self.lat_lims[1], src_crs=ccrs.PlateCarree())
        xx.append(self.static_fields['x'].sel(x=x, method="nearest"))
        yy.append(self.static_fields['y'].sel(y=y, method="nearest"))

        x = np.arange(np.nanmin(xx), (np.nanmax(xx)+1), self.dxdy)
        y = np.arange(np.nanmin(yy), (np.nanmax(yy)+1), self.dxdy)
        x = x[::self.int_x]
        y = y[::self.int_y]
        
        if self.check_plot:
            # figure to check if the right area was selected
            plt.close("all")
            plt.ion()
            fig, ax = plt.subplots(1,1,subplot_kw={'projection': ccrs.Mercator()})
            ax.set_extent([x[0]-0.1*abs(x[-1]-x[0]), x[-1]+0.1*abs(x[-1]-x[0]), y[0]-0.1*abs(y[-1]-y[0]), y[-1]+0.1*abs(y[-1]-y[0])], crs=data_crs)
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
            pic = ax.pcolormesh(self.static_fields['longitude'].sel(x=x, y=y), self.static_fields['latitude'].sel(x=x, y=y), (self.static_fields['surface_geopotential'].sel(x=x, y=y))/9.81, cmap=mpl.cm.terrain, shading="nearest", transform=ccrs.PlateCarree())
            cbar = plt.colorbar(pic, ax=ax)
            cbar.ax.set_ylabel("model grid elevation [m]")
            plt.show(block=False)
            plt.pause(5.)

            print("Type 'Y' to proceed to the download. Any other input will terminate the script immediately.")
            print("")
            switch_continue = input("Your input:  ")

            plt.close("all")

            if switch_continue != "Y":
                sys.exit(1)
                
            print("Starting data download...")
            
            
        chunks = []
        for filename in self.fileurls:
            with xr.open_dataset(filename) as full_file:
                data = full_file.isel(time=self.time_ind, hybrid=np.arange(-self.model_levels,0,1)).sel(x=x, y=y)[model_varis].squeeze()
                
            chunks.append(data)
            
            print(f"Done downloading {filename}.")
        
        ds = xr.concat(chunks, dim="time")
                
                
        ap, b, sp = xr.broadcast(ds["ap"], ds["b"], ds["surface_air_pressure"])
        ds["air_pressure_ml"] = ap + b*sp
        
        ds["z"] = xr.zeros_like(ds["air_pressure_ml"])
        
        for c, n in enumerate(range(1,len(ds["hybrid"])+1)):
            if c == 0:
                p_d = (ds["surface_air_pressure"]/ds["air_pressure_ml"].isel(hybrid=-n))
                Tm = (0.5 * (ds["air_temperature_0m"] + ds["air_temperature_ml"].isel(hybrid=-n)))
            else:
                p_d = (ds["air_pressure_ml"].isel(hybrid=-n+1)/ds["air_pressure_ml"].isel(hybrid=-n))
                Tm = (0.5 * (ds["air_temperature_ml"].isel(hybrid=-n+1) + ds["air_temperature_ml"].isel(hybrid=-n)))
            ds["z"][dict(hybrid=-n)] = ds["z"].isel(hybrid=-n+1) + 287.*Tm/9.81*np.log(p_d)
            
            
        if "x_wind_ml" in model_varis:
            # Wind u and v components in the original ds are grid-related.
            # Therefore, we rotate here the wind components from grid- to earth-related coordinates.
            cone = np.sin(np.abs(np.deg2rad(ds.projection_lambert.attrs["latitude_of_projection_origin"]))) # cone factor

            ds["diffn"] = ds.projection_lambert.attrs["longitude_of_central_meridian"] - ds.longitude
            ds["diffn"].values[ds["diffn"].values > 180.] -= 360
            ds["diffn"].values[ds["diffn"].values < -180.] += 360

            ds["alpha"]  = np.deg2rad(ds["diffn"]) * cone

            ds["eastward_wind_ml"] = ds['x_wind_ml'] * np.cos(ds["alpha"]) - ds['y_wind_ml'] * np.sin(ds["alpha"])
            ds["northward_wind_ml"] = ds['y_wind_ml'] * np.cos(ds["alpha"]) + ds['x_wind_ml'] * np.sin(ds["alpha"])

            # Calculating wind direction
            ds['wind_direction_ml'] = (np.rad2deg(np.arctan2(-ds['eastward_wind_ml'], -ds['northward_wind_ml']))+360.) % 360.

            # Calculating wind speed
            ds['wind_speed_ml'] = np.sqrt((ds['eastward_wind_ml']**2.) + (ds['northward_wind_ml']**2.))

            del ds['x_wind_ml']
            del ds['y_wind_ml']
            del ds['diffn']
            del ds['alpha']
            
            
        # Calculating potential temperature
        if 'air_temperature_ml' in model_varis:
            ds['air_potential_temperature_ml'] = (ds['air_temperature_ml'])*((1.e5/(ds['air_pressure_ml']))**(287./1005.))

            if 'specific_humidity_ml' in model_varis:
                T = ds["air_temperature_ml"] - 273.15
                e = (ds['specific_humidity_ml']*ds['air_pressure_ml'])/(0.622 + 0.378*ds['specific_humidity_ml'])
                ds['relative_humidty_ml'] = (e/(611.2 * np.exp((17.62*T)/(243.12+T))))
            
        del ds["ap"]
        del ds["b"]
        
        ds.to_netcdf(self.out_path)
        
        
        
        return
    
    
    
    def download_vertical_cross_section_data(self):
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


        data_crs = ccrs.CRS(self.static_fields.projection_lambert.attrs["proj4"])
        xx = []
        yy = []
        x, y = data_crs.transform_point(self.start_point[1], self.start_point[0], src_crs=ccrs.PlateCarree())
        xx.append(self.static_fields['x'].sel(x=x, method="nearest"))
        yy.append(self.static_fields['y'].sel(y=y, method="nearest"))
        x, y = data_crs.transform_point(self.end_point[1], self.end_point[0], src_crs=ccrs.PlateCarree())
        xx.append(self.static_fields['x'].sel(x=x, method="nearest"))
        yy.append(self.static_fields['y'].sel(y=y, method="nearest"))

        x = np.arange(np.nanmin(xx)-3*self.dxdy, (np.nanmax(xx)+3*self.dxdy+1), self.dxdy)
        y = np.arange(np.nanmin(yy)-3*self.dxdy, (np.nanmax(yy)+3*self.dxdy+1), self.dxdy)
        x = x[::self.int_x]
        y = y[::self.int_y]
        
        xp, yp = data_crs.transform_point(self.end_point[1], self.start_point[0], src_crs=ccrs.PlateCarree())
        xx.append(self.static_fields['x'].sel(x=xp, method="nearest"))
        yy.append(self.static_fields['y'].sel(y=yp, method="nearest"))
        xp, yp = data_crs.transform_point(self.start_point[1], self.end_point[0], src_crs=ccrs.PlateCarree())
        xx.append(self.static_fields['x'].sel(x=xp, method="nearest"))
        yy.append(self.static_fields['y'].sel(y=yp, method="nearest"))
        
        xp = np.arange(np.nanmin(xx)-3*self.dxdy, (np.nanmax(xx)+3*self.dxdy+1), self.dxdy)
        yp = np.arange(np.nanmin(yy)-3*self.dxdy, (np.nanmax(yy)+3*self.dxdy+1), self.dxdy)
        xp = xp[::self.int_x]
        yp = yp[::self.int_y]
        
        if self.check_plot:
            # figure to check if the right area was selected
            plt.close("all")
            plt.ion()
            fig, ax = plt.subplots(1,1,subplot_kw={'projection': ccrs.Mercator()})
            ax.set_extent([xp[0]-0.1*abs(xp[-1]-xp[0]), xp[-1]+0.1*abs(xp[-1]-xp[0]), yp[0]-0.1*abs(yp[-1]-yp[0]), yp[-1]+0.1*abs(yp[-1]-yp[0])], crs=data_crs)
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
            pic = ax.pcolormesh(self.static_fields['longitude'].sel(x=xp, y=yp, method="nearest"), self.static_fields['latitude'].sel(x=xp, y=yp, method="nearest"), (self.static_fields['surface_geopotential'].sel(x=xp, y=yp, method="nearest"))/9.81, cmap=mpl.cm.terrain, shading="nearest", transform=ccrs.PlateCarree())
            ax.plot([self.start_point[1], self.end_point[1]], [self.start_point[0], self.end_point[0]], 'r-', transform=ccrs.PlateCarree())
            cbar = plt.colorbar(pic, ax=ax)
            cbar.ax.set_ylabel("model grid elevation [m]")
            plt.show(block=False)
            plt.pause(5.)

            print("Type 'Y' to proceed to the download. Any other input will terminate the script immediately.")
            print("")
            switch_continue = input("Your input:  ")

            plt.close("all")

            if switch_continue != "Y":
                sys.exit(1)
                
            print("Starting data download...")
            
            
        chunks = []
        for filename in self.fileurls:
            with xr.open_dataset(filename) as full_file:
                data = full_file.isel(time=self.time_ind, hybrid=np.arange(-self.model_levels,0,1)).sel(x=x, y=y)[model_varis].squeeze()
                
            chunks.append(data)
            
            print(f"Done downloading {filename}.")
        
        ds = xr.concat(chunks, dim="time", data_vars="minimal")
                
        ap, b, sp = xr.broadcast(ds["ap"], ds["b"], ds["surface_air_pressure"])
        ds["air_pressure_ml"] = ap + b*sp
        
        ds["z"] = xr.zeros_like(ds["air_pressure_ml"])
        
        for c, n in enumerate(range(1,len(ds["hybrid"])+1)):
            if c == 0:
                p_d = (ds["surface_air_pressure"]/ds["air_pressure_ml"].isel(hybrid=-n))
                Tm = (0.5 * (ds["air_temperature_0m"] + ds["air_temperature_ml"].isel(hybrid=-n)))
            else:
                p_d = (ds["air_pressure_ml"].isel(hybrid=-n+1)/ds["air_pressure_ml"].isel(hybrid=-n))
                Tm = (0.5 * (ds["air_temperature_ml"].isel(hybrid=-n+1) + ds["air_temperature_ml"].isel(hybrid=-n)))
            ds["z"][dict(hybrid=-n)] = ds["z"].isel(hybrid=-n+1) + 287.*Tm/9.81*np.log(p_d)
            
            
        if "x_wind_ml" in model_varis:
            # Wind u and v components in the original ds are grid-related.
            # Therefore, we rotate here the wind components from grid- to earth-related coordinates.
            cone = np.sin(np.abs(np.deg2rad(ds.projection_lambert.attrs["latitude_of_projection_origin"]))) # cone factor

            ds["diffn"] = ds.projection_lambert.attrs["longitude_of_central_meridian"] - ds.longitude
            ds["diffn"].values[ds["diffn"].values > 180.] -= 360
            ds["diffn"].values[ds["diffn"].values < -180.] += 360

            ds["alpha"]  = np.deg2rad(ds["diffn"]) * cone

            ds["eastward_wind_ml"] = ds['x_wind_ml'] * np.cos(ds["alpha"]) - ds['y_wind_ml'] * np.sin(ds["alpha"])
            ds["northward_wind_ml"] = ds['y_wind_ml'] * np.cos(ds["alpha"]) + ds['x_wind_ml'] * np.sin(ds["alpha"])

            # Calculating wind direction
            ds['wind_direction_ml'] = (np.rad2deg(np.arctan2(-ds['eastward_wind_ml'], -ds['northward_wind_ml']))+360.) % 360.

            # Calculating wind speed
            ds['wind_speed_ml'] = np.sqrt((ds['eastward_wind_ml']**2.) + (ds['northward_wind_ml']**2.))

            del ds['x_wind_ml']
            del ds['y_wind_ml']
            del ds['diffn']
            del ds['alpha']
            
            
        # Calculating potential temperature
        if 'air_temperature_ml' in model_varis:
            ds['air_potential_temperature_ml'] = (ds['air_temperature_ml'])*((1.e5/(ds['air_pressure_ml']))**(287./1005.))

            if 'specific_humidity_ml' in model_varis:
                T = ds["air_temperature_ml"] - 273.15
                e = (ds['specific_humidity_ml']*ds['air_pressure_ml'])/(0.622 + 0.378*ds['specific_humidity_ml'])
                ds['relative_humidty_ml'] = (e/(611.2 * np.exp((17.62*T)/(243.12+T))))
                
        print("Starting cross section interpolations...")
        ds = ds.metpy.parse_cf().squeeze()
        cross_sections = []
        for vari in list(ds.variables):
            if vari not in ["longitude", "latitude"]:
                if (("x" in ds[vari].dims) and ("y" in ds[vari].dims)):
                    print(f"Calculating cross section for {vari}...")
                    cross_sections.append(metpy.interpolate.cross_section(ds[vari], self.start_point, self.end_point))
                
        cross_section = xr.merge(cross_sections, compat='override')
        del cross_section["metpy_crs"]
        attributes = self.static_fields.attrs
        attributes["comment"] = "Cross section data extracted from the full data files."
        cross_section.attrs = attributes
        cross_section["projection_lambert"] = self.static_fields["projection_lambert"]
        cross_section["distance"] = 1.e-3*np.sqrt((cross_section.x-cross_section.x.values[0])**2. + (cross_section.y-cross_section.y.values[0])**2.)
        cross_section = cross_section.set_coords(("distance"))
        cross_section.to_netcdf(self.out_path)
            
        return
    
    
    
    def download_horizontal_near_surface_section_data(self):
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
        
        data_crs = ccrs.CRS(self.static_fields.projection_lambert.attrs["proj4"])
        xx = []
        yy = []
        x, y = data_crs.transform_point(self.start_point[1], self.start_point[0], src_crs=ccrs.PlateCarree())
        xx.append(self.static_fields['x'].sel(x=x, method="nearest"))
        yy.append(self.static_fields['y'].sel(y=y, method="nearest"))
        x, y = data_crs.transform_point(self.end_point[1], self.end_point[0], src_crs=ccrs.PlateCarree())
        xx.append(self.static_fields['x'].sel(x=x, method="nearest"))
        yy.append(self.static_fields['y'].sel(y=y, method="nearest"))

        x = np.arange(np.nanmin(xx)-3*self.dxdy, (np.nanmax(xx)+3*self.dxdy+1), self.dxdy)
        y = np.arange(np.nanmin(yy)-3*self.dxdy, (np.nanmax(yy)+3*self.dxdy+1), self.dxdy)
        x = x[::self.int_x]
        y = y[::self.int_y]
        
        xp, yp = data_crs.transform_point(self.end_point[1], self.start_point[0], src_crs=ccrs.PlateCarree())
        xx.append(self.static_fields['x'].sel(x=xp, method="nearest"))
        yy.append(self.static_fields['y'].sel(y=yp, method="nearest"))
        xp, yp = data_crs.transform_point(self.start_point[1], self.end_point[0], src_crs=ccrs.PlateCarree())
        xx.append(self.static_fields['x'].sel(x=xp, method="nearest"))
        yy.append(self.static_fields['y'].sel(y=yp, method="nearest"))
        
        xp = np.arange(np.nanmin(xx)-3*self.dxdy, (np.nanmax(xx)+3*self.dxdy+1), self.dxdy)
        yp = np.arange(np.nanmin(yy)-3*self.dxdy, (np.nanmax(yy)+3*self.dxdy+1), self.dxdy)
        xp = xp[::self.int_x]
        yp = yp[::self.int_y]
        
        if self.check_plot:
            # figure to check if the right area was selected
            plt.close("all")
            plt.ion()
            fig, ax = plt.subplots(1,1,subplot_kw={'projection': ccrs.Mercator()})
            ax.set_extent([xp[0]-0.1*abs(xp[-1]-xp[0]), xp[-1]+0.1*abs(xp[-1]-xp[0]), yp[0]-0.1*abs(yp[-1]-yp[0]), yp[-1]+0.1*abs(yp[-1]-yp[0])], crs=data_crs)
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
            pic = ax.pcolormesh(self.static_fields['longitude'].sel(x=xp, y=yp, method="nearest"), self.static_fields['latitude'].sel(x=xp, y=yp, method="nearest"), (self.static_fields['surface_geopotential'].sel(x=xp, y=yp, method="nearest"))/9.81, cmap=mpl.cm.terrain, shading="nearest", transform=ccrs.PlateCarree())
            ax.plot([self.start_point[1], self.end_point[1]], [self.start_point[0], self.end_point[0]], 'r-', transform=ccrs.PlateCarree())
            cbar = plt.colorbar(pic, ax=ax)
            cbar.ax.set_ylabel("model grid elevation [m]")
            plt.show(block=False)
            plt.pause(5.)

            print("Type 'Y' to proceed to the download. Any other input will terminate the script immediately.")
            print("")
            switch_continue = input("Your input:  ")

            plt.close("all")

            if switch_continue != "Y":
                sys.exit(1)
                
            print("Starting data download...")
            
            
            
        chunks = []
        for filename in self.fileurls:
            with xr.open_dataset(filename) as full_file:
                data = full_file.isel(time=self.time_ind).sel(x=x, y=y)[model_varis].squeeze()
                
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
                        
            chunks.append(data)
            
            print(f"Done downloading {filename}.")
        
        ds = xr.concat(chunks, dim="time")

        if "integral_of_surface_net_downward_shortwave_flux_wrt_time" in model_varis:
            # Converting radiative fluxes from net into up
            ds["surface_upwelling_shortwave_flux_in_air"] = ds["surface_downwelling_shortwave_flux_in_air"] - ds["surface_net_downward_shortwave_flux"]
            ds["surface_upwelling_longwave_flux_in_air"] = ds["surface_downwelling_longwave_flux_in_air"] - ds["surface_net_downward_longwave_flux"]

            del ds['surface_net_downward_shortwave_flux']
            del ds['surface_net_downward_longwave_flux']
            
        if "x_wind_10m" in model_varis:
            # Wind u and v components in the original ds are grid-related.
            # Therefore, we rotate here the wind components from grid- to earth-related coordinates.
            cone = np.sin(np.abs(np.deg2rad(ds.projection_lambert.attrs["latitude_of_projection_origin"]))) # cone factor

            ds["diffn"] = ds.projection_lambert.attrs["longitude_of_central_meridian"] - ds.longitude
            ds["diffn"].values[ds["diffn"].values > 180.] -= 360
            ds["diffn"].values[ds["diffn"].values < -180.] += 360

            ds["alpha"]  = np.deg2rad(ds["diffn"]) * cone

            ds["eastward_wind_10m"] = ds['x_wind_10m'] * np.cos(ds["alpha"]) - ds['y_wind_10m'] * np.sin(ds["alpha"])
            ds["northward_wind_10m"] = ds['y_wind_10m'] * np.cos(ds["alpha"]) + ds['x_wind_10m'] * np.sin(ds["alpha"])

            # Calculating wind direction
            ds['wind_direction_10m'] = (np.rad2deg(np.arctan2(-ds['eastward_wind_10m'], -ds['northward_wind_10m']))+360.) % 360.

            # Calculating wind speed
            ds['wind_speed_10m'] = np.sqrt((ds['eastward_wind_10m']**2.) + (ds['northward_wind_10m']**2.))

            del ds['x_wind_10m']
            del ds['y_wind_10m']
            del ds['diffn']
            del ds['alpha']
            
        # Calculating potential temperature
        if (("air_temperature_2m" in model_varis) & ("surface_air_pressure" in model_varis)):
            ds['air_potential_temperature_2m'] = (ds['air_temperature_2m'])*((1.e5/(ds['surface_air_pressure']))**(287./1005.))

            if 'specific_humidity_2m' in model_varis:
                T = ds["air_temperature_2m"] - 273.15
                e = (ds['specific_humidity_2m']*ds['surface_air_pressure'])/(0.622 + 0.378*ds['specific_humidity_2m'])
                ds['relative_humidty_2m'] = (e/(611.2 * np.exp((17.62*T)/(243.12+T))))
                
            
        print("Starting cross section interpolations...")
        ds = ds.metpy.parse_cf().squeeze()
        cross_sections = []
        for vari in list(ds.variables):
            if vari not in ["longitude", "latitude"]:
                if (("x" in ds[vari].dims) and ("y" in ds[vari].dims)):
                    print(f"Calculating cross section for {vari}...")
                    cross_sections.append(metpy.interpolate.cross_section(ds[vari], self.start_point, self.end_point))
                
        cross_section = xr.merge(cross_sections, compat='override')
        del cross_section["metpy_crs"]
        attributes = self.static_fields.attrs
        attributes["comment"] = "Cross section data extracted from the full data files."
        cross_section.attrs = attributes
        cross_section["projection_lambert"] = self.static_fields["projection_lambert"]
        cross_section["distance"] = 1.e-3*np.sqrt((cross_section.x-cross_section.x.values[0])**2. + (cross_section.y-cross_section.y.values[0])**2.)
        cross_section = cross_section.set_coords(("distance"))
        cross_section.to_netcdf(self.out_path)
        
        cross_section.to_netcdf(self.out_path)
        
        
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

        data_crs = ccrs.CRS(self.static_fields.projection_lambert.attrs["proj4"])
        xx = []
        yy = []
        x, y = data_crs.transform_point(self.lon_lims[0], self.lat_lims[0], src_crs=ccrs.PlateCarree())
        xx.append(self.static_fields['x'].sel(x=x, method="nearest"))
        yy.append(self.static_fields['y'].sel(y=y, method="nearest"))
        x, y = data_crs.transform_point(self.lon_lims[1], self.lat_lims[0], src_crs=ccrs.PlateCarree())
        xx.append(self.static_fields['x'].sel(x=x, method="nearest"))
        yy.append(self.static_fields['y'].sel(y=y, method="nearest"))
        x, y = data_crs.transform_point(self.lon_lims[1], self.lat_lims[1], src_crs=ccrs.PlateCarree())
        xx.append(self.static_fields['x'].sel(x=x, method="nearest"))
        yy.append(self.static_fields['y'].sel(y=y, method="nearest"))
        x, y = data_crs.transform_point(self.lon_lims[0], self.lat_lims[1], src_crs=ccrs.PlateCarree())
        xx.append(self.static_fields['x'].sel(x=x, method="nearest"))
        yy.append(self.static_fields['y'].sel(y=y, method="nearest"))

        x = np.arange(np.nanmin(xx), (np.nanmax(xx)+1), self.dxdy)
        y = np.arange(np.nanmin(yy), (np.nanmax(yy)+1), self.dxdy)
        x = x[::self.int_x]
        y = y[::self.int_y]
        
        if self.check_plot:
            # figure to check if the right area was selected
            plt.close("all")
            plt.ion()
            fig, ax = plt.subplots(1,1,subplot_kw={'projection': ccrs.Mercator()})
            ax.set_extent([x[0]-0.1*abs(x[-1]-x[0]), x[-1]+0.1*abs(x[-1]-x[0]), y[0]-0.1*abs(y[-1]-y[0]), y[-1]+0.1*abs(y[-1]-y[0])], crs=data_crs)
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
            pic = ax.pcolormesh(self.static_fields['longitude'].sel(x=x, y=y), self.static_fields['latitude'].sel(x=x, y=y), (self.static_fields['surface_geopotential'].sel(x=x, y=y))/9.81, cmap=mpl.cm.terrain, shading="nearest", transform=ccrs.PlateCarree())
            cbar = plt.colorbar(pic, ax=ax)
            cbar.ax.set_ylabel("model grid elevation [m]")
            plt.show(block=False)
            plt.pause(5.)

            print("Type 'Y' to proceed to the download. Any other input will terminate the script immediately.")
            print("")
            switch_continue = input("Your input:  ")

            plt.close("all")

            if switch_continue != "Y":
                sys.exit(1)
                
            print("Starting data download...")
                
                
        chunks = []
        for filename in self.fileurls:
            with xr.open_dataset(filename) as full_file:
                data = full_file.isel(time=self.time_ind, hybrid=ind_p_levels).sel(x=x, y=y)[model_varis].squeeze()
                
            chunks.append(data)
            
            print(f"Done downloading {filename}.")
        
        ds = xr.concat(chunks, dim="time")
                
        if "x_wind_pl" in model_varis:
            # Wind u and v components in the original ds are grid-related.
            # Therefore, we rotate here the wind components from grid- to earth-related coordinates.
            cone = np.sin(np.abs(np.deg2rad(ds.projection_lambert.attrs["latitude_of_projection_origin"]))) # cone factor

            ds["diffn"] = ds.projection_lambert.attrs["longitude_of_central_meridian"] - ds.longitude
            ds["diffn"].values[ds["diffn"].values > 180.] -= 360
            ds["diffn"].values[ds["diffn"].values < -180.] += 360

            ds["alpha"]  = np.deg2rad(ds["diffn"]) * cone

            ds["eastward_wind_pl"] = ds['x_wind_pl'] * np.cos(ds["alpha"]) - ds['y_wind_pl'] * np.sin(ds["alpha"])
            ds["northward_wind_pl"] = ds['y_wind_pl'] * np.cos(ds["alpha"]) + ds['x_wind_pl'] * np.sin(ds["alpha"])

            # Calculating wind direction
            ds['wind_direction_pl'] = (np.rad2deg(np.arctan2(-ds['eastward_wind_pl'], -ds['northward_wind_pl']))+360.) % 360.

            # Calculating wind speed
            ds['wind_speed_pl'] = np.sqrt((ds['eastward_wind_pl']**2.) + (ds['northward_wind_pl']**2.))

            del ds['x_wind_pl']
            del ds['y_wind_pl']
            del ds['diffn']
            del ds['alpha']
        
        ds.to_netcdf(self.out_path)
        
        
        return
    
    
    
        
        
        
        

        

    
