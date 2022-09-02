__version__ = '0.1.1'
__authors__ = ['Lukas Frank <lukasf@unis.no', 'Jakob Dörr <jakob.dorr@uib.no']

from .Meteo import (
	read_MET_AWS,
	read_Campbell_AWS,
	read_Campbell_radiation,
	read_Irgason_flux,
	read_CSAT3_flux,
	read_Tinytag,
	read_HOBO,
	read_IWIN,
	read_AROME,
	initialize_empty_map,
	map_add_coastline,
	map_add_land_filled,
	map_add_bathymetry,
	map_add_total_topography,
	map_add_topography,
	map_add_surface_cover,
	map_add_crosssection_line,
	map_add_points,
	map_add_wind_arrows
)

from .Ocean import (
	cal_dist_dir_on_sphere,
	cart2pol,
	pol2cart,
	create_latlon_text,
	CTD_to_grid,
	mooring_to_grid,
	calc_freshwater_content,
	myloadmat,
	mat2py_time,
	present_dict,
	read_ADCP_UNIS,
	read_CTD,
	read_CTD_from_mat,
	read_mini_CTD,
	read_MSS,
	read_mooring_from_mat,
	read_mooring,
	contour_section,
	plot_CTD_section,
	plot_CTD_single_section,
	plot_CTD_station,
	plot_CTD_map,
	plot_empty_map,
	plot_CTD_ts,
	create_empty_ts,
	plot_ADCP_CTD_section
)

from .MET_model_tools import Rotate_uv_components, lonlat2xy, Calculate_height_levels_and_pressure
from .MET_model_download import download_MET_model_data, download_MET_model_static_fields, MET_model_download_class
