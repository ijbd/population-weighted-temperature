import os
import netCDF4 as cdf
import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point
from datetime import datetime, timedelta
import argparse

import matplotlib.pyplot as plt
from matplotlib import colors

CONTROL_AREAS_FILE = "control-areas-shp"
CONUS_DIR = "conus-temperature-data"
POPULATION_DIR = "population-data"
POPULATION_FILE = "gpw_v4_population_density_adjusted_rev11_2pt5_min.nc"
OUTPUT_DIR = "output"
GALLERY_DIR = "gallery"

THIS_PATH = os.path.dirname(__file__)
PROJ_PATH = os.path.join(THIS_PATH, "..")

BAL_AUTH_SHAPE_DATA_PATH = os.path.join(PROJ_PATH, CONTROL_AREAS_FILE)
TEMP_DATA_DIR_PATH = os.path.join(PROJ_PATH, CONUS_DIR) # path to folder
POP_DATA_PATH = os.path.join(PROJ_PATH, POPULATION_DIR, POPULATION_FILE)
OUTPUT_DIR_PATH = os.path.join(PROJ_PATH, OUTPUT_DIR)
GALLERY_DIR_PATH = os.path.join(PROJ_PATH, GALLERY_DIR)

YEARS = [2016, 2017, 2018, 2019]

NET_CDF_VARIABLE_NAMES = {"population" : "UN WPP-Adjusted Population Density, v4.11 (2000, 2005, 2010, 2015, 2020): 2.5 arc-minutes",
							"temperature" : "T2M"}

NET_CDF_POP_DENSITY_LAYER = {2000 : 0, # see population-data/gpw_v4_netcdf_contents_rev11.csv
							2005 : 1, 
							2010 : 2, 
							2015 : 3, 
							2020 : 4} 

BAL_AUTH_NAMES = { "MISO" : "MIDCONTINENT INDEPENDENT TRANSMISSION SYSTEM OPERATOR, INC..",
	"SWPP" : "SOUTHWEST POWER POOL", 
	"WACM" : "WESTERN AREA POWER ADMINISTRATION - ROCKY MOUNTAIN REGION",
	"BPAT" : "BONNEVILLE POWER ADMINISTRATION",
	"WALC" : "WESTERN AREA POWER ADMINISTRATION - DESERT SOUTHWEST REGION",
	"PACE" : "PACIFICORP - EAST",
	"PJM" : "PJM INTERCONNECTION, LLC",
	"ERCO" : "ELECTRIC RELIABILITY COUNCIL OF TEXAS, INC.",
	"NWMT" : "NORTHWESTERN ENERGY (NWMT)",
	"NEVP" : "NEVADA POWER COMPANY",
	"CISO" : "CALIFORNIA INDEPENDENT SYSTEM OPERATOR",
	"SOCO" : "SOUTHERN COMPANY SERVICES, INC. - TRANS",
	"PACW" : "PACIFICORP - WEST",
	"IPCO" : "IDAHO POWER COMPANY",
	"PNM" : "PUBLIC SERVICE COMPANY OF NEW MEXICO",
	"AECI" : "ASSOCIATED ELECTRIC COOPERATIVE, INC.",
	"TVA" : "TENNESSEE VALLEY AUTHORITY",
	"ISNE" : "ISO NEW ENGLAND INC.",
	"EPE" : "EL PASO ELECTRIC COMPANY",
	"WAUW" : "WESTERN AREA POWER ADMINISTRATION UGP WEST",
	"NYIS" : "NEW YORK INDEPENDENT SYSTEM OPERATOR",
	"AVA" : "AVISTA CORPORATION",
	"AZPS" : "ARIZONA PUBLIC SERVICE COMPANY",
	"PSCO" : "PUBLIC SERVICE COMPANY OF COLORADO",
	"CPLE" : "DUKE ENERGY PROGRESS EAST",
	"DUK" : "DUKE ENERGY CAROLINAS",
	# "CHUGACH ELECTRIC ASSN INC"
	"FPC" : "DUKE ENERGY FLORIDA INC",
	"AEC" : "POWERSOUTH ENERGY COOPERATIVE",
	"FPL" : "FLORIDA POWER & LIGHT COMPANY",
	"SEC" : "SEMINOLE ELECTRIC COOPERATIVE",
	"SC" : "SOUTH CAROLINA PUBLIC SERVICE AUTHORITY",
	"PSEI" : "PUGET SOUND ENERGY",
	"SRP" : "SALT RIVER PROJECT",
	"TPWR" : "CITY OF TACOMA, DEPARTMENT OF PUBLIC UTILITIES, LIGHT DIVISION",
	"TEPC" : "TUCSON ELECTRIC POWER COMPANY",
	"IID" : "IMPERIAL IRRIGATION DISTRICT",
	"DOPD" : "PUD NO. 1 OF DOUGLAS COUNTY",
	"LGEE" : "LOUISVILLE GAS AND ELECTRIC COMPANY AND KENTUCKY UTILITIES",
	"FMPP" : "FLORIDA MUNICIPAL POWER POOL",
	"CPLW" : "DUKE ENERGY PROGRESS WEST",
	"BANC" : "BALANCING AUTHORITY OF NORTHERN CALIFORNIA",
	"PGE" : "PORTLAND GENERAL ELECTRIC COMPANY",
	"GWA" : "NATURENER POWER WATCH, LLC (GWA)",
	"JEA" : "JEA",
	"CHPD" : "PUBLIC UTILITY DISTRICT NO. 1 OF CHELAN COUNTY",
	"GCPD" : "PUBLIC UTILITY DISTRICT NO. 2 OF GRANT COUNTY, WASHINGTON",
	# "ANCHORAGE MUNICIPAL LIGHT & POWER"
	"LDWP" : "LOS ANGELES DEPARTMENT OF WATER AND POWER",
	"HST" : "CITY OF HOMESTEAD",
	"HGMA" : "NEW HARQUAHALA GENERATING COMPANY, LLC - HGBA",
	"TEC" : "TAMPA ELECTRIC COMPANY",
	"GRMA" : "GILA RIVER POWER, LLC",
	"WWA" : "NATURENER WIND WATCH, LLC",
	# "GRIDFORCE SOUTH"
	"SEPA" : "SOUTHEASTERN POWER ADMINISTRATION",
	"NSB" : "NEW SMYRNA BEACH, UTILITIES COMMISSION OF",
	"GVL" : "GAINESVILLE REGIONAL UTILITIES",
	"TIDC" : "TURLOCK IRRIGATION DISTRICT",
	# "HAWAIIAN ELECTRIC CO INC"
	"EEI" : "ELECTRIC ENERGY, INC.",
	"GRID" : "GRIDFORCE ENERGY MANAGEMENT, LLC",
	"OVEC" : "OHIO VALLEY ELECTRIC CORPORATION",
	"GRIF" : "GRIFFITH ENERGY, LLC",
	"YAD" : "ALCOA POWER GENERATING, INC. - YADKIN DIVISION", 
	"SCL" : "SEATTLE CITY LIGHT",
	"TAL" : "CITY OF TALLAHASSEE",
	"DEAA" : "ARLINGTON VALLEY, LLC - AVBA",
	"NBSO" : "NEW BRUNSWICK SYSTEM OPERATOR",
	"SCEG" : "SOUTH CAROLINA ELECTRIC & GAS COMPANY"}

def load_cdf_data(filepath: str) -> cdf.Dataset:
	return cdf.Dataset(filepath)

def load_shape_data(filepath: str) -> gpd.GeoDataFrame:
	return gpd.read_file(filepath)

def get_temp_data_path(dir_path: str, year: int) -> str:

	temp_data_file = f"processedMERRAconus{year}-rstr.nc"

	return os.path.join(dir_path, temp_data_file)

def extract_temp(temp_data: cdf.Dataset) -> tuple:

	lats = temp_data.variables["lat"][:]
	lons = temp_data.variables["lon"][:]
	temp = temp_data.variables[NET_CDF_VARIABLE_NAMES["temperature"]][:]

	return temp, lats, lons

def extract_pop_density(pop_data: cdf.Dataset, pop_year: int) -> tuple:
	
	lats = pop_data.variables["latitude"][:]
	lons = pop_data.variables["longitude"][:]
	 
	pop = pop_data.variables[NET_CDF_VARIABLE_NAMES["population"]][NET_CDF_POP_DENSITY_LAYER[pop_year],:,:]

	# fill masked values with zero (matches expected behavior)
	pop = np.where(pop.mask, 0, pop)

	return pop, lats, lons

def select_bal_auth_shape(bal_auth_code: str, shape_data: gpd.GeoDataFrame) -> gpd.GeoDataFrame:

	return shape_data[shape_data["NAME"] == BAL_AUTH_NAMES[bal_auth_code]]

def get_bounds(shape: gpd.GeoDataFrame, tolerance: float=.75) -> tuple:
	''' include tolerance to search nearby coordinates outside of bal_auth for mapping temp. to pop.'''

	bounds = shape.total_bounds
	min_lon = round(bounds[0],3) - tolerance
	min_lat = round(bounds[1],3) - tolerance
	max_lon = round(bounds[2],3) + tolerance
	max_lat = round(bounds[3],3) + tolerance

	return min_lon, max_lon, min_lat, max_lat
	
def convert_to_lat_lon(shape: gpd.GeoDataFrame) -> gpd.GeoDataFrame:

	return shape.to_crs("EPSG:4326")

def crop_data(data: np.ndarray, lats: np.ndarray, lons: np.ndarray, min_lon: float, max_lon: float, min_lat:float, max_lat:float) -> tuple:

	lats_idx = np.where(np.logical_and(lats >= min_lat, lats <= max_lat))
	lons_idx = np.where(np.logical_and(lons >= min_lon, lons <= max_lon))

	min_lat_idx = lats_idx[0][0] # lower index
	max_lat_idx = lats_idx[0][-1] + 1 # upper index
	min_lon_idx = lons_idx[0][0] # lower index
	max_lon_idx = lons_idx[0][-1] + 1 # upper index

	new_lats = lats[min_lat_idx:max_lat_idx]
	new_lons = lons[min_lon_idx:max_lon_idx]
	new_data = data[min_lat_idx:max_lat_idx, min_lon_idx:max_lon_idx]
	
	return new_data, new_lats, new_lons

def find_nearest_idx(coord: float, coord_map: np.ndarray) -> int:

	return np.argmin(abs(coord-coord_map))

def map_high_to_low(arr: np.ndarray, high_res_lats: np.ndarray, high_res_lons: np.ndarray, low_res_lats: np.ndarray, low_res_lons: np.ndarray, ) -> np.ndarray:
	
	assert arr.shape[0] == len(high_res_lats)
	assert arr.shape[1] == len(high_res_lons)

	accum = np.zeros((len(low_res_lats), len(low_res_lons)))
	count = np.zeros((len(low_res_lats), len(low_res_lons)))

	for lat_idx, lat in enumerate(high_res_lats):
		for lon_idx, lon in enumerate(high_res_lons):

			lat_low_res_idx = find_nearest_idx(lat, low_res_lats)
			lon_low_res_idx = find_nearest_idx(lon, low_res_lons)

			assert abs(lat - low_res_lats[lat_low_res_idx]) < .5
			assert abs(lon - low_res_lons[lon_low_res_idx]) < .625

			accum[lat_low_res_idx, lon_low_res_idx] += arr[lat_idx, lon_idx]
			count[lat_low_res_idx, lon_low_res_idx] += 1

		print(f"\t\tMapped {lat_idx} of {len(high_res_lats)}...")

	mapped = accum / count
	
	return mapped

def map_bal_auth_to_coordinates(bal_auth_shape: gpd.GeoDataFrame, lats: np.ndarray, lons: np.ndarray) -> np.ndarray:
	
	bal_auth_mapped = np.zeros((len(lats), len(lons)))
	
	for lat_idx, lat in enumerate(lats):
		for lon_idx, lon in enumerate(lons):
			if bal_auth_shape.contains(Point(lon,lat)).any():
				bal_auth_mapped[lat_idx,lon_idx] = 1

		print(f"\t\tMapped {lat_idx} of {len(lats)}...")
			
	return bal_auth_mapped

def calc_weighted_average(arr: np.ndarray, weights: np.ndarray):
	
	arr = arr.transpose((2,0,1)) # reshape for broadcasted multiply
	weighted_average = np.sum(np.multiply(arr, weights), axis=(1,2)) / np.sum(weights)

	return weighted_average

def get_8760_dt_index(year: int):
	
	dt_arr = np.arange(datetime(year,1,1),datetime(year+1,1,1), timedelta(hours=1))

	index = pd.DatetimeIndex(dt_arr)

	# remove leap days
	index = index[~((index.month == 2) & (index.day == 29))]

	return index

def make_df(hourly_data: np.ndarray, year: int) -> pd.DataFrame:

	hourly_index = get_8760_dt_index(year)

	return pd.DataFrame(hourly_data, hourly_index, columns=["Temperature (K)"])

def save_output(df: pd.DataFrame, output_file: str) -> None:

	df.round(2).to_csv(output_file)

	return None

def plot_temp(temp, lats, lons, fname):

	fig, ax = plt.subplots()

	temp_c = np.mean(temp,axis=2) - 273

	pos = ax.imshow(temp_c, origin="lower")
	ax.grid(False)
	cbar = fig.colorbar(pos, ax=ax, extend="both")
	cbar.set_label("Mean Temperature ($\\degree$C)")

	steps = max(len(lats),len(lons),5)//5
	ax.set_xticks(np.arange(len(lons))[::steps])
	ax.set_xticklabels(lons[::steps].astype(int))
	ax.set_yticks(np.arange(len(lats))[::steps])
	ax.set_yticklabels(lats[::steps].astype(int))

	ax.set_xlabel("Longitude ($\degree$)")
	ax.set_ylabel("Latitude ($\degree$)")

	plt.savefig(fname)
	plt.close()

	return None

def plot_pop(pop, lats, lons, fname):

	fig, ax = plt.subplots()

	pos = ax.imshow(np.where(pop, pop, np.nan), norm=colors.LogNorm(), origin="lower")
	ax.grid(False)
	cbar = fig.colorbar(pos, ax=ax, extend="both")
	cbar.set_label("Population Density (Persons per km$^2$)")

	steps = max(len(lats),len(lons),10)//5
	ax.set_xticks(np.arange(len(lons))[::steps])
	ax.set_xticklabels(lons[::steps].round(1))
	ax.set_yticks(np.arange(len(lats))[::steps])
	ax.set_yticklabels(lats[::steps].round(1))

	ax.set_xlabel("Longitude ($\degree$)")
	ax.set_ylabel("Latitude ($\degree$)")

	plt.savefig(fname)
	plt.close()

	return None

def plot_weighted_unweighted(weighted_temp, unweighted_temp, centroid_temp, fname):

	fig, (scatter_ax, series_ax) = plt.subplots(2, 1, figsize=(7,10))

	scatter_ax.scatter(weighted_temp, unweighted_temp, s=1)
	scatter_ax.set_xlabel("Population Weighted Temperature ($\degree$C)")
	scatter_ax.set_ylabel("Unweighted Temperature ($\degree$C)")
	
	hrs_to_plot = 24*5
	peak_hour = np.argmax(weighted_temp) // (hrs_to_plot)
	series_ax.plot(weighted_temp.iloc[peak_hour:peak_hour+hrs_to_plot])
	series_ax.plot(unweighted_temp.iloc[peak_hour:peak_hour+hrs_to_plot])
	series_ax.plot(centroid_temp.iloc[peak_hour:peak_hour+hrs_to_plot])

	series_ax.set_xlim([weighted_temp.index[peak_hour], weighted_temp.index[peak_hour+hrs_to_plot]])

	series_ax.set_ylabel("Temperature ($\degree C$)")
	series_ax.legend(["Weighted", "Unweighted", "Centroid"])

	plt.tight_layout()
	plt.savefig(fname)
	plt.close()

	return None

def main(bal_auth_code: str, pop_year: int):
	
	# balancing authority shape data
	bal_auth_shape_data = load_shape_data(BAL_AUTH_SHAPE_DATA_PATH)
	bal_auth_shape_espg_3857 = select_bal_auth_shape(bal_auth_code, bal_auth_shape_data)
	bal_auth_shape = convert_to_lat_lon(bal_auth_shape_espg_3857)
	
	print(f"Found balancing authority: {bal_auth_code}...")

	# find bounds 
	min_lon, max_lon, min_lat, max_lat = get_bounds(bal_auth_shape)

	# get lat/lons (actual temperature data is loaded later)
	sample_data_year = YEARS[0]
	sample_temp_dataset = load_cdf_data(get_temp_data_path(TEMP_DATA_DIR_PATH, sample_data_year))
	sample_temp, lats, lons = extract_temp(sample_temp_dataset)
	sample_temp, lats, lons = crop_data(sample_temp, lats, lons, min_lon, max_lon, min_lat, max_lat)
	
	# population data	
	pop_dataset = load_cdf_data(POP_DATA_PATH)
	pop, pop_lats, pop_lons = extract_pop_density(pop_dataset, pop_year)
	pop, pop_lats, pop_lons = crop_data(pop, pop_lats, pop_lons, min_lon, max_lon, min_lat, max_lat)
	pop_mapped = map_high_to_low(pop, pop_lats, pop_lons, lats, lons)
	
	print("Found population data...")

	# map balancing authority to pop array
	bal_auth_map = map_bal_auth_to_coordinates(bal_auth_shape, lats, lons)

	# if no coordinates are in balancing authority, use the centroid coordinate
	if np.sum(bal_auth_map) == 0: 
		centroid_x = bal_auth_map.shape[0] // 2
		centroid_y = bal_auth_map.shape[1] // 2

		bal_auth_map[centroid_x, centroid_y] = 1

	# filter population by balancing authority
	bal_auth_pop = np.where(bal_auth_map, pop_mapped, 0)

	# if no available population data is available, use the centroid coordinate
	if np.sum(bal_auth_pop) == 0:
		centroid_x = bal_auth_pop.shape[0] // 2
		centroid_y = bal_auth_pop.shape[1] // 2
		bal_auth_pop[centroid_x, centroid_y] = np.average(pop_mapped)

	# temperature data (yearly)
	pop_weighted_temp_df = pd.DataFrame()
	unweighted_temp_df = pd.DataFrame()
	centroid_temp_df = pd.DataFrame()

	for year in YEARS:

		print(f"Running year: {year}...")

		temp_dataset = load_cdf_data(get_temp_data_path(TEMP_DATA_DIR_PATH, year))
		temp, temp_lats, temp_lons = extract_temp(temp_dataset)
		
		print("\tFound temperature data...")

		# plot
		temp, temp_lats, temp_lons = crop_data(temp, temp_lats, temp_lons, min_lon, max_lon, min_lat, max_lat)	

		assert np.array_equal(lats, temp_lats)
		assert np.array_equal(lons, temp_lons)

		print("\tMapped temperature to population...")

		# weighted average temp
		pop_weighted_temp = calc_weighted_average(temp, bal_auth_pop)
		pop_weighted_temp_df = pd.concat((pop_weighted_temp_df, make_df(pop_weighted_temp, year)))

		print("\tCalculated weighted average...")

		# unweighted_average_temp
		unweighted_temp = calc_weighted_average(temp, bal_auth_map)
		unweighted_temp_df = pd.concat((unweighted_temp_df, make_df(unweighted_temp, year)))

		# centroid temp
		centroid_x = temp.shape[0]//2
		centroid_y = temp.shape[1]//2
		centroid_temp = temp[centroid_x, centroid_y]
		centroid_temp_df = pd.concat((centroid_temp_df, make_df(centroid_temp, year)))

	# save
	pop_weighted_output_file = f"{bal_auth_code.lower()}-temperature-{pop_year}-pop.csv"
	unweighted_output_file = f"{bal_auth_code.lower()}-temperature-unweighted.csv"
	centroid_output_file = f"{bal_auth_code.lower()}-temperature-centroid.csv"

	pop_weighted_output_path = os.path.join(OUTPUT_DIR_PATH, pop_weighted_output_file)
	unweighted_output_path = os.path.join(OUTPUT_DIR_PATH, unweighted_output_file)
	centroid_output_path = os.path.join(OUTPUT_DIR_PATH, centroid_output_file)

	save_output(pop_weighted_temp_df, pop_weighted_output_path)
	save_output(unweighted_temp_df, unweighted_output_path)
	save_output(centroid_temp_df, centroid_output_path)

	# plot
	plt.style.use("ggplot")

	pop_plot_file = f"{bal_auth_code.lower()}-pop-map-{pop_year}.png"
	weighted_plot_file = f"{bal_auth_code.lower()}-pop-weighted-temp-{pop_year}-pop.png"
	temp_plot_file = f"{bal_auth_code.lower()}-temp-map.png"

	pop_plot_path = os.path.join(GALLERY_DIR_PATH, pop_plot_file)
	weighted_plot_path = os.path.join(GALLERY_DIR_PATH, weighted_plot_file)
	temp_plot_path = os.path.join(GALLERY_DIR_PATH, temp_plot_file)

	plot_pop(bal_auth_pop, lats, lons, pop_plot_path)
	plot_weighted_unweighted(pop_weighted_temp_df, unweighted_temp_df, centroid_temp_df, weighted_plot_path)
	plot_temp(temp, lats, lons, temp_plot_path)

	return None

if __name__ == "__main__":
	
	parser = argparse.ArgumentParser()
	parser.add_argument("bal_auth_code", type=str, choices=[code for code in BAL_AUTH_NAMES])
	parser.add_argument("pop_year", type=int, choices=[year for year in NET_CDF_POP_DENSITY_LAYER], help="Populatin data year.")

	args = parser.parse_args()

	main(args.bal_auth_code, args.pop_year)
