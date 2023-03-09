'''
Script for generating hourly population-weighted temperature data using 
GPW population data and reanalysis temperature data.

author: ijbd
'''
import logging
import argparse
from pathlib import Path
from datetime import datetime, timedelta

# external
import numpy as np
import xarray as xr
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point

import matplotlib.pyplot as plt
from matplotlib import colors

log = logging.getLogger(__name__)

PROJECT_DIR = Path(Path(__file__).parents[1])
BAL_AUTH_SHAPE_FILE = Path(PROJECT_DIR, 'control-areas-shp')
GPW_POPULATION_FILE = Path(PROJECT_DIR, 'population-data', 'gpw_v4_population_density_adjusted_rev11_2pt5_min.nc')
OUTPUT_DIR = Path(PROJECT_DIR, 'output')
GALLERY_DIR = Path(PROJECT_DIR, 'gallery')

GPW_POPULATION_VAR = 'UN WPP-Adjusted Population Density, v4.11 (2000, 2005, 2010, 2015, 2020): 2.5 arc-minutes'

GPW_POP_DENSITY_LAYER = {2000 : 0, # see population-data/gpw_v4_netcdf_contents_rev11.csv
                            2005 : 1, 
                            2010 : 2, 
                            2015 : 3, 
                            2020 : 4} 

BAL_AUTH_NAMES = { 'MISO' : 'MIDCONTINENT INDEPENDENT TRANSMISSION SYSTEM OPERATOR, INC..',
    'SWPP' : 'SOUTHWEST POWER POOL', 
    'WACM' : 'WESTERN AREA POWER ADMINISTRATION - ROCKY MOUNTAIN REGION',
    'BPAT' : 'BONNEVILLE POWER ADMINISTRATION',
    'WALC' : 'WESTERN AREA POWER ADMINISTRATION - DESERT SOUTHWEST REGION',
    'PACE' : 'PACIFICORP - EAST',
    'PJM' : 'PJM INTERCONNECTION, LLC',
    'ERCO' : 'ELECTRIC RELIABILITY COUNCIL OF TEXAS, INC.',
    'NWMT' : 'NORTHWESTERN ENERGY (NWMT)',
    'NEVP' : 'NEVADA POWER COMPANY',
    'CISO' : 'CALIFORNIA INDEPENDENT SYSTEM OPERATOR',
    'SOCO' : 'SOUTHERN COMPANY SERVICES, INC. - TRANS',
    'PACW' : 'PACIFICORP - WEST',
    'IPCO' : 'IDAHO POWER COMPANY',
    'PNM' : 'PUBLIC SERVICE COMPANY OF NEW MEXICO',
    'AECI' : 'ASSOCIATED ELECTRIC COOPERATIVE, INC.',
    'TVA' : 'TENNESSEE VALLEY AUTHORITY',
    'ISNE' : 'ISO NEW ENGLAND INC.',
    'EPE' : 'EL PASO ELECTRIC COMPANY',
    'WAUW' : 'WESTERN AREA POWER ADMINISTRATION UGP WEST',
    'NYIS' : 'NEW YORK INDEPENDENT SYSTEM OPERATOR',
    'AVA' : 'AVISTA CORPORATION',
    'AZPS' : 'ARIZONA PUBLIC SERVICE COMPANY',
    'PSCO' : 'PUBLIC SERVICE COMPANY OF COLORADO',
    'CPLE' : 'DUKE ENERGY PROGRESS EAST',
    'DUK' : 'DUKE ENERGY CAROLINAS',
    # 'CHUGACH ELECTRIC ASSN INC'
    'FPC' : 'DUKE ENERGY FLORIDA INC',
    'AEC' : 'POWERSOUTH ENERGY COOPERATIVE',
    'FPL' : 'FLORIDA POWER & LIGHT COMPANY',
    'SEC' : 'SEMINOLE ELECTRIC COOPERATIVE',
    'SC' : 'SOUTH CAROLINA PUBLIC SERVICE AUTHORITY',
    'PSEI' : 'PUGET SOUND ENERGY',
    'SRP' : 'SALT RIVER PROJECT',
    'TPWR' : 'CITY OF TACOMA, DEPARTMENT OF PUBLIC UTILITIES, LIGHT DIVISION',
    'TEPC' : 'TUCSON ELECTRIC POWER COMPANY',
    'IID' : 'IMPERIAL IRRIGATION DISTRICT',
    'DOPD' : 'PUD NO. 1 OF DOUGLAS COUNTY',
    'LGEE' : 'LOUISVILLE GAS AND ELECTRIC COMPANY AND KENTUCKY UTILITIES',
    'FMPP' : 'FLORIDA MUNICIPAL POWER POOL',
    'CPLW' : 'DUKE ENERGY PROGRESS WEST',
    'BANC' : 'BALANCING AUTHORITY OF NORTHERN CALIFORNIA',
    'PGE' : 'PORTLAND GENERAL ELECTRIC COMPANY',
    'GWA' : 'NATURENER POWER WATCH, LLC (GWA)',
    'JEA' : 'JEA',
    'CHPD' : 'PUBLIC UTILITY DISTRICT NO. 1 OF CHELAN COUNTY',
    'GCPD' : 'PUBLIC UTILITY DISTRICT NO. 2 OF GRANT COUNTY, WASHINGTON',
    # 'ANCHORAGE MUNICIPAL LIGHT & POWER'
    'LDWP' : 'LOS ANGELES DEPARTMENT OF WATER AND POWER',
    'HST' : 'CITY OF HOMESTEAD',
    'HGMA' : 'NEW HARQUAHALA GENERATING COMPANY, LLC - HGBA',
    'TEC' : 'TAMPA ELECTRIC COMPANY',
    'GRMA' : 'GILA RIVER POWER, LLC',
    'WWA' : 'NATURENER WIND WATCH, LLC',
    # 'GRIDFORCE SOUTH'
    'SEPA' : 'SOUTHEASTERN POWER ADMINISTRATION',
    'NSB' : 'NEW SMYRNA BEACH, UTILITIES COMMISSION OF',
    'GVL' : 'GAINESVILLE REGIONAL UTILITIES',
    'TIDC' : 'TURLOCK IRRIGATION DISTRICT',
    # 'HAWAIIAN ELECTRIC CO INC'
    'EEI' : 'ELECTRIC ENERGY, INC.',
    'GRID' : 'GRIDFORCE ENERGY MANAGEMENT, LLC',
    'OVEC' : 'OHIO VALLEY ELECTRIC CORPORATION',
    'GRIF' : 'GRIFFITH ENERGY, LLC',
    'YAD' : 'ALCOA POWER GENERATING, INC. - YADKIN DIVISION', 
    'SCL' : 'SEATTLE CITY LIGHT',
    'TAL' : 'CITY OF TALLAHASSEE',
    'DEAA' : 'ARLINGTON VALLEY, LLC - AVBA',
    'NBSO' : 'NEW BRUNSWICK SYSTEM OPERATOR',
    'SCEG' : 'SOUTH CAROLINA ELECTRIC & GAS COMPANY'}

def get_bal_auth_shape_hifld(bal_auth_list: str, shape_data: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    '''
    return subset of geopandas dataframe from HIFLD shape file.
    '''
    return shape_data[shape_data['NAME'].isin([BAL_AUTH_NAMES[bal_auth] for bal_auth in bal_auth_list])]

def get_bounds(shape: gpd.GeoDataFrame, tolerance: float=.75) -> tuple:
    ''' 
    include tolerance to search nearby coordinates outside of bal_auth for mapping temp. to pop.'''

    bounds = shape.total_bounds
    min_lon = round(bounds[0],3) - tolerance
    min_lat = round(bounds[1],3) - tolerance
    max_lon = round(bounds[2],3) + tolerance
    max_lat = round(bounds[3],3) + tolerance

    return min_lon, max_lon, min_lat, max_lat
    
def convert_to_lat_lon(shape: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    '''
    return shape with coordinates projected to lat/lon
    '''
    return shape.to_crs('EPSG:4326')

def get_temp_era5(temp_dataset: xr.Dataset, min_lon: int, max_lon: int, min_lat: int, max_lat: int) -> tuple:
    '''
    return temperature
    '''
    # crop dataset selection
    cropped_temp_dataset = temp_dataset.sel(longitude=slice(min_lon, max_lon), latitude=slice(max_lat, min_lat))
    
    # get temperature
    return cropped_temp_dataset['t2m']

def get_population_density_gpw(pop_dataset: xr.Dataset, pop_year: int, min_lon: int, max_lon: int, min_lat: int, max_lat: int) -> tuple:
    # crop dataset selection
    cropped_pop_dataset = pop_dataset.sel(raster=GPW_POP_DENSITY_LAYER[pop_year], longitude=slice(min_lon, max_lon), latitude=slice(max_lat, min_lat))

    # get population
    return cropped_pop_dataset[GPW_POPULATION_VAR]


def get_nearest_idx(coord: float, coord_arr: xr.DataArray) -> int:
    '''
    return index of nearest coordinate.
    '''
    return abs(coord - coord_arr).argmin()

def get_resolution(coord: xr.DataArray):
    '''
    return resolution of a uniformly resolved coordinate array.
    '''
    return float(abs(coord[1] - coord[0]))

def upscale_and_align_pop(
    high_res_pop_arr: xr.DataArray,
    low_res_temp_arr: xr.DataArray,
    method: str) -> xr.DataArray:
    '''
    Upscale and align a high-resolution array to fit a low-resolution array. 
    Upscaling method is either 'average' (e.g. temperature) or 'accumulate' (e.g. population)
    '''
    # check input
    assert get_resolution(high_res_pop_arr.coords['latitude']) < get_resolution(low_res_temp_arr.coords['latitude'])
    assert get_resolution(high_res_pop_arr.coords['longitude']) < get_resolution(low_res_temp_arr.coords['longitude'])
    assert method in ['average', 'accumulate']

    # make accumulation arrays
    accum = xr.DataArray(
        data=np.zeros((low_res_temp_arr.sizes['latitude'], low_res_temp_arr.sizes['longitude'])),
        coords=dict(
            latitude=low_res_temp_arr.latitude,
            longitude=low_res_temp_arr.longitude
        ),
        name=high_res_pop_arr.name,
        attrs=high_res_pop_arr.attrs
    )
    count = accum.copy()

    for lat_idx, lat in enumerate(high_res_pop_arr.coords['latitude']):
        for lon_idx, lon in enumerate(high_res_pop_arr.coords['longitude']):

            lat_low_res_idx = get_nearest_idx(lat, low_res_temp_arr.coords['latitude'])
            lon_low_res_idx = get_nearest_idx(lon, low_res_temp_arr.coords['longitude'])

            accum[lat_low_res_idx, lon_low_res_idx] += high_res_pop_arr[lat_idx, lon_idx]
            count[lat_low_res_idx, lon_low_res_idx] += 1

        log.info(f'\t\tMapped {lat_idx} of {len(high_res_pop_arr.coords["latitude"])}...')

    return accum if method == 'accumulate' else accum / count

def rasterize_shape(shape: gpd.GeoDataFrame, arr: xr.DataArray) -> xr.DataArray:
    '''
    Rasterize a geopandas shape to fit into an DataArray. 
    '''
    mapped = xr.zeros_like(arr)

    for lat_idx, lat in enumerate(arr.coords['latitude']):
        for lon_idx, lon in enumerate(arr.coords['longitude']):
            if shape.contains(Point(lon,lat)).any():
                mapped[lat_idx,lon_idx] = 1

        log.info(f'\tMapped {lat_idx} of {len(arr.coords["latitude"])}...')
            
    return mapped

def plot_temp(temp, fname):

    fig, ax = plt.subplots()

    temp_c = temp.mean(dim='time') - 273

    pos = ax.imshow(temp_c, origin='lower')
    ax.grid(False)
    cbar = fig.colorbar(pos, ax=ax, extend='both')
    cbar.set_label('Mean Temperature ($\\degree$C)')

    steps = max(temp.sizes['latitude'], temp.sizes['longitude'],5)//5
    ax.set_xticks(np.arange(temp.sizes['longitude'])[::steps])
    ax.set_xticklabels(temp.longitude[::steps].values.astype(int))
    ax.set_yticks(np.arange(temp.sizes['latitude'])[::steps])
    ax.set_yticklabels(temp.latitude[::steps].values.astype(int))

    ax.set_xlabel('Longitude ($\degree$)')
    ax.set_ylabel('Latitude ($\degree$)')

    plt.savefig(fname)
    plt.close()

    return None

def plot_pop(pop, fname):

    fig, ax = plt.subplots()

    pos = ax.imshow(pop.where(pop, np.nan), norm=colors.LogNorm(), origin='lower')
    ax.grid(False)
    cbar = fig.colorbar(pos, ax=ax, extend='both')
    cbar.set_label('Population Density (Persons per km$^2$)')

    steps = max(pop.sizes['latitude'], pop.sizes['longitude'],5)//5
    ax.set_xticks(np.arange(pop.sizes['longitude'])[::steps])
    ax.set_xticklabels(pop.longitude[::steps].values.astype(int))
    ax.set_yticks(np.arange(pop.sizes['latitude'])[::steps])
    ax.set_yticklabels(pop.latitude[::steps].values.astype(int))

    ax.set_xlabel('Longitude ($\degree$)')
    ax.set_ylabel('Latitude ($\degree$)')

    plt.savefig(fname)
    plt.close()

    return None

def plot_weighted_unweighted(weighted, unweighted, fname):

    fig, ax = plt.subplots()

    ax.scatter(weighted, unweighted)
    
    ax.set_xlabel('Hourly Population-weighted Temperature (K)')
    ax.set_ylabel('Hourly Unweighted Mean Temperature (K)')

    plt.savefig(fname)
    plt.close()

    return None


def set_centroid(arr, val):
    '''
    set value in the central-most index of a two-dimensional array
    '''
    centroid_x = arr.shape[0] // 2
    centroid_y = arr.shape[1] // 2
    arr[centroid_x, centroid_y] = val
    return None

def get_centroid(arr):
    '''
    return value in the central-most index of a two-dimensional array
    '''
    centroid_x = arr.shape[0] // 2
    centroid_y = arr.shape[1] // 2
    return arr[centroid_x, centroid_y]

def main(
    year: int, 
    area_name: str,
    bal_auth_list: list,
    reanalysis_temperature_file: Path,
    pop_year: int=2020):
    '''
    Generate population-weighted temperature series for a given balancing authority.
    '''

    # load balancing authority shape data
    bal_auth_shape_data = gpd.read_file(BAL_AUTH_SHAPE_FILE)
    bal_auth_shape_espg_3857 = get_bal_auth_shape_hifld(bal_auth_list, bal_auth_shape_data)
    bal_auth_shape = convert_to_lat_lon(bal_auth_shape_espg_3857)
    
    log.info(f'Finished loading balancing authority shape...')
    
    # find bounds of balancing authority
    min_lon, max_lon, min_lat, max_lat = get_bounds(bal_auth_shape)

    log.info(f'Using coordinate boundaries {min_lon, min_lat} -> {max_lon, max_lat}')
    
    # load temperature data
    temp_dataset = xr.open_dataset(reanalysis_temperature_file)
    temp = get_temp_era5(temp_dataset, min_lon, max_lon, min_lat, max_lat)
    temp_dataset.close()

    log.info('Finished loading temperature data...')
    
    # load poopulation data (mapping to reanalysis resolution)
    pop_dataset = xr.open_dataset(GPW_POPULATION_FILE)
    pop = get_population_density_gpw(pop_dataset, pop_year, min_lon, max_lon, min_lat, max_lat)
    pop_dataset.close()
    
    log.info('Finished loading population data...')

    # align coordinates
    log.info('Upscaling population to fit temperature data.')
    pop = upscale_and_align_pop(high_res_pop_arr=pop, low_res_temp_arr=temp, method='accumulate')
    
    # map balancing authority bounds to population array
    log.info('Rasterizing balancing authority map array...')
    bal_auth_map = rasterize_shape(shape=bal_auth_shape, arr=pop)

    # if no coordinates are in balancing authority, use the centroid coordinate
    if bal_auth_map.sum() == 0: 
        log.info('No discrete coordinates in balancing authority. Using centroid.')
        set_centroid(bal_auth_map, 1)

    # filter population by balancing authority
    bal_auth_pop = pop.where(bal_auth_map, pop, 0)

    # if no available population data is available, use the centroid coordinate
    if bal_auth_pop.sum() == 0:
        set_centroid(bal_auth_pop, 1)

    # get population weighted temperature
    pop_weighted_temp = (temp * bal_auth_pop).sum(dim=('latitude', 'longitude'))/bal_auth_pop.sum()
    pop_weighted_temp_ds = pop_weighted_temp.rename('Temperature (K)').to_pandas()

    # get average temperature
    unweighted_mean_temp = (temp * bal_auth_map).sum(dim=('latitude', 'longitude'))/bal_auth_map.sum()
    unweighted_mean_temp_ds = unweighted_mean_temp.rename('Temperature (K)').to_pandas()

    # save
    pop_weighted_temp_file = Path(OUTPUT_DIR, f'{area_name}-{year}-pop-weighted-temperature.csv')
    pop_weighted_temp_ds.round(2).to_csv(pop_weighted_temp_file)

    unweighted_mean_temp_file = Path(OUTPUT_DIR, f'{area_name}-{year}-unweighted-mean-temperature.csv')
    unweighted_mean_temp_ds.round(2).to_csv(unweighted_mean_temp_file)

    # plot
    plt.style.use('ggplot')

    pop_plot_file = Path(GALLERY_DIR, f'{area_name}-{year}-pop-map.png')
    temp_plot_file = Path(GALLERY_DIR, f'{area_name}-{year}-temp-map.png')
    weighted_unweighted_plot_file = Path(GALLERY_DIR, f'{area_name}-{year}-weighted-unweighted-scatter.png')

    plot_pop(bal_auth_pop, pop_plot_file)
    plot_temp(temp, temp_plot_file)
    plot_weighted_unweighted(pop_weighted_temp, unweighted_mean_temp, weighted_unweighted_plot_file)

    return None

if __name__ == '__main__':
    # set-up logger
    logging.basicConfig(level=logging.INFO)
    
    # arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('year', type=int)
    parser.add_argument('reanalysis_temperature_file', type=str)
    parser.add_argument('area_name', type=str)
    parser.add_argument('-b', '--bal-auth-list', type=str, required=True, nargs='+')
    parser.add_argument('-p', '--pop-year', type=int, default=2020, choices=[year for year in GPW_POP_DENSITY_LAYER], help='Population data year to use for weighting.')

    args = parser.parse_args()

    main(args.year, args.area_name, args.bal_auth_list, args.reanalysis_temperature_file, args.pop_year)
