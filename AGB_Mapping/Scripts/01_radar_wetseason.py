#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import xarray as xr
from os.path import join

# Set input and output paths
input_path ='../InputData/'
output_path = '../Outputs/'

q_threshold = 2/3  # 66th percentile

# Load the wet season mask from ESA CCI soil moisture data
wet_season_ds = xr.open_dataset(join(input_path, 'ESA_CCI_Soil_Moisture_8-9km.nc')) \
         .sel(time=slice('1993-01-01', '2020-12-31')) \
         .drop_vars('quantile')
wet_season_mask = wet_season_ds['wet_season_mask']

# Load the monthly radar signal data
radar_monthly = xr.open_dataset(join(input_path, 'Radar_monthly_8-9km.nc')) \
         .sel(time=slice('1993-01-01', '2020-12-31')) \
         .transpose('time','lat','lon')

def calc_local_wet_yrmean(radar_data, wet_season_mask):
    """Calculate the yearly mean radar signal during wet season for a single location.

    Parameters:
        radar_data : xarray.DataArray of radar signal for a single location (time dimension).
        wet_mask   : xarray.DataArray of wet season mask for the same location (time dimension).
    Returns:
        xarray.DataArray of yearly mean radar signal during wet season.
    """
    radar = radar_data['RadarSignal']

    # Make sure the underlying shapes are the same
    print(radar.shape, wet_season_mask.shape)

    wet_mask_on_radar_grid = xr.DataArray(
        wet_season_mask.values,
        coords=radar.coords,
        dims=radar.dims,
        name='wet_season_mask'
    )

    # Apply the wet season mask to the radar signal data
    radar_wet_season = radar.where(wet_mask_on_radar_grid)
    radar_wet_season_yrmean = radar_wet_season.groupby('time.year').mean('time', skipna=True)
    return radar_wet_season_yrmean

radar_wet_season_yrmean = calc_local_wet_yrmean(radar_monthly, wet_season_mask)

# Save the wet season mean radar scatter data 
radar_wet_season_yrmean.to_netcdf(join(output_path, 'Radar_wet_season_1993-2020.nc'))