"""
Compute Radar backscatter mean during wet season based on ESA CCI soil moisture data.
Steps:
    1. Load the wet season mask from ESA CCI soil moisture data.
    2. Load the monthly radar signal data.
    3. Apply the wet season mask to the radar signal data.
    4. Compute the mean radar signal during wet season.
    5. Save the wet season yearly mean radar signal as a NetCDF file.
"""

import os
import pandas as pd
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import mplotutils as mpu
from os.path import join

# Set input and output paths
radar_path = '../data/radsigdata_terraclim/'
sm_path= '../data/esacci_sm/'
fig_path = '../fig/'
result_path = '../result/'

q_threshold = 2/3  # 66th percentile

# Load the wet season mask from ESA CCI soil moisture data
wet_season_ds = xr.open_dataset(join(result_path, 'interm/comp08_esacci_sm_wet_season.8.9km.nc')) \
         .sel(time=slice('1993-01-01', '2020-12-31')) \
         .drop_vars('quantile')
wet_season_mask = wet_season_ds['wet_season_mask']

# Load the monthly radar signal data
radar_monthly_smooth = xr.open_dataset(join(radar_path, 'Global_RadarSignal_monthly_1992_2020_8_9km_smooth_3m.nc')) \
         .sel(time=slice('1993-01-01', '2020-12-31')) \
         .transpose('time','lat','lon')

radar_monthly = xr.open_dataset(join(radar_path, 'Global_RadarSignal_monthly_1992-2020_8.9km.nc')) \
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

radar_wet_season_yrmean = calc_local_wet_yrmean(radar_monthly_smooth, wet_season_mask)
radar_original_wet_season_yrmean = calc_local_wet_yrmean(radar_monthly, wet_season_mask)

# Save the wet season mean radar scatter data 
radar_wet_season_yrmean.to_netcdf(join(radar_path, 'comp09_Global_RadarSignal_1993_2020_8_9km_yrmean_q66wetmonths_smooth_3m.nc'))
radar_original_wet_season_yrmean.to_netcdf(join(radar_path, 'comp09_Global_RadarSignal_1993_2020_8_9km_yrmean_q66wetmonths.nc'))

print("wet season mask applied to radar signal data.")