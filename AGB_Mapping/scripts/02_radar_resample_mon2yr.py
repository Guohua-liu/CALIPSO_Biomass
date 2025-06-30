#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Convert monthly radar data to yearly data: including yearly mean and yearly max  

    Step 1. load radar data 

    Step 2. 12-month moving average to remove the noise (Get the long-term changes to remove the water content affect
           through 12-Month moving average), and then campute the yearly mean and yearly max  

    Step 3. store the yearly radar data 
"""

import os
import numpy as np
import pandas as pd
import xarray as xr
from os.path import join

# load the global monthly radar data 
in_path = '../data/radsigdata_terraclim/global_nc_from_tif/Global_RadarSignal_monthly_1992-2020_0d083.nc'
ds_rad = xr.open_dataset(in_path)
ds_rad = ds_rad .transpose('lat', 'lon', 'time')

def moving_average(xr_data, window_size=12):
    """
    Calculate the moving average of a target variable in an xarray dataset.

    Parameters:
    - xr_data: xarray dataset containing the target variable.
    - window_size: size of the rolling window for the moving average (default is 12).

    Returns:
    - xarray dataset with the moving average of the target variable.
    """
    return xr_data.rolling(time=window_size).construct('window').mean('window')

ds_rad = ds_rad.astype('float32')

# 12-month moving average for monthly NetCDF dataset
rad_smooth = moving_average(ds_rad, 12)

# yearly mean radar signal !!!
rad_yr_mean = rad_smooth.groupby('time.year').mean('time', skipna=True)

# yearly max radar signal !!!
rad_yr_max = rad_smooth.groupby('time.year').max('time', skipna=True)

# save the data 
out_path = '../data/radsigdata_terraclim/global_nc_from_tif/'
fname1 = 'Global_RadarSignal_yearly_1992_2022_mean_0d083.nc'
fname2 = 'Global_RadarSignal_yearly_1992_2022_max_0d083.nc'

rad_yr_mean.to_netcdf(os.path.join(out_path, fname1))
rad_yr_max.to_netcdf(os.path.join(out_path, fname2))

print('Done!')