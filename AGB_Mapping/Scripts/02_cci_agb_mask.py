#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import numpy as np
import xarray as xr
from os.path import join

# Set input and output paths
input_path ='../InputData/'
output_path = '../Outputs/'

# load cci data 
ds_cci = xr.open_dataset(join(input_path, 'ESA_CCI_Biomass.nc')) \
         .sel(year=slice(2017,2020)) 

# cal cv and mask 
ds_masked = ds_cci.copy()

# Compute CV safely
with np.errstate(divide='ignore', invalid='ignore'):
    cv = ds_masked['agb_se']/ ds_masked['agb']
    
# Replace non-finite results (infinities from division by zero) with NaN 
cv = cv.where(np.isfinite(cv), np.nan)
ds_masked['cv'] = cv
cv_thr = 0.8
mask = (cv <= cv_thr) | (ds_masked['agb'] == 0)

dims_except_year = [dim for dim in ds_masked['agb'].dims if dim != 'year']
total_count_by_year = ds_masked['agb'].groupby('year').count(dim=dims_except_year)
masked_count_by_year = (~mask).groupby('year').sum(dim=dims_except_year)
percent_masked_by_year = (masked_count_by_year / total_count_by_year * 100)
print("\nPercentage of grids masked for each year (relative to original grids):")

for year, percent in zip(percent_masked_by_year['year'].values,
                         percent_masked_by_year.values):
    print(f"Year {year}: {percent:.2f}% masked")

# Apply the mask to the 'agb' data 
ds_masked['agb'] = ds_masked['agb'].where(mask)

out_file = os.path.join(output_path, "ESA_CCI_Biomass_masked_cv.nc")
ds_masked.to_netcdf(out_file)
