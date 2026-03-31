#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#!/usr/bin/env python
"""
Script to filter the poor quality CCI AGB data by CV (=agb_se/agb) for a given CV threshold. 

"""

import os
import numpy as np
import xarray as xr
import argparse
import time
import logging

# Configure logging
logging.basicConfig(filename='cci_8_9km_filter_by_cv.log', level=logging.INFO)
logger = logging.getLogger(__name__)

parser = argparse.ArgumentParser(
    description='Mask AGB data based on CV threshold (agb_se / agb).')
parser.add_argument('--cv', type=float, default=0.3,
                    help='One or more CV threshold values (e.g., --cv 0.3 0.5)')
args = parser.parse_args()
cv_thr = args.cv

# load cci data 
in_path_cci = '../data/data_agb_mapping/agb_esacci/ESACCI-AGB-Map-V6_0d08186_2007-2022.nc'
ds_cci = xr.open_dataset(in_path_cci)
ds_cci = ds_cci.sel(year=slice(2015,2020))

# cal cv and mask 
ds_masked = ds_cci.copy()

# Compute CV safely even if agb == 0
with np.errstate(divide='ignore', invalid='ignore'):
    cv = ds_masked['agb_se']/ ds_masked['agb']
    
# Replace non-finite results (infinities from division by zero) with NaN 
cv = cv.where(np.isfinite(cv), np.nan)
ds_masked['cv'] = cv

mask = (cv <= cv_thr) | (ds_masked['agb'] == 0)

# ---------- Count total masked grids (over all years) 
masked_count_total = int((~mask).sum().item())
logger.info(f"Total number of cells masked for cv = {cv_thr}: {masked_count_total}")
print(f"Total number of cells masked for cv = {cv_thr}: {masked_count_total}")

dims_except_year = [dim for dim in ds_masked['agb'].dims if dim != 'year']
total_count_by_year = ds_masked['agb'].groupby('year').count(dim=dims_except_year)
masked_count_by_year = (~mask).groupby('year').sum(dim=dims_except_year)
percent_masked_by_year = (masked_count_by_year / total_count_by_year * 100)
print("\nPercentage of grids masked for each year (relative to original grids):")

for year, percent in zip(percent_masked_by_year['year'].values,
                         percent_masked_by_year.values):
    logger.info(f"Year {year}: {percent:.2f}% masked")
    print(f"Year {year}: {percent:.2f}% masked")

# Compute overall average percentage of masked grids using the total counts.
total_grids = ds_masked['agb'].size
average_percent_masked = masked_count_total / total_grids * 100
logger.info(f"Average percentage of grids masked: {average_percent_masked:.2f}%")
print(f"\nAverage percentage of grids masked (over all years): {average_percent_masked:.2f}%")

# ------
# Apply the mask to the 'agb' data 
ds_masked['agb'] = ds_masked['agb'].where(mask)

out_path = '../data/data_agb_mapping/agb_esacci/'
out_file = os.path.join(out_path, f"ESACCI-AGB-Map-V6_0d08186_2007-2022_masked_cv_{int(cv_thr * 100)}.nc")
ds_masked.to_netcdf(out_file)
