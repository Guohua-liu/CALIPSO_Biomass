#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Script to read the orignal monthly tiff data (1992-2020) and store them to one netcdf file, the spatial reolution is the same with terraclimate  
    - step1: load the geotif files and convert them to the nc files 
    - step2: change these radar nc files to the same spatial resolution with terraclim
    - step3: store the final nc data
"""

import os
import numpy as np
import xarray as xr
import glob
from osgeo import gdal
from os.path import join

# ------- step1: convert tif to nc 
# Load the TIFF files
in_path_tif = '/Net/Groups/BGI/people/gliu/calipso/data/GobalRadar'  # Monthly data from Jan 1992 to Aug 2021
tif_files = sorted(glob.glob(os.path.join(in_path_tif, '*.tif')))

# Filter the files from 1992 Jan to 2020 Dec
tif_files = [f for f in tif_files if '1992' <= os.path.basename(f).split('.')[0][:4] < '2021']

# Initialize lists to store data and time
data_arrays = []
time_stamps = []

# Read the data and extract timestamps
for tif_file in tif_files:
    ds = gdal.Open(tif_file)
    data = ds.ReadAsArray()
    ocean_mask = (data == 0)
    land_data = np.ma.masked_array(data, mask=ocean_mask)
    data_arrays.append(land_data.filled(np.nan))
    
    # Extract timestamp from file name (assuming the format is consistent)
    year_month = os.path.basename(tif_file).split('.')[0]
    year = int(year_month[:4])
    month = int(year_month[4:6])
    time_stamps.append(pd.Timestamp(year=year, month=month, day=1))

# Stack data arrays along the time dimension
data_stack = np.stack(data_arrays, axis=0)

# Get geotransform information from the first file
geotransform = ds.GetGeoTransform()
originX = geotransform[0]
originY = geotransform[3]
pixelWidth = geotransform[1]
pixelHeight = geotransform[5]

# Calculate latitude and longitude arrays
latitudes = originY + np.arange(data_stack.shape[1]) * pixelHeight
longitudes = originX + np.arange(data_stack.shape[2]) * pixelWidth

# Create a new xarray Dataset
ds_rad = xr.Dataset(
    data_vars = {
        'RadarSignal': (['time', 'lat', 'lon'],data_stack )
    },
    coords={
        'time': time_stamps,
        'lat': latitudes,
        'lon': longitudes
    }
)

# change the latitude from the low to the high 
ds_rad = ds_rad.reindex(lat=ds_rad.lat[::-1])

# ------- step2: change the resolution as same as terraclim
# the desired grids from terraclimate 
in_path_clim = '/Net/Groups/BGI/data/DataStructureMDI/DATA/grid/Global/0d083_monthly/TerraClimate/v2018/Data/tmax/tmax.4320.2160.2020.nc'
ds_clim = xr.open_dataset(in_path_clim)
ds_clim =ds_clim.rename({'latitude': 'lat', 'longitude': 'lon'})
ds_rad_0d083 = ds_rad.interp(lon=ds_clim.lon.values, lat=ds_clim.lat.values, method='nearest')

# ------- step3: save the netcdf file 
out_path = '../data/radsigdata_terraclim/global_nc_from_tif'
ds_rad_0d083.to_netcdf(
            join(out_path, 'Global_RadarSignal_monthly_1992-2020_0d083.nc'),
            format='netCDF4',
            engine='netcdf4',
            encoding={'RadarSignal': {'zlib': True, 'complevel': 9}}
        )

print('Done!')
