#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import numpy as np
import xarray as xr
from os.path import join

# Set input and output paths
input_path ='../InputData/'
output_path = '../Outputs/'

# load the data 
# radar signal data (wet season)
ds_radar = xr.open_dataset(join(output_path, 'Radar_wet_season_1993-2020.nc')) 

# ESA-CCI AGB data (masked poor quality)
ds_agb = xr.open_dataset(join(output_path, 'ESA_CCI_Biomass_masked_cv.nc')) 

# ESA-CCI PFT maps
ds_pft = xr.open_dataset(join(input_path, 'ESA_CCI_PFT_8-9km.nc')) 
# Trees total
ds_trees = ds_pft['TREES-BD'] + ds_pft['TREES-BE'] + ds_pft['TREES-ND'] + ds_pft['TREES-NE']
# Barren area
ds_barren = ds_pft['BARE']
# Urban areas 
ds_urban = ds_pft['BUILT']

# load the wetland map 
ds_wet = xr.open_dataset(join(input_path, 'Wetland_globe_8-9km.nc')) 

# load the peatland map 
ds_peat = xr.open_dataset(join(input_path, 'Peatland_globe_8-9km.nc')) 

# ------------ Mask the data 
mask1 = (ds_trees >= 10) & (ds_urban <= 50) & (ds_barren <= 50) & (ds_wet['wetland']== 0)& (ds_peat['PEATLAND_P']<= 10)
radar_masked1 = ds_radar.where(mask1, drop=True)
agb_masked1 = ds_agb.where(mask1, drop=True)

# mask the abnormal data
radar_signal = radar_masked1['RadarSignal']
radar_mean = radar_signal.mean(dim='year', skipna=True)
radar_std = radar_signal.std(dim='year', skipna=True)

anom_thred = 4
lower_bound = radar_mean - anom_thred * radar_std
upper_bound = radar_mean + anom_thred * radar_std

# Detect anomalies, considering NaN values
anomalies = xr.where(
    ((radar_signal < lower_bound) | (radar_signal > upper_bound)),
    1,  # Mark as anomaly
    0   # Mark as normal
)
anomaly_mask = anomalies.astype(bool)

# Apply the mask to radar data - set anomalies to NaN
radar_masked_fin = radar_masked1.where(~anomaly_mask)
# ------ Apply the same mask to AGB data
anomaly_mask_agb = anomaly_mask.sel(year=slice(2017,2020))
agb_masked_fin = agb_masked1.where(~anomaly_mask_agb)

radar_masked_fin.to_netcdf(join(output_path, 'Radar_wet_season_1993-2020_filtered.nc'))
agb_masked_fin.to_netcdf(join(output_path, 'ESA_CCI_Biomass_filtered.nc'))

print('done!')