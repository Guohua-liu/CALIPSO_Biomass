#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pandas as pd
import xarray as xr
from os.path import join

# Set input and output paths
input_path ='../InputData/'
output_path = '../Outputs/'

# Function to load and prepare datasets
def load_and_prepare_data(include_target=False):
    # ---------- Predictor data: radar signal 
    ds_rad = xr.open_dataset(join(output_path, 'Radar_wet_season_1993-2020_filtered.nc'))
    rad_data = ds_rad.sel(year=slice('2017', '2020')).RadarSignal

    # ---------- static Predictor data: climate[mean of 1991-2020]
    ds_clim = xr.open_dataset(join(input_path, 'TerraClimate_MAT_MAP_8-9km.nc'))

    # ---------- dynamic Predictor data: climate[1991-2020]
    ds_clim_yr = xr.open_dataset(join(input_path, 'TerraClimate_MAT_MAP_8-9km_yr.nc'))
    ds_clim_yr = ds_clim_yr.sel(year=slice(2017,2020))
    ds_clim_yr = ds_clim_yr.transpose('lat', 'lon','year')
    
    rad_flat = rad_data.to_dataframe(name='rad').reset_index()
    cci_flat = cci_data.to_dataframe(name='cci').reset_index()
    clim_flat = ds_clim.to_dataframe().reset_index()
    clim_yr_flat = ds_clim_yr.to_dataframe().reset_index().rename(
        columns={'MAT': 'MAT_yr', 'MAP': 'MAP_yr'})

    # Merge data
    merged_data = pd.merge(rad_flat, clim_yr_flat, on=['lat', 'lon', 'year'])
    merged_data = pd.merge(merged_data, clim_flat, on=['lat', 'lon'])

    if include_target:
        # Load target data: CCI biomass
        # ---------- Target data: CCI biomass (global, 2017-2020)
        ds_cci = xr.open_dataset(join(output_path, 'ESA_CCI_Biomass_filtered.nc'))
        cci_data = ds_cci.sel(year=slice('2017', '2020')).agb
        cci_data = cci_data.reindex(lat=cci_data.lat[::-1])
        cci_flat = cci_data.to_dataframe(name='cci').reset_index()
        merged_data = pd.merge(merged_data, cci_flat, on=['lat', 'lon', 'year'])

    merged_data = merged_data.dropna()
    sorted_data = merged_data.sort_values(by=['year', 'lat', 'lon'])

    columns = ['year', 'lat', 'lon', 'rad', 'MAT_yr', 'MAP_yr', 'MAT', 'MAP']
    if include_target:
        columns.insert(4, 'cci')
    sorted_data = sorted_data[columns]

    return sorted_data

# save data for training
training_data = load_and_prepare_data(include_target=True)
training_data.to_csv(join(output_path, 'Merged_data_for_train.csv'))

# save data for prediction
prediction_data = load_and_prepare_data(include_target=False)
prediction_data.to_csv(join(output_path, 'Merged_data_for_pred.csv'))