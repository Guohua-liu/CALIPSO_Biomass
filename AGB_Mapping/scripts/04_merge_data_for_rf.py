#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
This script prepares and merges datasets for input into random forest models for both training and prediction.
    - Training: Merges target and predictor datasets.
    - Prediction: Prepares predictor datasets.

Method: 
    - Step1: Load all datasets for target and predictors.
    - Step2: Convert the xarray data to flat array data suitable for random forest model inputs.
    - Step3: Merge all flat data and drop NaN.

Outputs:
    - Merged data for above-ground biomass random forest model training and prediction.
"""

import pandas as pd
import xarray as xr

# Set the general parameters
radar_index = 'max'
filter_type = 'filter1-water'  # 'filter2-biome'

# Function to load and prepare datasets
def load_and_prepare_data(include_target=False):
    

    # ---------- Predictor data: radar signal 
    in_path = "../data/data_agb_mapping/global_radar_data_1993-2020_filtered_fin.nc"
    ds_rad = xr.open_dataset(in_path)
    rad_data = ds_rad.sel(year=slice('2017', '2020')).RadarSignal

    # ---------- static Predictor data: climate[mean of 1991-2020]
    in_path_clim = "../data/data_agb_mapping/terraclimate/terraclimate_MAT_MAP_1991_2020_8.9km.nc"
    ds_clim = xr.open_dataset(in_path_clim)
    ds_clim = ds_clim .interp(lat=ds_rad.lat,lon=ds_rad.lon,method='nearest')

    # ---------- dynamic Predictor data: climate[1991-2020]
    in_path_clim_yr = '../data/data_agb_mapping/terraclimate_MAT_MAP_1991_2020_8.9km_annual_mean.nc'
    ds_clim_yr = xr.open_dataset(in_path_clim_yr)
    ds_clim_yr = ds_clim_yr.sel(year=slice(2017,2020))
    ds_clim_yr = ds_clim_yr.transpose('lat', 'lon','year')
    ds_clim_yr = ds_clim_yr.interp(lat=ds_rad.lat,lon=ds_rad.lon,method='nearest')
    
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
        in_path_cci = "../data/data_agb_mapping/global_agb_data_2017-2020_filtered_fin.nc"
        ds_cci = xr.open_dataset(in_path_cci)
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

# Prepare data for training
training_data = load_and_prepare_data(include_target=True)
training_output_path = '../data/data_agb_mapping/Input_mergeddata_for_rf_model_train_fin.csv'
training_data.to_csv(training_output_path, index=False)

# Prepare data for prediction
prediction_data = load_and_prepare_data(include_target=False)
prediction_output_path = '../data/data_agb_mapping/Input_mergeddata_for_rf_model_pred_fin.csv'
prediction_data.to_csv(prediction_output_path, index=False)

print('Done!')