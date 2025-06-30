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
    # Load predictor data: radar signal
    in_path = f'../data/data_processed/Radar_{radar_index}_0d083_global_1992-2020_{filter_type}.nc'
    ds_rad = xr.open_dataset(in_path)
    rad_data = ds_rad.RadarSignal

    # Load predictor data: tree coverage
    in_path_tc = '/Net/Groups/BGI/data/DataStructureMDI/DATA/grid/Global/1d00_annual/Hanson_treecover/GFC-2023-v1.11/Data/treecover.360.180.2000.nc'
    ds_tc = xr.open_dataset(in_path_tc)
    ds_tc = ds_tc.rename({'latiude': 'lat', 'longitude': 'lon'})
    ds_tc = ds_tc.interp(lon=ds_rad.lon.values, lat=ds_rad.lat.values, method='nearest')
    ds_tc = ds_tc.squeeze('time', drop=True)

    # Load predictor data: tree density
    in_path_td = '/Net/Groups/BGI/data/DataStructureMDI/DATA/grid/Global/0d083_static/ConopyHeight/TreeDensity/Crowther_2015/Data/TreeDensity.biome_model.4320.2160.nc'
    ds_td = xr.open_dataset(in_path_td)
    ds_td = ds_td.rename({'y': 'lat', 'x': 'lon'})
    ds_td = ds_td.interp(lon=ds_rad.lon.values, lat=ds_rad.lat.values, method='nearest')
    ds_td = ds_td.squeeze('band', drop=True)

    # Load predictor data: climate
    in_path_clim = '../data/preprocessed/MAT_MAP_1991-2020_globe_0d083.nc'
    ds_clim = xr.open_dataset(in_path_clim)

    # Convert to flat dataframes
    rad_flat = rad_data.to_dataframe(name='rad').reset_index()
    tc_flat = ds_tc.to_dataframe().reset_index().rename(columns={'treecover': 'tc'})
    td_flat = ds_td.to_dataframe().reset_index().rename(columns={'TreeDensity': 'td'})
    clim_flat = ds_clim.to_dataframe().reset_index()

    # Merge data
    merged_data = pd.merge(rad_flat, tc_flat, on=['lat', 'lon'])
    merged_data = pd.merge(merged_data, td_flat, on=['lat', 'lon'])
    merged_data = pd.merge(merged_data, clim_flat, on=['lat', 'lon'])

    if include_target:
        # Load target data: CCI biomass
        in_path_cci = f'../data/data_processed/ESA-CCI_0d083_global_2010-2017-2020_{filter_type}.nc'
        ds_cci = xr.open_dataset(in_path_cci)
        cci_data = ds_cci.sel(year=slice('2017', '2020')).AGB
        cci_flat = cci_data.to_dataframe(name='cci').reset_index()
        merged_data = pd.merge(merged_data, cci_flat, on=['lat', 'lon', 'year'])

    merged_data = merged_data.dropna()
    sorted_data = merged_data.sort_values(by=['year', 'lat', 'lon'])
    columns = ['year', 'lat', 'lon', 'rad', 'tc', 'td', 'MAT', 'MAP']
    if include_target:
        columns.insert(4, 'cci')
    sorted_data = sorted_data[columns]

    return sorted_data

# Prepare data for training
training_data = load_and_prepare_data(include_target=True)
training_output_path = f'../data/data_processed/Input_mergeddata_for_rf_model_{radar_index}_{filter_type}.csv'
training_data.to_csv(training_output_path, index=False)

# Prepare data for prediction
prediction_data = load_and_prepare_data(include_target=False)
prediction_output_path = f'../data/data_processed/Input_mergeddata_for_rf_pred_{radar_index}_{filter_type}_1992-2020.csv'
prediction_data.to_csv(prediction_output_path, index=False)

print('Done!')