
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

Predicting Biomass from radar and other variables using random forest models
    Step 1. load inputs variables: radar signal data, background climate data, tree cover data, tree density data

    Step 2. load the random forest models 

    Step 3. predict the biomass from 1992-2020 

"""

import os
import numpy as np
import pandas as pd
import joblib
import argparse

parser = argparse.ArgumentParser(description="Parse model parameters")

# Add arguments
parser.add_argument('--model_name', type=str, default='rf-rad-tc', help='Name of the model')
parser.add_argument('--radar_index', type=str, default='mean', help='Radar index: mean or max')
parser.add_argument('--filter_type', type=str, default='filter1-water', help='Type of filter to apply: filter1-water, filter2-biomass')

# Parse the arguments
args = parser.parse_args()

# Assign parsed arguments to variables
model_name = args.model_name
radar_index = args.radar_index
filter_type = args.filter_type

predictors_dict = {
        'rf-rad': ['rad'],
        'rf-rad-tc': ['rad', 'tc'],
        'rf-rad-td': ['rad', 'td'],
        'rf-rad-clim': ['rad', 'MAT', 'MAP'],
        'rf-rad-tc-td': ['rad', 'tc', 'td'],
        'rf-rad-tc-clim': ['rad', 'tc', 'MAT', 'MAP'],
        'rf-rad-td-clim': ['rad', 'td', 'MAT', 'MAP'],
        'rf-rad-tc-td-clim': ['rad', 'tc', 'td', 'MAT', 'MAP']
    }

# load the flated data 
merged_path = f'../data/data_processed/Input_mergeddata_for_rf_pred_{radar_index}_{filter_type}_1992-2020.csv'
merged_data = pd.read_csv(merged_path)

# Step2: predict and calculate mean predictions
rf_preds_all_models = []
rf_preds = []

predictors = predictors_dict.get(model_name)
X = merged_data[predictors]

# load models
model_dir = '/Net/Groups/BGI/scratch/gliu/calipso/outputs/models'
models = [f for f in os.listdir(model_dir) if f.startswith(f'{model_name}_{radar_index}_{filter_type}_fold') and f.endswith('.joblib')]
for model_file in models:
    model_path = os.path.join(model_dir, model_file)
    model = joblib.load(model_path)
    preds = model.predict(X)
    rf_preds.append(preds)
# the mean predictation for all models     
mean_preds = np.mean(rf_preds, axis=0)
rf_preds_all_models = np.column_stack(rf_preds)

# add lat, lon and year info 
pred_each_model = np.column_stack((merged_data[['year', 'lat', 'lon']].values, rf_preds_all_models))
model_columns = [f'pred_model_{i+1}' for i in range(rf_preds_all_models.shape[1])]
columns = ['year', 'lat', 'lon'] + model_columns
df_preds_each_model = pd.DataFrame(pred_each_model, columns=columns)
df_preds_each_model['pred_mean'] = mean_preds
df_preds_each_model['year'] = df_preds_each_model['year'].astype(int)

ds_preds = df_preds_each_model.set_index(['lon', 'lat', 'year']).to_xarray()

# Save the Xarray Dataset to a NetCDF file
output_path = f'../outputs/predictions/predictions_agb_{model_name}_{radar_index}_{filter_type}_1992-2020_globe.nc'
ds_preds.to_netcdf(output_path)

print('Done!')

