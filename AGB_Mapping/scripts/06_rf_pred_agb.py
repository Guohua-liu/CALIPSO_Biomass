
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Predicting Biomass from radar and other variables using random forest models

"""
import os
import numpy as np
import pandas as pd
import joblib
import argparse

parser = argparse.ArgumentParser(description="Parse model parameters")
# Add arguments
parser.add_argument('--model_name', type=str, default='rf-rad', help='Name of the model')

# Parse the arguments
args = parser.parse_args()

# Assign parsed arguments to variables
model_name = args.model_name

predictors_dict = {
        "rf-rad": ["rad"],
        "rf-clim": ["MAT", "MAP"],
        "rf-rad-clim": ["rad", "MAT", "MAP"],
        "rf-rad-clim2": ["rad", "MAT", "MAP", "MAT_yr", "MAP_yr"]
    }
# load the flated data 
merged_path = '../data/data_agb_mapping/Input_mergeddata_for_rf_pred_forest_gap_filled_interpolated_v2.csv'

merged_data = pd.read_csv(merged_path)

# Step2: predict and calculate mean predictions
rf_preds_all_models = []
rf_preds = []

predictors = predictors_dict.get(model_name)
X = merged_data[predictors]

# load models
base_output = "/Net/Groups/BGI/work_2/ForExD/CALIPSO/agb_mapping/spacetime_block_no_shap/"
model_dir = os.path.join(base_output, "models")

models = [f for f in os.listdir(model_dir) if f.startswith(f'{model_name}_fold') and f.endswith('.joblib')]
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
output_path = f'../outputs/agb_mapping/spacetime_block/predictions_agb_{model_name}_1992-2020_globe_forest.nc'
ds_preds.to_netcdf(output_path)

print('Done!')


