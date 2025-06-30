
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Develope the random forest models aiming at predicting AGB from rad, tc, td, and background climate 
using leave-one-year-out cross-validation for all years:
    - target: cci-agb; predictor: rad, tc, td, clim (different for each model) 
    - random forest
    - leave one year out: train & test 
    - evaluate the models: R2, RMSE, BIC
    - evaluate the shap value

model_names = [
        'rf-rad',
        'rf-rad-tc',
        'rf-rad-td',
        'rf-rad-clim',
        'rf-rad-tc-td',
        'rf-rad-tc-clim',
        'rf-rad-td-clim',
        'rf-rad-tc-td-clim'
    ]

predictor for models:
    1. predictor: rad
    2. predictor: rad + tc
    3. predictor: rad + td
    4. predictor: rad + clim
    5. predictor: rad + tc + td
    6. predictor: rad + tc + clim
    7. predictor: rad + td + clim 
    8. predictor: rad + tc + td + clim 

Output: 
    - 8 models
    - agb prediction for testing dataset 
    - evaluation metrics for test datasets (R2, RMSE, BIC)
    - Shap values 

"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.metrics import r2_score, mean_squared_error
import joblib
import argparse
import time


def train_evaluate_model(data,model_name, radar_index=None, filter_type=None, predictors_dict=None, model_dir =None):
    predictors = predictors_dict.get(model_name)
    X = data[predictors]
    y = data['cci']
    groups = data['year']

    logo = LeaveOneGroupOut()
    r2_scores, rmse_scores, bic_scores = [], [], []
    predictions, observations, test_groups = [], [], []
    X_tests, y_tests = [], []
    
    fold = 0
    total_folds = logo.get_n_splits(groups=groups)
    start_time = time.time()
    for train_idx, test_idx in logo.split(X, y, groups):
        fold += 1
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        # train the model
        model = RandomForestRegressor(n_estimators=100, n_jobs=-1, random_state=42)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        # evaluate the model performance 
        r2 = r2_score(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        n = len(y_test)
        p = X_train.shape[1]
        bic = n * np.log(rmse**2) + p * np.log(n)

        r2_scores.append(r2)
        rmse_scores.append(rmse)
        bic_scores.append(bic)
        predictions.extend(y_pred)
        observations.extend(y_test)
        test_groups.extend(groups.iloc[test_idx])
        X_tests.append(X_test)
        y_tests.append(y_test)

        results = {
        'R2': round(np.mean(r2_scores), 3),
        'RMSE': round(np.mean(rmse_scores), 3),
        'BIC': round(np.mean(bic_scores), 2),
        'Predictions': [round(pred, 2) for pred in predictions],
        'Observations': [round(obs, 2) for obs in observations],
        'TestGroups': test_groups,
        'X_tests': X_tests,
        'y_tests': y_tests
    }

        # Save the model
        model_filename = f'{model_dir}/{model_name}_{radar_index}_{filter_type}_fold{fold}.joblib'
        joblib.dump(model, model_filename)

        elapsed_time = time.time() - start_time
        print(f'Fold {fold}/{total_folds} completed in {elapsed_time/3600:.2f} hours.')


    return results

if __name__ == "__main__":
    # # Initialize the parser
    parser = argparse.ArgumentParser(description="Parse model parameters")
    # Add arguments
    parser.add_argument('--model_name', type=str, default='rf-rad-tc', help='Name of the model')
    parser.add_argument('--radar_index', type=str, default='mean', help='Radar index to use')
    parser.add_argument('--filter_type', type=str, default='filter1-water', help='Type of filter to apply')
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

    # store to scratch
    model_dir = '/Net/Groups/BGI/scratch/gliu/calipso/outputs/models'

    # load the data 
    merged_path = f'../data/data_processed/Input_mergeddata_for_rf_model_{radar_index}_{filter_type}.csv'
    merged_data = pd.read_csv(merged_path)
    
    # train and evaluate the models 
    overall_start_time = time.time()
    evaluation_results = train_evaluate_model(merged_data,model_name, radar_index, filter_type, predictors_dict, model_dir)
    total_time = time.time() - overall_start_time

    # Store the predictions and observations for test
    output_path = '/Net/Groups/BGI/scratch/gliu/calipso/outputs'
    model_info = f'random_forest_{model_name}_{radar_index}_{filter_type}'

    df_test_pred_obs = pd.DataFrame({
        'Predictions': evaluation_results['Predictions'],
        'Observations': evaluation_results['Observations'],
        'TestGroups': evaluation_results['TestGroups']})

    pred_test_file = f'{output_path}/predictions/{model_info}_pred_testset_biomass.csv'
    df_test_pred_obs.to_csv(pred_test_file, index=False)

    # save the model evaluation
    summary_results = {
        'R2': evaluation_results['R2'],
        'RMSE': evaluation_results['RMSE'],
        'BIC': evaluation_results['BIC']
    }

    results_df = pd.DataFrame([summary_results])
    results_file = f'{output_path}/evaluation/{model_info}_evaluation.csv'
    results_df.to_csv(results_file, index=False)

    X_tests_file = f'{output_path}/shap_values/{model_info}_X_tests.pkl'
    y_tests_file = f'{output_path}/shap_values/{model_info}_y_tests.pkl'
    with open(X_tests_file, 'wb') as f:
        joblib.dump(evaluation_results['X_tests'], f)
    with open(y_tests_file, 'wb') as f:
        joblib.dump(evaluation_results['y_tests'], f)

    print(f'Total time taken: {total_time/3600:.2f} hours.')
    print('All Done!')