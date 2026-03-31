
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Develope the random forest models aiming at predicting AGB
"""

import os
import time
import argparse
import joblib
import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.model_selection import GroupKFold


def create_dirs(*paths):
    """Create directories if they do not exist."""
    for path in paths:
        os.makedirs(path, exist_ok=True)

def train_evaluate_model(data, model_name, predictors_dict, model_dir, output_path):
    """
    Train and evaluate a RandomForest model using spatiotemporal block cross-validation.
      (i) The training data (2017–2019) are divided into spatial blocks using a 0.5° grid per year.
      (ii) A spatiotemporal group ID is created combining year, longitude block, and latitude block.
      (iii) GroupKFold (with 5 folds) is used so that each spatiotemporal block is kept intact.
      (iv) For each fold, the model is trained and evaluation metrics (R2, RMSE, BIC) are computed.
      (v) A final model is trained on the full training set and evaluated on the independent 2020 dataset.
    """
    # Create output directories
    create_dirs(
        model_dir,
        os.path.join(output_path, "predictions"),
        os.path.join(output_path, "evaluation"),
        os.path.join(output_path, "shap_values")
    )

    predictors = predictors_dict.get(model_name)

    # Build a GeoDataFrame to help with spatial operations
    gdf = gpd.GeoDataFrame(
        data,
        geometry=gpd.points_from_xy(data["lon"], data["lat"]),
        crs="EPSG:4326"
    )

    # Split data: training data for 2017–2019, and independent test data for 2020.
    gdf_train = gdf[gdf["year"].isin([2017, 2018, 2019])].copy()
    gdf_test2020 = gdf[gdf["year"] == 2020].copy()

    # Define spatial block size (in degrees)
    spacing = 0.5
    # For each training record, compute block indices based on longitude and latitude.
    # Using np.floor will assign each coordinate into a discrete block
    gdf_train["block_lon"] = np.floor(gdf_train["lon"] / spacing)
    gdf_train["block_lat"] = np.floor(gdf_train["lat"] / spacing)
    # Combine the year with the spatial block indices to create a spatiotemporal group ID.
    gdf_train["group"] = gdf_train["year"].astype(str) + "_" + \
                           gdf_train["block_lon"].astype(str) + "_" + \
                           gdf_train["block_lat"].astype(str)
    groups = gdf_train["group"].values

    # Set up GroupKFold CV with 5 folds, ensuring each spatiotemporal block remains intact.
    group_kfold = GroupKFold(n_splits=5)
    folds = list(group_kfold.split(gdf_train, gdf_train["cci"], groups=groups))
    
    # Containers for cross-validation metrics and predictions.
    cv_train_r2, cv_train_rmse, cv_train_bic = [], [], []
    cv_test_r2, cv_test_rmse, cv_test_bic = [], [], []
    cv_train_preds, cv_train_obs = [], []
    cv_test_preds, cv_test_obs = [], []
    cv_test_indices = []

    # Containers for storing 2020 evaluation metrics from each fold.
    cv2020_r2, cv2020_rmse, cv2020_bic = [], [], []
    cv2020_preds = []  # List to hold 2020 predictions per fold

    start_time = time.time()
    n_folds = len(folds)

    # For each fold produced by GroupKFold, train and evaluate.
    for fold_idx, (train_idx, test_idx) in enumerate(folds):
        # Select training and test data from the training set.
        X_train_cv = gdf_train.iloc[train_idx][predictors]
        y_train_cv = gdf_train.iloc[train_idx]["cci"]
        X_test_cv = gdf_train.iloc[test_idx][predictors]
        y_test_cv = gdf_train.iloc[test_idx]["cci"]

        model = RandomForestRegressor(n_estimators=100, n_jobs=-1, random_state=42)
        model.fit(X_train_cv, y_train_cv)

        y_pred_train = model.predict(X_train_cv)
        y_pred_test = model.predict(X_test_cv)

        # Compute training metrics.
        r2_train = r2_score(y_train_cv, y_pred_train)
        rmse_train = np.sqrt(mean_squared_error(y_train_cv, y_pred_train))
        n_train = len(y_train_cv)
        p = X_train_cv.shape[1]
        bic_train = n_train * np.log(rmse_train**2) + p * np.log(n_train)

        cv_train_r2.append(r2_train)
        cv_train_rmse.append(rmse_train)
        cv_train_bic.append(bic_train)
        cv_train_preds.extend(y_pred_train)
        cv_train_obs.extend(y_train_cv)

        # Compute validation metrics.
        r2_test = r2_score(y_test_cv, y_pred_test)
        rmse_test = np.sqrt(mean_squared_error(y_test_cv, y_pred_test))
        n_test = len(y_test_cv)
        bic_test = n_test * np.log(rmse_test**2) + p * np.log(n_test)

        cv_test_r2.append(r2_test)
        cv_test_rmse.append(rmse_test)
        cv_test_bic.append(bic_test)
        cv_test_preds.extend(y_pred_test)
        cv_test_obs.extend(y_test_cv)
        cv_test_indices.extend(gdf_train.iloc[test_idx].index.tolist())

        # Evaluate on the independent 2020 dataset.
        X_test2020 = gdf_test2020[predictors]
        y_test2020 = gdf_test2020["cci"]

        y_pred_2020 = model.predict(X_test2020)
        r2_2020 = r2_score(y_test2020, y_pred_2020)
        rmse_2020 = np.sqrt(mean_squared_error(y_test2020, y_pred_2020))
        n2020 = len(y_test2020)
        bic_2020 = n2020 * np.log(rmse_2020 ** 2) + p * np.log(n2020)

        cv2020_r2.append(r2_2020)
        cv2020_rmse.append(rmse_2020)
        cv2020_bic.append(bic_2020)
        cv2020_preds.append(y_pred_2020)

        # Save the CV fold model.
        model_filename = os.path.join(model_dir, f"{model_name}_fold{fold_idx+1}.joblib")
        joblib.dump(model, model_filename)

        elapsed_time = time.time() - start_time
        print(f"GroupKFold CV Fold {fold_idx+1}/{n_folds} completed in {elapsed_time/60:.2f} minutes.")

    # Aggregate cross-validation results.
    cv_results = {
        "CV_Train_R2": round(np.mean(cv_train_r2), 3) if cv_train_r2 else None,
        "CV_Train_RMSE": round(np.mean(cv_train_rmse), 3) if cv_train_rmse else None,
        "CV_Train_BIC": round(np.mean(cv_train_bic), 2) if cv_train_bic else None,
        "CV_Test_R2": round(np.mean(cv_test_r2), 3) if cv_test_r2 else None,
        "CV_Test_RMSE": round(np.mean(cv_test_rmse), 3) if cv_test_rmse else None,
        "CV_Test_BIC": round(np.mean(cv_test_bic), 2) if cv_test_bic else None,
        "CV_Train_Predictions": cv_train_preds,
        "CV_Train_Observations": cv_train_obs,
        "CV_Test_Predictions": cv_test_preds,
        "CV_Test_Observations": cv_test_obs,
        "CV_Test_Indices": cv_test_indices
    }

    # Aggregate 2020 results across folds.
    test2020_results = {
        "Test2020_R2": round(np.mean(cv2020_r2), 3),
        "Test2020_RMSE": round(np.mean(cv2020_rmse), 3),
        "Test2020_BIC": round(np.mean(cv2020_bic), 2)
    }
    
    results = {"CV_Results": cv_results, "Test2020_Results": test2020_results}
    return results, gdf_train, predictors


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="rf-rad")
    args = parser.parse_args()

    # Define predictor sets for candidate models.
    predictors_dict = {
        "rf-rad": ["rad"],
        "rf-clim": ["MAT", "MAP"],
        "rf-rad-clim": ["rad", "MAT", "MAP"],
        "rf-rad-clim2": ["rad", "MAT", "MAP", "MAT_yr", "MAP_yr"]
    }

    # Set base output directories and input file path.
    base_output = "/Net/Groups/BGI/work_2/ForExD/CALIPSO/agb_mapping/spacetime_block_no_shap/"
    model_dir = os.path.join(base_output, "models")
    os.makedirs(model_dir, exist_ok=True)
    merged_path = '../data/data_agb_mapping/Input_mergeddata_for_rf_model_train_fin.csv'

    # Load the merged data.
    merged_data = pd.read_csv(merged_path)

    # Train and evaluate the model.
    results, gdf_train, predictors = train_evaluate_model(
        merged_data,
        args.model_name,
        predictors_dict,
        model_dir,
        base_output
    )

    model_info = f"random_forest_{args.model_name}"

    # Save CV test predictions along with extra fields.
    cv_test_df = pd.DataFrame({
        "CV_Test_Predictions": results["CV_Results"]["CV_Test_Predictions"],
        "CV_Test_Observations": results["CV_Results"]["CV_Test_Observations"],
        "Indices": results["CV_Results"]["CV_Test_Indices"]
    })
    cv_test_df["lat"] = merged_data.loc[cv_test_df["Indices"], "lat"].values
    cv_test_df["lon"] = merged_data.loc[cv_test_df["Indices"], "lon"].values
    cv_test_df["year"] = merged_data.loc[cv_test_df["Indices"], "year"].values
    cv_test_df.to_csv(os.path.join(base_output, "predictions", f"{model_info}_pred_cv_test.csv"), index=False)

    # Save evaluation metrics.
    cv_eval_df = pd.DataFrame({
        "CV_Train_R2": [results["CV_Results"]["CV_Train_R2"]],
        "CV_Train_RMSE": [results["CV_Results"]["CV_Train_RMSE"]],
        "CV_Train_BIC": [results["CV_Results"]["CV_Train_BIC"]],
        "CV_Test_R2": [results["CV_Results"]["CV_Test_R2"]],
        "CV_Test_RMSE": [results["CV_Results"]["CV_Test_RMSE"]],
        "CV_Test_BIC": [results["CV_Results"]["CV_Test_BIC"]]
    })
    cv_eval_df.to_csv(os.path.join(base_output, "evaluation", f"{model_info}_eval_cv.csv"), index=False)

    test2020_eval_df = pd.DataFrame({
        "Test2020_R2": [results["Test2020_Results"]["Test2020_R2"]],
        "Test2020_RMSE": [results["Test2020_Results"]["Test2020_RMSE"]],
        "Test2020_BIC": [results["Test2020_Results"]["Test2020_BIC"]]
    })
    test2020_eval_df.to_csv(os.path.join(base_output, "evaluation", f"{model_info}_eval_test2020.csv"), index=False)

    print(f"Training and evaluation completed. Results saved to {base_output}")
    print("All Done!")