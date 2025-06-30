This document provides guidelines for the code used to map above-ground biomass based on radar backscatter data. Figure 1 illustrates the workflow of whole data process and model building. Each script is described below.
**Step 1: Data Preprocessing**
  1.	01_radar_reformat_to_nc.py
  Converts radar backscatter data from TIFF format to netCDF format.
  2.	02_radar_resample_mon2yr.py
  Aggregates monthly radar backscatter data to annual data.
  3.	03_filtering_radar_biomass.py
  Filters the biomass and radar data.
  4.	04_merge_data_for_rf.py
  Prepares input data for training random forest models and predicting above-ground biomass.
**Step 2: Building the Random Forest Model**
  1.	05_rf_train_agb_from_rad.py
  Trains the random forest model for above-ground biomass.
  2.	06_rf_pred_agb.py
  Predicts above-ground biomass using the trained random forest models based on the radar backscatter data.

[https://github.com/Guohua-liu/CALIPSO_Biomass/raw/main/AGB_Mapping/figs/Fig_workflow-with-code.png](https://github.com/Guohua-liu/CALIPSO_Biomass/blob/main/AGB_Mapping/figs/Fig_workflow_with-code.png)

 Figure 1. Workflow for Mapping Above-Ground Biomass from Radar Backscatter Data
