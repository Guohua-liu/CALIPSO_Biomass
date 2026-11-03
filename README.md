# Code Instructions
This repository contains the code for the manuscript:
Liu, G., Ciais, P., Tao, S., Yang, H., Xiao, C., and Bastos, A.: A Moderate-Resolution, Long-Term Global Radar-Based Forest Above-Ground Biomass Dataset from 1993 to 2020, Earth Syst. Sci. Data, https://doi.org/10.5194/essd-2026-246, 2026.

This document describes the code used to map above-ground biomass (AGB) from radar backscatter data. Figure 1 shows the data-processing workflow and model-building steps.

## 1. Setup and requirements
Install Python and the required packages before running the scripts. The project includes:
- `environment.yml`, which defines the Conda environment.
- `requirements.txt`, which lists the required packages and their versions.

Please create the Conda environment from this file and activate it:
```bash
conda env create -f environment.yml
conda activate agb-mapping
```

## 2. Prepare the input data
The workflow uses the following datasets:
- Satellite radar backscatter (Tao et al., 2023)
- ESA CCI biomass, version 6 (Santoro and Cartus, 2025)
- TerraClimate (Abatzoglou et al., 2018)
- ESA CCI soil moisture, version 07.1 (Preimesberger et al., 2021)
- ESA CCI land cover (Harper et al., 2023)
- Wetland dataset (Tootchi et al., 2019)
- Peatland dataset (Melton et al., 2022)

Download the original datasets and use CDO to aggregate them to 8.9 km resolution, matching the satellite radar backscatter data. Place the processed datasets in the `InputData` folder.

## 3. Run the scripts
All scripts are in the `Scripts` folder. Run them in the order listed below.

### Step 1: Data preprocessing
1. `01_radar_wetseason.py`  
   Calculates the mean radar backscatter for wet-season months from the monthly radar data.
2. `02_cci_agb_mask.py`  
   Masks low-quality ESA CCI above-ground biomass data.
3. `03_filtering_radar_biomass.py`  
   Filters the biomass and radar data.
4. `04_merge_data_for_rf.py`  
   Prepares the input data for training the random forest models and predicting above-ground biomass.

### Step 2: Build and run the random forest model
1. `05_rf_train_agb_from_rad.py`  
   Trains random forest models to estimate above-ground biomass.
2. `06_rf_pred_agb.py`  
   Uses the trained random forest models and radar backscatter data to predict above-ground biomass.

**Figure 1.** Workflow for mapping above-ground biomass from radar backscatter data.
[https://github.com/Guohua-liu/CALIPSO_Biomass/blob/main/AGB_Mapping/figs/Fig1.%20workflow.png]


