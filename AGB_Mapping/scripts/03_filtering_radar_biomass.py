
# """
# Filterring the satellite radar & biomass data: 
#      1. Data: radar  & biomass data   
#          - radar data: wet sean mean  
#          - biomass data:  CCI (2010, 2017-2020)
#          
#      2. Filtering #1: Mask wetland & peatland & bare vegetation area 
#          non-forest: tree cover < 10% based on ESA-CCI PFT percentage map
#          wetland: regularly flooded wetlands or lake > 80%
#          peatland: > 10%
#          urban areas: >=50%
#          crop land: >=50%
#          abnormal radar backscatter 
#
# """

import os
import numpy as np
import xarray as xr
import rioxarray as rio

# load the data 

# radar signal data (wet season)
radar_inpath = '../data/data_agb_mapping/radar/Global_RadarSignal_1993_2020_8_9km_wet_season_local_mean_esacci_filled_gaps.nc'
ds_radar = xr.open_dataset(radar_inpath) 

# ESA-CCI AGB data (masked poor quality)
agb_inpath = '../data/data_agb_mapping/agb_esacci/ESACCI-AGB-Map-V6_0d08186_2017-2020_masked_cv_80_aligned.nc'
ds_agb = xr.open_dataset(agb_inpath)

# ESA-CCI PFT maps
pft_inpath =  "../data/data_agb_mapping/pft_esacci/ESACCI-PFT-Map-8_9km_1992-2020_aligned.nc"
ds_pft = xr.open_dataset(pft_inpath)
# Trees total
ds_trees = ds_pft['TREES-BD'] + ds_pft['TREES-BE'] + ds_pft['TREES-ND'] + ds_pft['TREES-NE']
# Barren area
ds_barren = ds_pft['BARE']
# Urban areas 
ds_urban = ds_pft['BUILT']

# load the wetland map 
wet_inpath = "../data/data_agb_mapping/wetland_global_8_9km_aligned.nc"
ds_wet = xr.open_dataset(wet_inpath)

# load the peatland map 
peat_inpath = "../data/data_agb_mapping/peatland_global_8_9km_aligned.nc"
ds_peat = xr.open_dataset(peat_inpath)

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

radar_masked_fin.to_netcdf("../data/data_agb_mapping/global_radar_data_1993-2020_filtered_fin.nc")
agb_masked_fin.to_netcdf("../data/data_agb_mapping/global_agb_data_2017-2020_filtered_fin.nc")

print('done!')