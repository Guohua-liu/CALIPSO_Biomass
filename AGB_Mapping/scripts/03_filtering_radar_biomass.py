
# """
# Filterring the satellite radar & biomass data: 
#    
#      1. Data: radar  & biomass data   
#          - radar data: yearly mean, max data  
#          - biomass data:  CCI (2010, 2017-2020), Saatchi (2010, 2017-2019)
#          
#  
#      2. Filtering #1: Mask wetland & peatland & bare vegetation area
#          wetland: 
#              - 500m to 0.083d
#              - mask: regularly flooded wetlands or lake > 80%
# 
#          peatland: 
#              - regid to the same grids
#              - remove grids > 10%
#
#          bare areas:
#              - ESA-CCI land cover map 
#              - remove bare areas: value is 200 
#
# """

import os
import numpy as np
import xarray as xr
import rioxarray as rio

# load the data 
# ---------- rad signal 
rad_index = 'max'  # 'mean' or 'max'
in_path_rad = f'../data/radsigdata_terraclim/global_nc_from_tif/Global_RadarSignal_yearly_1992_2022_{rad_index}_0d083.nc'
ds_rad = xr.open_dataset(in_path_rad)
ds_rad = ds_rad.sel(year=slice(1993, 2020))
ds_rad

# ---------- biomass - ESA-CCI 
in_path_agb = '../data/data_processed/ESACCI-AGB-Map-V4_0d083_2010-2017-2020.nc'
ds_cci = xr.open_dataset(in_path_agb)
ds_cci = ds_cci[['lon', 'lat', 'year', 'AGB']]

# ----------- wetland map [Contains three values (0, 1 and 2) 0: non-wetlands, 1: Regularly flooded wetlands (RFWs), 2: Lakes]
in_path_peat = '../data/data_processed/ESACCI-LC-L4-LCCS-Map-1992_2015-v2.0.7.nc' 
ds_wetland = xr.open_dataset(in_path_peat)
ds_wetland['wetland'] = ds_wetland['__xarray_dataarray_variable__']
ds_wetland = ds_wetland.drop_vars(['__xarray_dataarray_variable__'])

# ----------- peatland map 
in_path_peat = '../data/data_processed/Peat-ML_global_peatland_olson-Map-0d083.nc'
ds_peat = xr.open_dataset(in_path_peat)
ds_peat = ds_peat[['lon', 'lat', 'PEATLAND_P']]

# ---------- land cover map (ESA-CCI)
in_path_lc = '../data/data_processed/ESACCI-LC-L4-LCCS-Map-0d083-2001.nc'
ds_lc = xr.open_dataset(in_path_lc)
ds_lc = ds_lc[['lon', 'lat', 'lc']]

# ------------ Mask for radar signal
# Filtering #1
radar_masked_f1 = xr.where(ds_wetland== 0, ds_rad['RadarSignal'], np.nan)
radar_masked_f1 = xr.where(ds_peat['PEATLAND_P'] <= 10, radar_masked_f1, np.nan)
radar_masked_f1 = xr.where((ds_lc['lc'] != 200), radar_masked_f1, np.nan)
radar_masked_f1 = radar_masked_f1.rename({'wetland': 'RadarSignal'})

radar_masked_f1.to_netcdf(f'../data/data_processed/Radar_{rad_index}_0d083_global_1992-2020_filter1-water.nc')


# ------------ Mask for biomass data
# Filtering #1
ds_cci = ds_cci.interp_like(ds_wetland['wetland'])
cci_masked_f1 = xr.where(ds_wetland['wetland']== 0, ds_cci, np.nan)
cci_masked_f1 = xr.where(ds_peat['PEATLAND_P'] <= 10, cci_masked_f1, np.nan)
cci_masked_f1 = xr.where((ds_lc['lc'] != 200), cci_masked_f1, np.nan)
cci_masked_f1['year'] = cci_masked_f1['year'].dt.year.astype(int)

cci_masked_f1.to_netcdf('../data/data_processed/ESA-CCI_0d083_global_2010-2017-2020_filter1-water.nc')

print('done!')