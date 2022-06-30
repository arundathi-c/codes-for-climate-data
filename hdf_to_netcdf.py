import os

import matplotlib as mpl
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap
import numpy as np
from pyhdf.SD import SD, SDC
from mpl_toolkits.basemap import Basemap
import matplotlib.pyplot as plt
import numpy as np
import warnings
import matplotlib.cbook
import matplotlib.ticker as mticker

warnings.filterwarnings("ignore", category=matplotlib.cbook.mplDeprecation)
from netCDF4 import Dataset
import matplotlib.cm as cm
import shapefile as shp
import shapely.geometry as sgeom
import cartopy.feature as cfeature
import cartopy.crs as ccrs
from matplotlib.ticker import FuncFormatter
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
import matplotlib.ticker as ticker
import matplotlib as mpl
import matplotlib.colors
from matplotlib.colors import ListedColormap, LinearSegmentedColormap
from scipy.interpolate import interp1d

###############################################################################################################################################################################################
## AIRS TEMPERATURE DATASET
DATAFIELD_NAME_A = 'Temperature_A'

airs_01 = SD(r'./airs.2010.01.hdf', SDC.READ)
airs_02 = SD(r'./airs.2010.02.hdf', SDC.READ)
airs_12 = SD(r'./airs.2010.12.hdf', SDC.READ)
airs_winter_temp = np.average(np.stack((airs_01.select(DATAFIELD_NAME_A)[3, :, :],\
                                        airs_02.select(DATAFIELD_NAME_A)[3, :, :],\
                                        airs_12.select(DATAFIELD_NAME_A)[3, :, :]), axis=0), axis=0)

airs_03 = SD(r'./airs.2010.03.hdf', SDC.READ)
airs_04 = SD(r'./airs.2010.04.hdf', SDC.READ)
airs_05 = SD(r'./airs.2010.05.hdf', SDC.READ)
airs_spring_temp = np.average(np.stack((airs_03.select(DATAFIELD_NAME_A)[3, :, :],\
                                        airs_04.select(DATAFIELD_NAME_A)[3, :, :],\
                                        airs_05.select(DATAFIELD_NAME_A)[3, :, :]), axis=0), axis=0)

airs_06 = SD(r'./airs.2010.06.hdf', SDC.READ)
airs_07 = SD(r'./airs.2010.07.hdf', SDC.READ)
airs_08 = SD(r'./airs.2010.08.hdf', SDC.READ)
airs_summer_temp = np.average(np.stack((airs_06.select(DATAFIELD_NAME_A)[3, :, :],\
                                        airs_07.select(DATAFIELD_NAME_A)[3, :, :],\
                                        airs_08.select(DATAFIELD_NAME_A)[3, :, :]), axis=0), axis=0)

airs_09 = SD(r'./airs.2010.09.hdf', SDC.READ)
airs_10 = SD(r'./airs.2010.10.hdf', SDC.READ)
airs_11 = SD(r'./airs.2010.11.hdf', SDC.READ)
airs_autumn_temp = np.average(np.stack((airs_09.select(DATAFIELD_NAME_A)[3, :, :],\
                                        airs_10.select(DATAFIELD_NAME_A)[3, :, :],\
                                        airs_11.select(DATAFIELD_NAME_A)[3, :, :]), axis=0), axis=0)

airs = SD(r'./airs.2010.12.hdf', SDC.READ).select(DATAFIELD_NAME_A)

attrs = airs.attributes(full=1)
fillvalue = attrs["_FillValue"]
fv = fillvalue[0]

airs_winter_temp[airs_winter_temp == fv] = np.nan
airs_winter_temp[airs_winter_temp < 0.0] = np.nan
airs_winter_temp = np.ma.masked_array(airs_winter_temp, np.isnan(airs_winter_temp))

airs_spring_temp[airs_spring_temp == fv] = np.nan
airs_spring_temp[airs_spring_temp < 0.0] = np.nan
airs_spring_temp = np.ma.masked_array(airs_spring_temp, np.isnan(airs_spring_temp))

airs_summer_temp[airs_summer_temp == fv] = np.nan
airs_summer_temp[airs_summer_temp < 0.0] = np.nan
airs_summer_temp = np.ma.masked_array(airs_summer_temp, np.isnan(airs_summer_temp))

airs_autumn_temp[airs_autumn_temp == fv] = np.nan
airs_autumn_temp[airs_autumn_temp < 0.0] = np.nan
airs_autumn_temp = np.ma.masked_array(airs_autumn_temp, np.isnan(airs_autumn_temp))

## AIRS WATER VAPOUR DATASET
DATAFIELD_NAME_B = 'H2O_MMR_A'

airs_01 = SD(r'./airs.2010.01.hdf', SDC.READ)
airs_02 = SD(r'./airs.2010.02.hdf', SDC.READ)
airs_12 = SD(r'./airs.2010.12.hdf', SDC.READ)
airs_winter_wv = np.average(np.stack((airs_01.select(DATAFIELD_NAME_B)[3, :, :],\
                                      airs_02.select(DATAFIELD_NAME_B)[3, :, :],\
                                      airs_12.select(DATAFIELD_NAME_B)[3, :, :]), axis=0), axis=0)

airs_03 = SD(r'./airs.2010.03.hdf', SDC.READ)
airs_04 = SD(r'./airs.2010.04.hdf', SDC.READ)
airs_05 = SD(r'./airs.2010.05.hdf', SDC.READ)
airs_spring_wv = np.average(np.stack((airs_03.select(DATAFIELD_NAME_B)[3, :, :],\
                                      airs_04.select(DATAFIELD_NAME_B)[3, :, :],\
                                      airs_05.select(DATAFIELD_NAME_B)[3, :, :]), axis=0), axis=0)

airs_06 = SD(r'./airs.2010.06.hdf', SDC.READ)
airs_07 = SD(r'./airs.2010.07.hdf', SDC.READ)
airs_08 = SD(r'./airs.2010.08.hdf', SDC.READ)
airs_summer_wv = np.average(np.stack((airs_06.select(DATAFIELD_NAME_B)[3, :, :],\
                                      airs_07.select(DATAFIELD_NAME_B)[3, :, :],\
                                      airs_08.select(DATAFIELD_NAME_B)[3, :, :]), axis=0), axis=0)

airs_09 = SD(r'./airs.2010.09.hdf', SDC.READ)
airs_10 = SD(r'./airs.2010.10.hdf', SDC.READ)
airs_11 = SD(r'./airs.2010.11.hdf', SDC.READ)
airs_autumn_wv = np.average(np.stack((airs_09.select(DATAFIELD_NAME_B)[3, :, :],\
                                      airs_10.select(DATAFIELD_NAME_B)[3, :, :],\
                                      airs_11.select(DATAFIELD_NAME_B)[3, :, :]), axis=0), axis=0)

airs_1 = SD(r'./airs.2010.12.hdf', SDC.READ).select(DATAFIELD_NAME_B)

attrs = airs_1.attributes(full=1)
fillvalue = attrs["_FillValue"]
fv = fillvalue[0]

airs_winter_wv[airs_winter_wv == fv] = np.nan
airs_winter_wv[airs_winter_wv < 0.0] = np.nan
airs_winter_wv = np.ma.masked_array(airs_winter_wv, np.isnan(airs_winter_wv))

airs_spring_wv[airs_spring_wv == fv] = np.nan
airs_spring_wv[airs_spring_wv < 0.0] = np.nan
airs_spring_wv = np.ma.masked_array(airs_spring_wv, np.isnan(airs_spring_wv))

airs_summer_wv[airs_summer_wv == fv] = np.nan
airs_summer_wv[airs_summer_wv < 0.0] = np.nan
airs_summer_wv = np.ma.masked_array(airs_summer_wv, np.isnan(airs_summer_wv))

airs_autumn_wv[airs_autumn_wv == fv] = np.nan
airs_autumn_wv[airs_autumn_wv < 0.0] = np.nan
airs_autumn_wv = np.ma.masked_array(airs_autumn_wv, np.isnan(airs_autumn_wv))

lats = SD(r'./airs.2010.12.hdf', SDC.READ).select('Latitude')
latitude = lats[:, 0]
lons = SD(r'./airs.2010.12.hdf', SDC.READ).select('Longitude')
longitude = lons[0, :]


fn = r'./airs.winter.nc'
ds1 = Dataset(fn, 'w', format='NETCDF4')
lat = ds1.createDimension('lat', 10)
lon = ds1.createDimension('lon', 10)
lats = ds1.createVariable('lat', 'f4', ('lat',))
lons = ds1.createVariable('lon', 'f4', ('lon',))
temp = ds1.createVariable('temp', 'f4', ('lat', 'lon',))
wv = ds1.createVariable('wv', 'f4', ('lat', 'lon',))
temp.units = 'K'
wv.units = 'g/kg'

lats[:] = latitude
lons[:] = longitude
temp[:,:] = airs_winter_temp
wv[:,:] = airs_winter_wv
ds1.close()

fn = r'./airs.spring.nc'
ds2 = Dataset(fn, 'w', format='NETCDF4')
lat = ds2.createDimension('lat', 10)
lon = ds2.createDimension('lon', 10)
lats = ds2.createVariable('lat', 'f4', ('lat',))
lons = ds2.createVariable('lon', 'f4', ('lon',))
temp = ds2.createVariable('temp', 'f4', ('lat', 'lon',))
wv = ds2.createVariable('wv', 'f4', ('lat', 'lon',))
temp.units = 'K'
wv.units = 'g/kg'

lats[:] = latitude
lons[:] = longitude
temp[:,:] = airs_spring_temp
wv[:,:] = airs_spring_wv
ds2.close()

fn = r'./airs.summer.nc'
ds3 = Dataset(fn, 'w', format='NETCDF4')
lat = ds3.createDimension('lat', 10)
lon = ds3.createDimension('lon', 10)
lats = ds3.createVariable('lat', 'f4', ('lat',))
lons = ds3.createVariable('lon', 'f4', ('lon',))
temp = ds3.createVariable('temp', 'f4', ('lat', 'lon',))
wv = ds3.createVariable('wv', 'f4', ('lat', 'lon',))
temp.units = 'K'
wv.units = 'g/kg'

lats[:] = latitude
lons[:] = longitude
temp[:,:] = airs_summer_temp
wv[:,:] = airs_summer_wv
ds3.close()

fn = r'./airs.autumn.nc'
ds4 = Dataset(fn, 'w', format='NETCDF4')
lat = ds4.createDimension('lat', 10)
lon = ds4.createDimension('lon', 10)
lats = ds4.createVariable('lat', 'f4', ('lat',))
lons = ds4.createVariable('lon', 'f4', ('lon',))
temp = ds4.createVariable('temp', 'f4', ('lat', 'lon',))
wv = ds4.createVariable('wv', 'f4', ('lat', 'lon',))
temp.units = 'K'
wv.units = 'g/kg'

lats[:] = latitude
lons[:] = longitude
temp[:,:] = airs_autumn_temp
wv[:,:] = airs_autumn_wv
ds4.close()
