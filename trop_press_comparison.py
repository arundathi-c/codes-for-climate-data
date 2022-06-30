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
warnings.filterwarnings("ignore",category=matplotlib.cbook.mplDeprecation)
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


fig, axs = plt.subplots(2,4,figsize=(12.5, 5.5),sharex=True,sharey=True)
axs[0,0].set_xlabel("Winter(DJF)",size = 8)
axs[0,1].set_xlabel("Spring(MAM)",size = 8)
axs[0,2].set_xlabel("Summer(JJA)",size = 8)
axs[0,3].set_xlabel("Autumn(SON)",size = 8)
axs[0,0].set_ylabel("AIRS(hPa)",size=8)
axs[1,0].set_ylabel("GEOS-Chem(hPa)",size=8)
axs[0,0].xaxis.set_label_coords(0.5,1.2)
axs[0,1].xaxis.set_label_coords(0.5,1.2)
axs[0,2].xaxis.set_label_coords(0.5,1.2)
axs[0,3].xaxis.set_label_coords(0.5,1.2)


###############################################################################################################################################################################################
## AIRS DATASET IMPORTING

airs_winter = Dataset(r"D:\GEOS-Chem Validation\AIRS\airs.winter.regrid.nc")
latitude = airs_winter.variables['lat']
longitude = airs_winter.variables['lon']
airs_winter_trop_p = airs_winter.variables['trop_p'][:,:]
airs_winter_trop_p[airs_winter_trop_p > 1000] = np.nan

airs_spring = Dataset(r"D:\GEOS-Chem Validation\AIRS\airs.spring.regrid.nc")
airs_spring_trop_p = airs_spring.variables['trop_p'][:,:]
airs_spring_trop_p[airs_spring_trop_p > 1000] = np.nan


airs_summer = Dataset(r"D:\GEOS-Chem Validation\AIRS\airs.summer.regrid.nc")
airs_summer_trop_p = airs_summer.variables['trop_p'][:,:]
airs_summer_trop_p[airs_summer_trop_p > 1000] = np.nan


airs_autumn = Dataset(r"D:\GEOS-Chem Validation\AIRS\airs.autumn.regrid.nc")
airs_autumn_trop_p = airs_autumn.variables['trop_p'][:,:]
airs_autumn_trop_p[airs_autumn_trop_p > 1000] = np.nan


###############################################################################################################################################

## READING IN GEOS CHEM DATASETS AND VARIABLES
data_model = Dataset(r"D:\GEOS-Chem Validation\met_send\GEOSChem.StateMet.2010.airs.nc4")
lats_model = data_model.variables['lat']
lons_model = data_model.variables['lon']
time_model = data_model.variables['time']

trop_p_model_winter = np.average(data_model.variables['Met_TropP'][[0,1,11],:,:],axis = 0)
trop_p_model_winter[np.isnan(airs_winter_trop_p)] = np.nan
trop_p_model_spring = np.average(data_model.variables['Met_TropP'][2:5,:,:],axis = 0)
trop_p_model_spring[np.isnan(airs_spring_trop_p)] = np.nan
trop_p_model_summer = np.average(data_model.variables['Met_TropP'][5:8,:,:],axis = 0)
trop_p_model_summer[np.isnan(airs_spring_trop_p)] = np.nan
trop_p_model_autumn = np.average(data_model.variables['Met_TropP'][8:11,:,:],axis = 0)
trop_p_model_autumn[np.isnan(airs_autumn_trop_p)] = np.nan

############################################################################################################################################################
## SETTING THE COLORS FOR THE COLORBAR
class FixPointNormalize(matplotlib.colors.Normalize):
    """
    Inspired by https:\\stackoverflow.com\questions\20144529\shifted-colorbar-matplotlib
    Subclassing Normalize to obtain a colormap with a fixpoint
    somewhere in the middle of the colormap.

    This may be useful for a `terrain` map, to set the "sea level"
    to a color in the blue\turquise range.
    """
    def __init__(self, vmin=None, vmax=None, sealevel=0, col_val = 0.21875, clip=False):
        # sealevel is the fix point of the colormap (in data units)
        self.sealevel = sealevel
        # col_val is the color value in the range [0,1] that should represent the sealevel.
        self.col_val = col_val
        matplotlib.colors.Normalize.__init__(self, vmin, vmax, clip)

    def __call__(self, value, clip=None):
        x, y = [self.vmin, self.sealevel, self.vmax], [0, self.col_val, 1]
        return np.ma.masked_array(np.interp(value, x, y))

# Combine the lower and upper range of the terrain colormap with a gap in the middle
# to let the coastline appear more prominently.
# inspired by https:\\stackoverflow.com\questions\31051488\combining-two-matplotlib-colormaps
colors_undersea = plt.cm.terrain(np.linspace(0, 0.17, 56))
colors_land = plt.cm.terrain(np.linspace(0.25, 1, 200))
# combine them and build a new colormap
colors = np.vstack((colors_undersea, colors_land))
cut_terrain_map = matplotlib.colors.LinearSegmentedColormap.from_list('cut_terrain', colors)
norm = FixPointNormalize(sealevel=0, vmax=5500)


colors = (['#0063FF','#009696','#00C633','#63FF00','#96FF00','#C6FF33','#FFFF00','#FFC600','#FFA000','#FF7C00','#FF1900'])

cmap_name = 'my_list'
cm = LinearSegmentedColormap.from_list(cmap_name, colors, N=100)

#############################################################################################################################################################################
levels1 = [90,105,120,135,150,165,180,195,210,225,240,450]

map = Basemap(resolution='c',projection='cyl', llcrnrlat=0.,urcrnrlat=40.,llcrnrlon=60.,urcrnrlon=100., suppress_ticks=False,ax = axs[0,0])
lon, lat = np.meshgrid(longitude, latitude)
xi, yi = map(lon, lat)
map.readshapefile(r'D:\GEOS-Chem Validation\India_world_full_kashmir\India_world_full_kashmir', 'India_world_full_kashmir') # drawbounds = False)
im_tropp = map.contourf(xi,yi,airs_winter_trop_p, extend='both', origin='lower', cmap=cm,levels = levels1)

map = Basemap(resolution='c',projection='cyl', llcrnrlat=0.,urcrnrlat=40.,llcrnrlon=60.,urcrnrlon=100., suppress_ticks=False,ax = axs[0,1])
lon, lat = np.meshgrid(longitude, latitude)
xi, yi = map(lon, lat)
map.readshapefile(r'D:\GEOS-Chem Validation\India_world_full_kashmir\India_world_full_kashmir', 'India_world_full_kashmir') # drawbounds = False)
im_tropp = map.contourf(xi,yi,airs_spring_trop_p, extend='both', origin='lower', cmap=cm,levels = levels1)

map = Basemap(resolution='c',projection='cyl', llcrnrlat=0.,urcrnrlat=40.,llcrnrlon=60.,urcrnrlon=100., suppress_ticks=False,ax = axs[0,2])
lon, lat = np.meshgrid(longitude, latitude)
xi, yi = map(lon, lat)
map.readshapefile(r'D:\GEOS-Chem Validation\India_world_full_kashmir\India_world_full_kashmir', 'India_world_full_kashmir') # drawbounds = False)
im_tropp = map.contourf(xi,yi,airs_summer_trop_p, extend='both', origin='lower', cmap=cm,levels = levels1)

map = Basemap(resolution='c',projection='cyl', llcrnrlat=0.,urcrnrlat=40.,llcrnrlon=60.,urcrnrlon=100., suppress_ticks=False,ax = axs[0,3])
lon, lat = np.meshgrid(longitude, latitude)
xi, yi = map(lon, lat)
map.readshapefile(r'D:\GEOS-Chem Validation\India_world_full_kashmir\India_world_full_kashmir', 'India_world_full_kashmir') # drawbounds = False)
im_tropp = map.contourf(xi,yi,airs_autumn_trop_p, extend='both', origin='lower', cmap=cm,levels = levels1)

map = Basemap(resolution='c',projection='cyl', llcrnrlat=0.,urcrnrlat=40.,llcrnrlon=60.,urcrnrlon=100., suppress_ticks=False,ax = axs[1,0])
lon, lat = np.meshgrid(lons_model, lats_model)
xi, yi = map(lon, lat)
map.readshapefile(r'D:\GEOS-Chem Validation\India_world_full_kashmir\India_world_full_kashmir', 'India_world_full_kashmir') # drawbounds = False)
im_tropp = map.contourf(xi,yi,trop_p_model_winter, extend='both', origin='lower', cmap=cm,levels = levels1)

map = Basemap(resolution='c',projection='cyl', llcrnrlat=0.,urcrnrlat=40.,llcrnrlon=60.,urcrnrlon=100., suppress_ticks=False,ax = axs[1,1])
lon, lat = np.meshgrid(lons_model, lats_model)
xi, yi = map(lon, lat)
map.readshapefile(r'D:\GEOS-Chem Validation\India_world_full_kashmir\India_world_full_kashmir', 'India_world_full_kashmir') # drawbounds = False)
im_tropp = map.contourf(xi,yi,trop_p_model_spring, extend='both', origin='lower', cmap=cm,levels = levels1)

map = Basemap(resolution='c',projection='cyl', llcrnrlat=0.,urcrnrlat=40.,llcrnrlon=60.,urcrnrlon=100., suppress_ticks=False,ax = axs[1,2])
lon, lat = np.meshgrid(lons_model, lats_model)
xi, yi = map(lon, lat)
map.readshapefile(r'D:\GEOS-Chem Validation\India_world_full_kashmir\India_world_full_kashmir', 'India_world_full_kashmir') # drawbounds = False)
im_tropp = map.contourf(xi,yi,trop_p_model_summer, extend='both', origin='lower', cmap=cm,levels = levels1)

map = Basemap(resolution='c',projection='cyl', llcrnrlat=0.,urcrnrlat=40.,llcrnrlon=60.,urcrnrlon=100., suppress_ticks=False,ax = axs[1,3])
lon, lat = np.meshgrid(lons_model, lats_model)
xi, yi = map(lon, lat)
map.readshapefile(r'D:\GEOS-Chem Validation\India_world_full_kashmir\India_world_full_kashmir', 'India_world_full_kashmir') # drawbounds = False)
im_tropp = map.contourf(xi,yi,trop_p_model_autumn, extend='both', origin='lower', cmap=cm,levels = levels1)

## LATITUDE AND LONGITUDE FORMATTING
for i in range(2):
    for j in range(4):
        # axs[i][j].xaxis.set_ticks_position('bottom')
        axs[i][j].tick_params(which='major', width=1.00, length=7, labelsize=8, top=True)
        axs[i][j].tick_params(which='minor', width=0.75, length=4.5, labelsize=8, top=True)

        # axs[i][j].yaxis.set_ticks_position('left')
        axs[i][j].tick_params(which='major', width=1.00, length=7, labelsize=8, right=True)
        axs[i][j].tick_params(which='minor', width=0.75, length=4.5, labelsize=8, right=True)

        axs[i][j].xaxis.set_major_locator(ticker.AutoLocator())
        axs[i][j].xaxis.set_minor_locator(ticker.AutoMinorLocator())

        axs[i][j].yaxis.set_major_locator(ticker.AutoLocator())
        axs[i][j].yaxis.set_minor_locator(ticker.AutoMinorLocator())

        axs[i][j].xaxis.set_major_formatter(LONGITUDE_FORMATTER)
        axs[i][j].yaxis.set_major_formatter(LATITUDE_FORMATTER)

        # axs[i][j].set_xlabel('Longitude', fontsize=9, fontweight='bold')
        # axs[i][j].set_ylabel('Latitude', fontsize=9, fontweight='bold')

plt.tight_layout(h_pad=1,w_pad = 2)

cbar1 = plt.colorbar(im_tropp, cmap=cm,ax = axs[0:2,:],pad = 0.05,ticks = [90,105,120,135,150,165,180,195,210,225,240,450])

plt.savefig('trop_p.pdf')
