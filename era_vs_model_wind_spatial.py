from __future__ import (absolute_import, division, print_function)

from __future__ import unicode_literals
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

fig, ax = plt.subplots(4,4,figsize=(12, 10),sharex=True,sharey=True)
ax[0,0].set_ylabel("ERA Interim(m/s)",size=8)
ax[1,0].set_ylabel("GEOS-Chem(m/s)",size=8)
ax[2,0].set_ylabel("ERA Interim(m/s)",size=8)
ax[3,0].set_ylabel("GEOS-Chem(m/s)",size=8)
ax[0,0].set_xlabel("Winter(DJF)",size = 8)
ax[0,1].set_xlabel("Spring(MAM)",size = 8)
ax[0,2].set_xlabel("Summer(JJA)",size = 8)
ax[0,3].set_xlabel("Autumn(SON)",size = 8)
ax[0,0].xaxis.set_label_coords(0.5,1.3)
ax[0,1].xaxis.set_label_coords(0.5,1.3)
ax[0,2].xaxis.set_label_coords(0.5,1.3)
ax[0,3].xaxis.set_label_coords(0.5,1.3)

ax[0,1].set_title("Zonal Wind Component",loc = 'right',fontdict={'weight' : 'bold','size':8,},position=(1.8,1.05))
ax[2,1].set_title("Meridional Wind Component",loc = 'right',fontdict={'weight' : 'bold','size':8,},position=(1.8,1.05))

## LOADING REANALYSIS DATA
data_era = Dataset(r'D:\GEOS-Chem Validation\era.interim.2010.full.nc', mode='r')
lons_era = data_era.variables['longitude']
lats_era = data_era.variables['latitude']
time_era = data_era.variables['time']
u_era_winter = np.average(data_era.variables['u'][[0,1,11],25,:,:],axis = 0)
u_era_spring = np.average(data_era.variables['u'][2:5,25,:,:],axis = 0)
u_era_summer = np.average(data_era.variables['u'][5:8,25,:,:],axis = 0)
u_era_autumn = np.average(data_era.variables['u'][8:11,25,:,:],axis = 0)

v_era_winter = np.average(data_era.variables['v'][[0,1,11],25,:,:],axis = 0)
v_era_spring = np.average(data_era.variables['v'][2:5,25,:,:],axis = 0)
v_era_summer = np.average(data_era.variables['v'][5:8,25,:,:],axis = 0)
v_era_autumn = np.average(data_era.variables['v'][8:11,25,:,:],axis = 0)

data_model = Dataset(r"D:\GEOS-Chem Validation\met_send\GEOSChem.StateMet.2010.700hPa.nc4")
## LOADING MODEL DATA
lats_model = data_model.variables['lat']
lons_model = data_model.variables['lon']
time_model = data_model.variables['time']

u_model_winter = np.average(data_model.variables['Met_U'][[0,1,11],0,:,:],axis = 0)
u_model_spring = np.average(data_model.variables['Met_U'][2:5,0,:,:],axis = 0)
u_model_summer = np.average(data_model.variables['Met_U'][5:8,0,:,:],axis = 0)
u_model_autumn = np.average(data_model.variables['Met_U'][8:11,0,:,:],axis = 0)

v_model_winter = np.average(data_model.variables['Met_V'][[0,1,11],0,:,:],axis = 0)
v_model_spring = np.average(data_model.variables['Met_V'][2:5,0,:,:],axis = 0)
v_model_summer = np.average(data_model.variables['Met_V'][5:8,0,:,:],axis = 0)
v_model_autumn = np.average(data_model.variables['Met_V'][8:11,0,:,:],axis = 0)


## SETTING THE COLORS FOR THE COLORBAR
class FixPointNormalize(matplotlib.colors.Normalize):
    """
    Inspired by https://stackoverflow.com/questions/20144529/shifted-colorbar-matplotlib
    Subclassing Normalize to obtain a colormap with a fixpoint
    somewhere in the middle of the colormap.

    This may be useful for a `terrain` map, to set the "sea level"
    to a color in the blue/turquise range.
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
# inspired by https://stackoverflow.com/questions/31051488/combining-two-matplotlib-colormaps
colors_undersea = plt.cm.terrain(np.linspace(0, 0.17, 56))
colors_land = plt.cm.terrain(np.linspace(0.25, 1, 200))
# combine them and build a new colormap
colors = np.vstack((colors_undersea, colors_land))
cut_terrain_map = matplotlib.colors.LinearSegmentedColormap.from_list('cut_terrain', colors)
norm = FixPointNormalize(sealevel=0, vmax=5500)


colors = (['#0063FF','#009696','#00C633','#63FF00','#96FF00','#C6FF33','#FFFF00','#FFC600','#FFA000','#FF7C00','#FF1900'])

cmap_name = 'my_list'
#fig, ax = plt.subplots(figsize=(9, 6))
cm = LinearSegmentedColormap.from_list(cmap_name, colors, N=100)

#######################################################################################################################################################
## PLOTTING U WIND ERA
map = Basemap(resolution='c',projection='cyl', llcrnrlat=0.,urcrnrlat=40.,llcrnrlon=60.,urcrnrlon=100., suppress_ticks=False,ax=ax[0,0])
lon, lat = np.meshgrid(lons_era, lats_era)
xi, yi = map(lon, lat)
map.readshapefile(r'D:\GEOS-Chem Validation\India_world_full_kashmir\India_world_full_kashmir', 'India_world_full_kashmir') # drawbounds = False)
im_u = map.contourf(xi,yi,u_era_winter, interpolation='nearest',levels = np.linspace(-5,10,16), extend='both', origin='lower', cmap=cm)

map = Basemap(resolution='c',projection='cyl', llcrnrlat=0.,urcrnrlat=40.,llcrnrlon=60.,urcrnrlon=100., suppress_ticks=False,ax=ax[0,1])
lon, lat = np.meshgrid(lons_era, lats_era)
xi, yi = map(lon, lat)
map.readshapefile(r'D:\GEOS-Chem Validation\India_world_full_kashmir\India_world_full_kashmir', 'India_world_full_kashmir') # drawbounds = False)
im_u = map.contourf(xi,yi,u_era_spring, interpolation='nearest',levels = np.linspace(-5,10,16), extend='both', origin='lower', cmap=cm)

map = Basemap(resolution='c',projection='cyl', llcrnrlat=0.,urcrnrlat=40.,llcrnrlon=60.,urcrnrlon=100., suppress_ticks=False,ax=ax[0,2])
lon, lat = np.meshgrid(lons_era, lats_era)
xi, yi = map(lon, lat)
map.readshapefile(r'D:\GEOS-Chem Validation\India_world_full_kashmir\India_world_full_kashmir', 'India_world_full_kashmir') # drawbounds = False)
im_u = map.contourf(xi,yi,u_era_summer, interpolation='nearest',levels = np.linspace(-5,10,16), extend='both', origin='lower', cmap=cm)

map = Basemap(resolution='c',projection='cyl', llcrnrlat=0.,urcrnrlat=40.,llcrnrlon=60.,urcrnrlon=100., suppress_ticks=False,ax=ax[0,3])
lon, lat = np.meshgrid(lons_era, lats_era)
xi, yi = map(lon, lat)
map.readshapefile(r'D:\GEOS-Chem Validation\India_world_full_kashmir\India_world_full_kashmir', 'India_world_full_kashmir') # drawbounds = False)
im_u = map.contourf(xi,yi,u_era_autumn, interpolation='nearest',levels = np.linspace(-5,10,16), extend='both', origin='lower', cmap=cm)

## PLOTTING U WIND MODEL
map = Basemap(resolution='c',projection='cyl', llcrnrlat=0.,urcrnrlat=40.,llcrnrlon=60.,urcrnrlon=100., suppress_ticks=False,ax=ax[1,0])
lon, lat = np.meshgrid(lons_model, lats_model)
xi, yi = map(lon, lat)
map.readshapefile(r'D:\GEOS-Chem Validation\India_world_full_kashmir\India_world_full_kashmir', 'India_world_full_kashmir') # drawbounds = False)
im_u = map.contourf(xi,yi,u_model_winter, interpolation='nearest',levels = np.linspace(-5,10,16), extend='both', origin='lower', cmap=cm)

map = Basemap(resolution='c',projection='cyl', llcrnrlat=0.,urcrnrlat=40.,llcrnrlon=60.,urcrnrlon=100., suppress_ticks=False,ax=ax[1,1])
lon, lat = np.meshgrid(lons_model, lats_model)
xi, yi = map(lon, lat)
map.readshapefile(r'D:\GEOS-Chem Validation\India_world_full_kashmir\India_world_full_kashmir', 'India_world_full_kashmir') # drawbounds = False)
im_u = map.contourf(xi,yi,u_model_spring, interpolation='nearest',levels = np.linspace(-5,10,16), extend='both', origin='lower', cmap=cm)

map = Basemap(resolution='c',projection='cyl', llcrnrlat=0.,urcrnrlat=40.,llcrnrlon=60.,urcrnrlon=100., suppress_ticks=False,ax=ax[1,2])
lon, lat = np.meshgrid(lons_model, lats_model)
xi, yi = map(lon, lat)
map.readshapefile(r'D:\GEOS-Chem Validation\India_world_full_kashmir\India_world_full_kashmir', 'India_world_full_kashmir') # drawbounds = False)
im_u = map.contourf(xi,yi,u_model_summer, interpolation='nearest',levels = np.linspace(-5,10,16), extend='both', origin='lower', cmap=cm)

map = Basemap(resolution='c',projection='cyl', llcrnrlat=0.,urcrnrlat=40.,llcrnrlon=60.,urcrnrlon=100., suppress_ticks=False,ax=ax[1,3])
lon, lat = np.meshgrid(lons_model, lats_model)
xi, yi = map(lon, lat)
map.readshapefile(r'D:\GEOS-Chem Validation\India_world_full_kashmir\India_world_full_kashmir', 'India_world_full_kashmir') # drawbounds = False)
im_u = map.contourf(xi,yi,u_model_autumn, interpolation='nearest',levels = np.linspace(-5,10,16), extend='both', origin='lower', cmap=cm)

## PLOTTING V WIND ERA
map = Basemap(resolution='c',projection='cyl', llcrnrlat=0.,urcrnrlat=40.,llcrnrlon=60.,urcrnrlon=100., suppress_ticks=False,ax=ax[2,0])
lon, lat = np.meshgrid(lons_era, lats_era)
xi, yi = map(lon, lat)
map.readshapefile(r'D:\GEOS-Chem Validation\India_world_full_kashmir\India_world_full_kashmir', 'India_world_full_kashmir') # drawbounds = False)
im_v = map.contourf(xi,yi,v_era_winter, interpolation='nearest',levels = np.linspace(-7,8,16), extend='both', origin='lower', cmap=cm)

map = Basemap(resolution='c',projection='cyl', llcrnrlat=0.,urcrnrlat=40.,llcrnrlon=60.,urcrnrlon=100., suppress_ticks=False,ax=ax[2,1])
lon, lat = np.meshgrid(lons_era, lats_era)
xi, yi = map(lon, lat)
map.readshapefile(r'D:\GEOS-Chem Validation\India_world_full_kashmir\India_world_full_kashmir', 'India_world_full_kashmir') # drawbounds = False)
im_v = map.contourf(xi,yi,v_era_spring, interpolation='nearest',levels = np.linspace(-7,8,16), extend='both', origin='lower', cmap=cm)

map = Basemap(resolution='c',projection='cyl', llcrnrlat=0.,urcrnrlat=40.,llcrnrlon=60.,urcrnrlon=100., suppress_ticks=False,ax=ax[2,2])
lon, lat = np.meshgrid(lons_era, lats_era)
xi, yi = map(lon, lat)
map.readshapefile(r'D:\GEOS-Chem Validation\India_world_full_kashmir\India_world_full_kashmir', 'India_world_full_kashmir') # drawbounds = False)
im_v = map.contourf(xi,yi,v_era_summer, interpolation='nearest',levels = np.linspace(-7,8,16), extend='both', origin='lower', cmap=cm)

map = Basemap(resolution='c',projection='cyl', llcrnrlat=0.,urcrnrlat=40.,llcrnrlon=60.,urcrnrlon=100., suppress_ticks=False,ax=ax[2,3])
lon, lat = np.meshgrid(lons_era, lats_era)
xi, yi = map(lon, lat)
map.readshapefile(r'D:\GEOS-Chem Validation\India_world_full_kashmir\India_world_full_kashmir', 'India_world_full_kashmir') # drawbounds = False)
im_v = map.contourf(xi,yi,v_era_autumn, interpolation='nearest',levels = np.linspace(-7,8,16), extend='both', origin='lower', cmap=cm)

## PLOTTING V WIND MODEL
map = Basemap(resolution='c',projection='cyl', llcrnrlat=0.,urcrnrlat=40.,llcrnrlon=60.,urcrnrlon=100., suppress_ticks=False,ax=ax[3,0])
lon, lat = np.meshgrid(lons_model, lats_model)
xi, yi = map(lon, lat)
map.readshapefile(r'D:\GEOS-Chem Validation\India_world_full_kashmir\India_world_full_kashmir', 'India_world_full_kashmir') # drawbounds = False)
im_v = map.contourf(xi,yi,v_model_winter, interpolation='nearest',levels = np.linspace(-7,8,16), extend='both', origin='lower', cmap=cm)

map = Basemap(resolution='c',projection='cyl', llcrnrlat=0.,urcrnrlat=40.,llcrnrlon=60.,urcrnrlon=100., suppress_ticks=False,ax=ax[3,1])
lon, lat = np.meshgrid(lons_model, lats_model)
xi, yi = map(lon, lat)
map.readshapefile(r'D:\GEOS-Chem Validation\India_world_full_kashmir\India_world_full_kashmir', 'India_world_full_kashmir') # drawbounds = False)
im_v = map.contourf(xi,yi,v_model_spring, interpolation='nearest',levels = np.linspace(-7,8,16), extend='both', origin='lower', cmap=cm)

map = Basemap(resolution='c',projection='cyl', llcrnrlat=0.,urcrnrlat=40.,llcrnrlon=60.,urcrnrlon=100., suppress_ticks=False,ax=ax[3,2])
lon, lat = np.meshgrid(lons_model, lats_model)
xi, yi = map(lon, lat)
map.readshapefile(r'D:\GEOS-Chem Validation\India_world_full_kashmir\India_world_full_kashmir', 'India_world_full_kashmir') # drawbounds = False)
im_v = map.contourf(xi,yi,v_model_summer, interpolation='nearest',levels = np.linspace(-7,8,16), extend='both', origin='lower', cmap=cm)

map = Basemap(resolution='c',projection='cyl', llcrnrlat=0.,urcrnrlat=40.,llcrnrlon=60.,urcrnrlon=100., suppress_ticks=False,ax=ax[3,3])
lon, lat = np.meshgrid(lons_model, lats_model)
xi, yi = map(lon, lat)
map.readshapefile(r'D:\GEOS-Chem Validation\India_world_full_kashmir\India_world_full_kashmir', 'India_world_full_kashmir') # drawbounds = False)
im_v = map.contourf(xi,yi,v_model_autumn, interpolation='nearest',levels = np.linspace(-7,8,16), extend='both', origin='lower', cmap=cm)

## LATITUDE AND LONGITUDE FORMATTING
for i in range(4):
    for j in range(4):
        # ax[i][j].xaxis.set_ticks_position('bottom')
        ax[i][j].tick_params(which='major', width=1.00, length=7, labelsize=8, top=True)
        ax[i][j].tick_params(which='minor', width=0.75, length=4.5, labelsize=8, top=True)

        # ax[i][j].yaxis.set_ticks_position('left')
        ax[i][j].tick_params(which='major', width=1.00, length=7, labelsize=8, right=True)
        ax[i][j].tick_params(which='minor', width=0.75, length=4.5, labelsize=8, right=True)

        ax[i][j].xaxis.set_major_locator(ticker.AutoLocator())
        ax[i][j].xaxis.set_minor_locator(ticker.AutoMinorLocator())

        ax[i][j].yaxis.set_major_locator(ticker.AutoLocator())
        ax[i][j].yaxis.set_minor_locator(ticker.AutoMinorLocator())

        ax[i][j].xaxis.set_major_formatter(LONGITUDE_FORMATTER)
        ax[i][j].yaxis.set_major_formatter(LATITUDE_FORMATTER)

        # ax[i][j].set_xlabel('Longitude', fontsize=9, fontweight='bold')
        # ax[i][j].set_ylabel('Latitude', fontsize=9, fontweight='bold')


plt.tight_layout()
cbar1 = plt.colorbar(im_u, cmap=cm,ax = ax[0:2,:])
cbar2 = plt.colorbar(im_v, cmap=cm,ax = ax[2:4,:])

plt.savefig('era_wind_comparison.pdf')