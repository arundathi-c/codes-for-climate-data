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

fig, ax = plt.subplots(3,4,figsize=(12, 7),sharex=True,sharey=True)
ax[0,0].set_xlabel("Winter(DJF)",size = 10)
ax[0,1].set_xlabel("Spring(MAM)",size = 10)
ax[0,2].set_xlabel("Summer(JJA)",size = 10)
ax[0,3].set_xlabel("Autumn(SON)",size = 10)
ax[0,0].set_ylabel("TRMM(mm/day)",size=10)
ax[1,0].set_ylabel("GEOS-Chem(mm/day)",size=8)
ax[2,0].set_ylabel("Difference(GEOS-Chem - TRMM)",size=8)
ax[0,0].xaxis.set_label_coords(0.5,1.1)
ax[0,1].xaxis.set_label_coords(0.5,1.1)
ax[0,2].xaxis.set_label_coords(0.5,1.1)
ax[0,3].xaxis.set_label_coords(0.5,1.1)

## LOADING TRMM DATA
data_trmm_winter = Dataset(r"D:\GEOS-Chem Validation\TRMM\regrid.winter.trmm.nc4",mode = 'r')
trmm_winter = np.transpose(data_trmm_winter.variables['precipitation'][:,:])

data_trmm_spring = Dataset(r"D:\GEOS-Chem Validation\TRMM\regrid.spring.trmm.nc4",mode = 'r')
trmm_spring = np.transpose(data_trmm_spring.variables['precipitation'][:,:])

data_trmm_summer = Dataset(r"D:\GEOS-Chem Validation\TRMM\regrid.summer.trmm.nc4",mode = 'r')
trmm_summer = np.transpose(data_trmm_summer.variables['precipitation'][:,:])

data_trmm_autumn = Dataset(r"D:\GEOS-Chem Validation\TRMM\regrid.autumn.trmm.nc4",mode = 'r')
trmm_autumn = np.transpose(data_trmm_autumn.variables['precipitation'][:,:])

## LOADING MODEL DATA
data_model = Dataset(r"D:\GEOS-Chem Validation\met_send\GEOSChem.StateMet.2010.nc4")
lats = data_model.variables['lat']
lons = data_model.variables['lon']
time_model = data_model.variables['time']

model_winter = np.average(data_model.variables['Met_PRECTOT'][[0,1,11],:,:],axis = 0)
model_spring = np.average(data_model.variables['Met_PRECTOT'][2:5,:,:],axis = 0)
model_summer = np.average(data_model.variables['Met_PRECTOT'][5:8,:,:],axis = 0)
model_autumn = np.average(data_model.variables['Met_PRECTOT'][8:11,:,:],axis = 0)


diff_winter = model_winter-trmm_winter
diff_spring = model_spring-trmm_spring
diff_summer = model_summer-trmm_summer
diff_autumn = model_autumn-trmm_autumn

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

## PLOTTING TRMM DATA
map = Basemap(resolution='c',projection='cyl', llcrnrlat=0.,urcrnrlat=40.,llcrnrlon=60.,urcrnrlon=100., suppress_ticks=False,ax=ax[0,0])
lon, lat = np.meshgrid(lons, lats)
xi, yi = map(lon, lat)
map.readshapefile(r'D:\GEOS-Chem Validation\India_world_full_kashmir\India_world_full_kashmir', 'India_world_full_kashmir') # drawbounds = False)
im_prec = map.contourf(xi,yi,trmm_winter,levels = np.linspace(0,55,12), extend='both', origin='lower', cmap=cm)

map = Basemap(resolution='c',projection='cyl', llcrnrlat=0.,urcrnrlat=40.,llcrnrlon=60.,urcrnrlon=100., suppress_ticks=False,ax=ax[0,1])
lon, lat = np.meshgrid(lons, lats)
xi, yi = map(lon, lat)
map.readshapefile(r'D:\GEOS-Chem Validation\India_world_full_kashmir\India_world_full_kashmir', 'India_world_full_kashmir') # drawbounds = False)
im_prec = map.contourf(xi,yi,trmm_spring,levels = np.linspace(0,55,12), extend='both', origin='lower', cmap=cm)

map = Basemap(resolution='c',projection='cyl', llcrnrlat=0.,urcrnrlat=40.,llcrnrlon=60.,urcrnrlon=100., suppress_ticks=False,ax=ax[0,2])
lon, lat = np.meshgrid(lons, lats)
xi, yi = map(lon, lat)
map.readshapefile(r'D:\GEOS-Chem Validation\India_world_full_kashmir\India_world_full_kashmir', 'India_world_full_kashmir') # drawbounds = False)
im_prec = map.contourf(xi,yi,trmm_summer,levels = np.linspace(0,55,12), extend='both', origin='lower', cmap=cm)

map = Basemap(resolution='c',projection='cyl', llcrnrlat=0.,urcrnrlat=40.,llcrnrlon=60.,urcrnrlon=100., suppress_ticks=False,ax=ax[0,3])
lon, lat = np.meshgrid(lons, lats)
xi, yi = map(lon, lat)
map.readshapefile(r'D:\GEOS-Chem Validation\India_world_full_kashmir\India_world_full_kashmir', 'India_world_full_kashmir') # drawbounds = False)
im_prec = map.contourf(xi,yi,trmm_autumn,levels = np.linspace(0,55,12), extend='both', origin='lower', cmap=cm)

## PLOTTING MODEL DATA
map = Basemap(resolution='c',projection='cyl', llcrnrlat=0.,urcrnrlat=40.,llcrnrlon=60.,urcrnrlon=100., suppress_ticks=False,ax=ax[1,0])
lon, lat = np.meshgrid(lons, lats)
xi, yi = map(lon, lat)
map.readshapefile(r'D:\GEOS-Chem Validation\India_world_full_kashmir\India_world_full_kashmir', 'India_world_full_kashmir') # drawbounds = False)
im_prec = map.contourf(xi,yi,model_winter,levels = np.linspace(0,55,12), extend='both', origin='lower', cmap=cm)

map = Basemap(resolution='c',projection='cyl', llcrnrlat=0.,urcrnrlat=40.,llcrnrlon=60.,urcrnrlon=100., suppress_ticks=False,ax=ax[1,1])
lon, lat = np.meshgrid(lons, lats)
xi, yi = map(lon, lat)
map.readshapefile(r'D:\GEOS-Chem Validation\India_world_full_kashmir\India_world_full_kashmir', 'India_world_full_kashmir') # drawbounds = False)
im_prec = map.contourf(xi,yi,model_spring,levels = np.linspace(0,55,12), extend='both', origin='lower', cmap=cm)

map = Basemap(resolution='c',projection='cyl', llcrnrlat=0.,urcrnrlat=40.,llcrnrlon=60.,urcrnrlon=100., suppress_ticks=False,ax=ax[1,2])
lon, lat = np.meshgrid(lons, lats)
xi, yi = map(lon, lat)
map.readshapefile(r'D:\GEOS-Chem Validation\India_world_full_kashmir\India_world_full_kashmir', 'India_world_full_kashmir') # drawbounds = False)
im_prec = map.contourf(xi,yi,model_summer,levels = np.linspace(0,55,12), extend='both', origin='lower', cmap=cm)

map = Basemap(resolution='c',projection='cyl', llcrnrlat=0.,urcrnrlat=40.,llcrnrlon=60.,urcrnrlon=100., suppress_ticks=False,ax=ax[1,3])
lon, lat = np.meshgrid(lons, lats)
xi, yi = map(lon, lat)
map.readshapefile(r'D:\GEOS-Chem Validation\India_world_full_kashmir\India_world_full_kashmir', 'India_world_full_kashmir') # drawbounds = False)
im_prec = map.contourf(xi,yi,model_autumn,levels = np.linspace(0,55,12), extend='both', origin='lower', cmap=cm)

## PLOTTING DIFFERENCE
map = Basemap(resolution='c',projection='cyl', llcrnrlat=0.,urcrnrlat=40.,llcrnrlon=60.,urcrnrlon=100., suppress_ticks=False,ax=ax[2,0])
lon, lat = np.meshgrid(lons, lats)
xi, yi = map(lon, lat)
map.readshapefile(r'D:\GEOS-Chem Validation\India_world_full_kashmir\India_world_full_kashmir', 'India_world_full_kashmir') # drawbounds = False)
im_diff = map.contourf(xi,yi,diff_winter,levels = np.linspace(-15,40,12), extend='both', origin='lower', cmap=cm)

map = Basemap(resolution='c',projection='cyl', llcrnrlat=0.,urcrnrlat=40.,llcrnrlon=60.,urcrnrlon=100., suppress_ticks=False,ax=ax[2,1])
lon, lat = np.meshgrid(lons, lats)
xi, yi = map(lon, lat)
map.readshapefile(r'D:\GEOS-Chem Validation\India_world_full_kashmir\India_world_full_kashmir', 'India_world_full_kashmir') # drawbounds = False)
im_diff = map.contourf(xi,yi,diff_spring,levels = np.linspace(-15,40,12), extend='both', origin='lower', cmap=cm)

map = Basemap(resolution='c',projection='cyl', llcrnrlat=0.,urcrnrlat=40.,llcrnrlon=60.,urcrnrlon=100., suppress_ticks=False,ax=ax[2,2])
lon, lat = np.meshgrid(lons, lats)
xi, yi = map(lon, lat)
map.readshapefile(r'D:\GEOS-Chem Validation\India_world_full_kashmir\India_world_full_kashmir', 'India_world_full_kashmir') # drawbounds = False)
im_diff = map.contourf(xi,yi,diff_summer,levels = np.linspace(-15,40,12), extend='both', origin='lower', cmap=cm)

map = Basemap(resolution='c',projection='cyl', llcrnrlat=0.,urcrnrlat=40.,llcrnrlon=60.,urcrnrlon=100., suppress_ticks=False,ax=ax[2,3])
lon, lat = np.meshgrid(lons, lats)
xi, yi = map(lon, lat)
map.readshapefile(r'D:\GEOS-Chem Validation\India_world_full_kashmir\India_world_full_kashmir', 'India_world_full_kashmir') # drawbounds = False)
im_diff = map.contourf(xi,yi,diff_autumn,levels = np.linspace(-15,40,12), extend='both', origin='lower', cmap=cm)

## LATITUDE AND LONGITUDE FORMATTING
for i in range(3):
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
        m = Basemap(resolution='c', projection='cyl', llcrnrlat=0., urcrnrlat=40., llcrnrlon=60., urcrnrlon=100.,
                    suppress_ticks=False)
        m.readshapefile(r'D:\GEOS-Chem Validation\India_world_full_kashmir\India_world_full_kashmir',
                        'India_world_full_kashmir')  # drawbounds = False)
        m.drawcoastlines(linewidth=0.5)
        m.drawparallels(np.arange(-90., 120., 30.), labels=[1, 0, 0, 0])
        m.drawmeridians(np.arange(-180., 181., 45.), labels=[0, 0, 0, 1])
        m.pcolormesh(longitude, latitude, data, latlon=True, alpha=0.90)
        cb = m.colorbar()
        cb.set_label('Unit:K')
        basename = os.path.basename(FILE_NAME)
        plt.title('{0}\n {1} at TempPrsLvls=0'.format(basename, DATAFIELD_NAME))
        fig = plt.gcf()
        # plt.show()
        pngfile = "{0}.py.png".format(basename)
        fig.savefig(pngfile)

plt.tight_layout()
cbar1 = plt.colorbar(im_prec, cmap=cm,ax = ax[0:2,:],aspect = 25)
cbar2 = plt.colorbar(im_diff, cmap=cm,ax = ax[2,:],aspect  =12)

plt.savefig('precip_comparison_final_2.pdf')
