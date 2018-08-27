# -*- coding: utf-8 -*-
"""
Created on Sun Jul 22 17:59:06 2018

@author: pnola
"""

#Domain [-300 300]^2 km
#Origin (41.3209371228N, 289.46309961W)
#Projection Lambert
import matplotlib
#matplotlib.use('Agg')
matplotlib.rcParams['text.usetex']=True
matplotlib.rcParams['mathtext.fontset'] = 'cm'

import matplotlib.pyplot as plt
plt.close('all')
plt.rc('font', **{'family': 'serif', 'serif': ['cmr10']})

from mpl_toolkits.basemap import Basemap
from mpl_toolkits.axes_grid1 import make_axes_locatable
import numpy as np

#import matplotlib.animation as animation
import h5py as hp
import scipy.ndimage.filters as filter
from cyanorangecmap import co
initday = 4
inittime = 00-4 #Run Initialization(UTC) - 4hrs for EDT Conversion
#inittime = 12-4+4 #Run Initialization(UTC) - 4hrs for EDT Conversion + 4hrs for integration time
stepsize = 10 #minutes
'''
amaxits = 7
rmaxits = 7
astd = 2
rstd = 2
atol =-0.002
rtol =-0.002
'''
amaxits = 9
rmaxits = 9
astd = 1
rstd = 1
atol =-0.005
rtol =-0.005
parallelres = 7
meridianres = 7
dx = 300
dy = 300
star = [37.19838, -80.57834]
plt.register_cmap(name='CO', data=co())

#xlen = 201
#ylen = 197
xlen = 259
ylen = 257
tlen = 1267
dim = [tlen,ylen,xlen]
fig, ax = plt.subplots(figsize=(16, 12), dpi=100, facecolor='w', edgecolor='k')
origin = [37.208, -80.5803]

#plt.rcParams.update({'axes.titlesize': 'large'})

print("Begin Map")
''
m = Basemap(width=77700,height=76800,\
    rsphere=(6378137.00,6356752.3142),\
    resolution='c',area_thresh=0.,projection='lcc',\
    lat_1=35.,lat_0=origin[0],lon_0=origin[1])#,ax=ax)

#m.drawcoastlines(linewidth=2.0)
#m.drawcountries()
m.drawparallels(np.linspace(m.llcrnrlat,m.urcrnrlat,parallelres),labels=[True,False,False,False])
m.drawmeridians(np.linspace(m.llcrnrlon,m.urcrnrlon,meridianres),labels=[False,False,False,True])
m.drawstates()
m.drawrivers()
with hp.File('850mb_300m_10min_NAM_LCS_t=4-215hrs_Sept2017_int=-1.hdf5','r') as loadfile:
    ftle=loadfile['ftle'][:]
    dirdiv=loadfile['directionalderivative'][:]
    concav=loadfile['concavity'][:]
    loadfile.close()

#thresh=np.percentile(ftle,95,axis=None)
#thresh = 0.3000400059223705/3600 #90th percentile for the -4hr FTLE timeseries, see plot_time_series.py
#thresh = 0
thresh = 0.5571152601155729/3600 #90th percentile for the -1hr FTLE timeseries, see plot_time_series.py
#thresh = 0.1742656513095688/3600 #50th percentile for the -1hr FTLE timeseries, see plot_time_series.py
dirdiv = np.ma.masked_where(concav>0,dirdiv)
dirdiv = np.ma.masked_where(ftle<=thresh,dirdiv)

x = np.linspace(0, m.urcrnrx, dim[2])
y = np.linspace(0, m.urcrnry, dim[1])
xx, yy = np.meshgrid(x, y)
x, y = m(star[1],star[0]) 
passing_times = []
for t in range(dirdiv.shape[0]):
    ridge = m.contour(xx,yy,dirdiv[t,:,:],levels =[0])#,latlon=True)
    pp = ridge.collections[0].get_paths()
    for p in range(len(pp)):
        v=pp[p].vertices
        if any((v[:,0]-x)**2+(v[:,1]-y)**2<1000**2):
            print('hit @ {0}'.format(t))
            passing_times.append(t)
            break
