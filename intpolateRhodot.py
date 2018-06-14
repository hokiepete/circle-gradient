# -*- coding: utf-8 -*-
"""
Created on Wed May 09 12:51:17 2018

@author: pnola
"""

import h5py as hp
import numpy as np
import scipy.interpolate as sint
#import scipy.io as sio
import matplotlib.pyplot as plt

with hp.File('850mb_300m_10min_NAM_Rhodot_t=0-215hrs_Sept2017.hdf5','r') as data:
    rhodot = data['rhodot'][:]
del data
with hp.File('850mb_NAM_gridpoints.hdf5','r') as data:
    x = data['x'][:]
    y = data['y'][:]
    t = data['t'][:]
del data
print x.shape
x0=169.4099
y0=-1043.9
points = zip(x.ravel(),y.ravel())
rhodot_origin = np.empty(t.shape)
for tt in range(len(t)):
    print tt
    rhodot_origin[tt] = sint.griddata(points,rhodot[tt,:,:].ravel(),(x0,y0),method='cubic')
    #f = sint.interp2d(x, y, rhodot[tt,:,:], kind='cubic')
    #rhodot_origin[tt] = f(x0,y0)
    #tck = sint.bisplrep(x, y, rhodot[tt,:,:], s=0)
    #rhodot_origin[tt] = sint.bisplev(x0,y0,tck)
with hp.File('850mb_300m_10min_NAM_Rhodot_Origin_t=0-215hrs_Sept2017.hdf5','w') as savefile:
        savefile.create_dataset('t',shape=t.shape,data=t)
        savefile.create_dataset('rhodot',shape=rhodot_origin.shape,data=rhodot_origin)
        savefile.close()
plt.plot(t,rhodot_origin)
        #np.savez('850mb_NAM_Rhodot_Origin_t=46-62hrs_Sept2017.npz',t,rhodot_origin)