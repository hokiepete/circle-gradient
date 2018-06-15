# -*- coding: utf-8 -*-
"""
Created on Wed Jun 13 15:59:44 2018

@author: pnola
"""
import h5py as hp
import numpy as np
import scipy.interpolate as sint
#import scipy.io as sio
import matplotlib.pyplot as plt
plt.close('all')

with hp.File('850mb_300m_10min_NAM_Rhodot_Origin_t=0-215hrs_Sept2017.hdf5','r') as data:
    rhodot = data['rhodot'][:]
    t = data['t'][:]
    data.close()
    
with hp.File('simflightdata_2000.mat','r') as data:
    ptrhodot = data['rhodot'][:]
    to = data['timeout'][:]
    data.close()

with hp.File('hunterdata.mat','r') as data:
    rhodotx = data['rhodot'][:].squeeze()
    tx = data['timeout'][:].squeeze()

#rhodot=f['rhodot']

#with hp.File('simflight2000_10xhr_halfsecondres.mat','r') as data:
#    prhodot = data['rhodot'][:]
#    po = data['timeout'][:]
#    data.close()
#plt.plot(t,rhodot_origin,color='b')
#plt.plot(t,rhodot[:,125,130],color='r')

tmin = np.max([t.min(),to.min(),tx.min()])
tmax = np.min([t.max(),to.max(),tx.max()])

tw = tx[tx>=tmin]
tw = tw[tw<=tmax]
#tw = tw[tw<=tmax]

tcku = sint.splrep(t, rhodot, s=0)
rhodot= sint.splev(tw, tcku, der=0)
tcku = sint.splrep(to, ptrhodot, s=0)
ptrhodot = sint.splev(tw, tcku, der=0)
tcku = sint.splrep(tx, rhodotx, s=0)
rhodotx = sint.splev(tw, tcku, der=0)

del tcku, tx, to, t, tmin, tmax
                     
import pandas as pd
from pandas.plotting import scatter_matrix
A = pd.DataFrame(np.transpose([rhodot,ptrhodot,rhodotx]),columns=['rhodot','peters flight','hunters simulation'])
print A.corr()
#fig = plt.figure(1)
#scatter_matrix(A)


fig = plt.figure(2)
ax4=plt.plot(tw,abs(ptrhodot)*3600,color='y',label="Peter's virtual flight")
ax1=plt.plot(tw,abs(rhodotx)*3600,color='b',label="Hunter's flight simulation")
ax3=plt.plot(tw,abs(rhodot)*3600,color='r',label="Rhodot")

plt.ylabel('hrs^{-1}')
#ax4=plt.plot(to,ptrhodot,color='b',label="Peter's virtual flight 10x/hr")
#ax3=plt.plot(t,rhodot,color='r',label="Rhodot")

plt.axhline(0,color='k')
plt.legend()#handles=[ax1,ax2,ax3])
#plt.tight_layout()
#plt.axis('tight')
plt.autoscale(enable=True, axis='x', tight=True)
plt.ylim([0,2])
'''
rwant -= rwant.min()
rwant = rwant/rwant.max()
rhodotx -= rhodotx.min()
rhodotx = rhodotx/rhodotx.max()
fig = plt.figure(3)
ax1 = plt.scatter(rwant,rhodotx)
'''