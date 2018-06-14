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


with hp.File('850mb_300m_10min_NAM_Rhodot_Origin_t=0-215hrs_Sept2017.hdf5','r') as data:
    rhodot = data['rhodot'][:]
    t = data['t'][:]
    data.close()
    
#with hp.File('simflight2000_20xhr_halfsecondres.mat','r') as data:
#with hp.File('simflightdata_1000.mat','r') as data:
#    htrhodot = data['rhodot'][:]
#    to = data['timeout'][:]
#    data.close()

with hp.File('hunterdata.mat','r') as data:
    rhodotx = data['rhodot'][:].squeeze()
    tx = data['timeout'][:].squeeze()

#rhodot=f['rhodot']
'''
'''
#with hp.File('simflight2000_10xhr_halfsecondres.mat','r') as data:
#    prhodot = data['rhodot'][:]
#    po = data['timeout'][:]
#    data.close()
#plt.plot(t,rhodot_origin,color='b')
#plt.plot(t,rhodot[:,125,130],color='r')

tcku = sint.splrep(t, rhodot, s=0)
rwant= sint.splev(tx, tcku, der=0)
#
import pandas as pd
from pandas.plotting import scatter_matrix
A = pd.DataFrame(np.transpose([rwant,rhodotx]),columns=['rho','hun'])
print A.corr()
fig = plt.figure(1)
scatter_matrix(A)

'''
cor = np.correlate(rwant.squeeze(),rhodotx.squeeze(),mode='full')
print cor

del tx, t, rhodot, tcku
from scipy.stats import linregress
show = linregress(rwant, rhodotx)
print show
'''

plt.close('all')

fig = plt.figure(2)
#ax2=plt.plot((po-46)*3600,prhodot,color='k',label="Peter's virtual flight 5x/hr")
#ax4=plt.plot((to-46)*7200,htrhodot,color='b',label="Peter's virtual flight 10x/hr")
ax1=plt.plot(tx,rhodotx*3600,color='y',label="Hunter's flight simulation")
ax4=plt.plot(tx,rwant*3600,color='b',label="Rhodot")

#ax3=plt.plot(t,rhodot,color='r',label="Rhodot")

plt.axhline(0)
plt.legend()#handles=[ax1,ax2,ax3])

rwant -= rwant.min()
rwant = rwant/rwant.max()
rhodotx -= rhodotx.min()
rhodotx = rhodotx/rhodotx.max()
fig = plt.figure(3)
ax1 = plt.scatter(rwant,rhodotx)
