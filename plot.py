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



def circleaverage(rhodot,time,theta):
    theta=np.mod(theta-theta[0],-2*np.pi)
    #%theta=mod(theta1,-2*pi);
    temprho = []
    temptime = []
    meanrho = []
    meantime = []
    temprho.append(rhodot[0])
    temptime.append(time[0])
    #tempindex = 1
    #meanindex = 0
    previoustheta = theta[0]
    for i in range(1,theta.size):
        if previoustheta < theta[i]:
            meanrho.append(np.mean(temprho))
            meantime.append(np.mean(temptime))
            del temprho, temptime
            temprho =[]
            temptime = []
            #tempindex=0
            #meanindex=meanindex+1
            temprho.append(rhodot[i])
            temptime.append(time[i])
        else:
            temprho.append(rhodot[i])
            temptime.append(time[i])
            #tempindex = tempindex+1

        previoustheta = theta[i]

    meanrho.append(np.mean(temprho))
    meantime.append(np.mean(temptime))
    del temprho, temptime
    meanrho=meanrho[0:-1]
    meantime=meantime[0:-1]
    return np.array(meanrho), np.array(meantime)

plt.close('all')

with hp.File('850mb_300m_10min_NAM_Rhodot_Origin_t=0-215hrs_Sept2017.hdf5','r') as data:
    rhodot = data['rhodot'][:]
    t = data['t'][:]
    data.close()
    
with hp.File('simflightdata_2000.mat','r') as data:
    prhodot = data['rhodot'][:]
    pt = data['timeout'][:]
    data.close()

with hp.File('hunterdata.mat','r') as data:
    hrhodot = data['rhodot'][:].squeeze()
    ht = data['timeout'][:].squeeze()
    htheta = data['thetaout'][:].squeeze()


hrhodot, ht = circleaverage(hrhodot,ht,htheta)
#prhodot, pt = circleaverage(prhodot,pt,ptheta)    
    
    

#rhodot=f['rhodot']

#with hp.File('simflight2000_10xhr_halfsecondres.mat','r') as data:
#    prhodot = data['rhodot'][:]
#    po = data['timeout'][:]
#    data.close()
#plt.plot(t,rhodot_origin,color='b')
#plt.plot(t,rhodot[:,125,130],color='r')
'''
tmin = np.max([t.min(),pt.min(),ht.min()])
tmax = np.min([t.max(),pt.max(),ht.max()])

tw = ht[ht>=tmin]
tw = tw[tw<=tmax]
#tw = tw[tw<=tmax]

tcku = sint.splrep(t, rhodot, s=0)
rhodot= sint.splev(tw, tcku, der=0)
tcku = sint.splrep(pt, prhodot, s=0)
prhodot = sint.splev(tw, tcku, der=0)
tcku = sint.splrep(ht, hrhodot, s=0)
hrhodot = sint.splev(tw, tcku, der=0)

del tcku, ht, pt, t, tmin, tmax
                     
import pandas as pd
A = pd.DataFrame(np.transpose([rhodot,prhodot,hrhodot]),columns=['rhodot','peters flight','hunters simulation'])
print A.corr()
#fig = plt.figure(1)
#scatter_matrix(A)


fig = plt.figure(2)
ax4=plt.plot(tw,prhodot*3600,color='y',label="Peter's virtual flight")
ax1=plt.plot(tw,hrhodot*3600,color='b',label="Hunter's flight simulation")
ax3=plt.plot(tw,rhodot*3600,color='r',label="Rhodot")

'''

fig = plt.figure(2)
ax4=plt.plot(pt,prhodot*3600,color='y',label="Peter's virtual flight")
ax1=plt.plot(ht,hrhodot*3600,color='b',label="Hunter's flight simulation")
ax3=plt.plot(t,rhodot*3600,color='r',label="Rhodot")
plt.ylabel('hrs^{-1}')
#ax4=plt.plot(pt,prhodot,color='b',label="Peter's virtual flight 10x/hr")
#ax3=plt.plot(t,rhodot,color='r',label="Rhodot")

plt.axhline(0,color='k')
plt.legend()#handles=[ax1,ax2,ax3])
#plt.tight_layout()
#plt.axis('tight')
plt.autoscale(enable=True, axis='x', tight=True)
plt.ylim([-2,2])
'''
rwant -= rwant.min()
rwant = rwant/rwant.max()
hrhodot -= hrhodot.min()
hrhodot = hrhodot/hrhodot.max()
fig = plt.figure(3)
ax1 = plt.scatter(rwant,hrhodot)
'''