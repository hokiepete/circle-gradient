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
    rhodot = data['rhodot'][:].squeeze()
    t = data['t'][:].squeeze()
    data.close()
    
with hp.File('simflightdata_2000.mat','r+') as data:
    prhodot = data['rhodot'][:].squeeze()
    #data['timeout'][:] = 3600*data['timeout'][:]
    pt = data['timeout'][:].squeeze()
    ptheta = data['thetaout'][:].squeeze()
    data.close()

with hp.File('hunterdata_r=02km.mat','r') as data:
    h2rhodot = data['rhodot'][:].squeeze()
    h2t = data['timeout'][:].squeeze()
    h2theta = data['thetaout'][:].squeeze()
    data.close()

with hp.File('hunterdata_r=05km.mat','r') as data:
    h5rhodot = data['rhodot'][:].squeeze()
    h5t = data['timeout'][:].squeeze()
    h5theta = data['thetaout'][:].squeeze()
    data.close()

with hp.File('hunterdata_r=10km.mat','r') as data:
    h10rhodot = data['rhodot'][:].squeeze()
    h10t = data['timeout'][:].squeeze()
    h10theta = data['thetaout'][:].squeeze()
    data.close()

with hp.File('hunterdata_r=15km.mat','r') as data:
    h15rhodot = data['rhodot'][:].squeeze()
    h15t = data['timeout'][:].squeeze()
    h15theta = data['thetaout'][:].squeeze()
    data.close()

h2rhodot, h2t = circleaverage(h2rhodot,h2t,h2theta)
h5rhodot, h5t = circleaverage(h5rhodot,h5t,h5theta)
h10rhodot, h10t = circleaverage(h10rhodot,h10t,h10theta)
h15rhodot, h15t = circleaverage(h15rhodot,h15t,h15theta)
prhodot, pt = circleaverage(prhodot,pt,ptheta)    
    
    

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
'''
fig = plt.figure()
#ax4=plt.plot(tw,abs(prhodot)*3600,color='y',label="Peter's virtual flight")
ax2=plt.plot(h2t,h2rhodot*3600,color='b',label="Hunter's flight simulation 02km")
ax5=plt.plot(h5t,h5rhodot*3600,color='g',label="Hunter's flight simulation 05km")
ax10=plt.plot(h10t,h10rhodot*3600,color='m',label="Hunter's flight simulation 10km")
ax15=plt.plot(h15t,h15rhodot*3600,color='c',label="Hunter's flight simulation 15km")
ax3=plt.plot(t,rhodot*3600,color='r',label="Rhodot")
'''

fig = plt.figure()
ax4=plt.plot(tw,prhodot*3600,color='y',label="Idealized Flight Path, 2km Radius")
ax1=plt.plot(tw,hrhodot*3600,color='b',label="Simulated Flight Path, 2km Radius")
ax3=plt.plot(tw,rhodot*3600,color='r',label="True Rhodot")
plt.ylabel('hrs^{-1}')
#ax4=plt.plot(pt,prhodot,color='b',label="Peter's virtual flight 10x/hr")
#ax3=plt.plot(t,rhodot,color='r',label="Rhodot")
'''
plt.axhline(0,color='k')
plt.legend()#handles=[ax1,ax2,ax3])
#plt.tight_layout()
#plt.axis('tight')
plt.autoscale(enable=True, axis='x', tight=True)
#plt.ylim([-2,2])
#plt.ylim([-1.3,1.3])
plt.show()

'''
rwant -= rwant.min()
rwant = rwant/rwant.max()
hrhodot -= hrhodot.min()
hrhodot = hrhodot/hrhodot.max()
fig = plt.figure(3)
ax1 = plt.scatter(rwant,hrhodot)
'''