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
    rhodot = 3600*data['rhodot'][:].squeeze()
    t = data['t'][:].squeeze()
    data.close()
    
with hp.File('simflightdata_2000.mat','r') as data:
    p2rhodot = 3600*data['rhodot'][:].squeeze()
    p2t = data['timeout'][:].squeeze()
    p2theta = data['thetaout'][:].squeeze()
    data.close()

with hp.File('simflightdata_5000.mat','r') as data:
    p5rhodot = 3600*data['rhodot'][:].squeeze()
    p5t = data['timeout'][:].squeeze()
    p5theta = data['thetaout'][:].squeeze()
    data.close()

with hp.File('simflightdata_10000.mat','r') as data:
    p10rhodot = 3600*data['rhodot'][:].squeeze()
    p10t = data['timeout'][:].squeeze()
    p10theta = data['thetaout'][:].squeeze()
    data.close()
    
with hp.File('simflightdata_15000.mat','r') as data:
    p15rhodot = 3600*data['rhodot'][:].squeeze()
    p15t = data['timeout'][:].squeeze()
    p15theta = data['thetaout'][:].squeeze()
    data.close()

with hp.File('hunterdata_r=02km.mat','r') as data:
    h2rhodot = 3600*data['rhodot'][:].squeeze()
    h2t = data['timeout'][:].squeeze()
    h2theta = data['thetaout'][:].squeeze()
    data.close()

with hp.File('hunterdata_r=05km.mat','r') as data:
    h5rhodot = 3600*data['rhodot'][:].squeeze()
    h5t = data['timeout'][:].squeeze()
    h5theta = data['thetaout'][:].squeeze()
    data.close()

with hp.File('hunterdata_r=10km.mat','r') as data:
    h10rhodot = 3600*data['rhodot'][:].squeeze()
    h10t = data['timeout'][:].squeeze()
    h10theta = data['thetaout'][:].squeeze()
    data.close()

with hp.File('hunterdata_r=15km.mat','r') as data:
    h15rhodot = 3600*data['rhodot'][:].squeeze()
    h15t = data['timeout'][:].squeeze()
    h15theta = data['thetaout'][:].squeeze()
    data.close()

h2rhodot, h2t = circleaverage(h2rhodot,h2t,h2theta)
h5rhodot, h5t = circleaverage(h5rhodot,h5t,h5theta)
h10rhodot, h10t = circleaverage(h10rhodot,h10t,h10theta)
h15rhodot, h15t = circleaverage(h15rhodot,h15t,h15theta)
p2rhodot, p2t = circleaverage(p2rhodot,p2t,p2theta)    
p5rhodot, p5t = circleaverage(p5rhodot,p5t,p5theta)    
p10rhodot, p10t = circleaverage(p10rhodot,p10t,p10theta)    
p15rhodot, p15t = circleaverage(p15rhodot,p15t,p15theta)    
    
    
tmin = np.max([t.min(),p2t.min(),p5t.min(),p10t.min(),p15t.min(),h2t.min(),h5t.min(),h10t.min(),h15t.min()])
tmax = np.min([t.max(),p2t.max(),p5t.max(),p10t.max(),p15t.max(),h2t.max(),h5t.max(),h10t.max(),h15t.max()])

tw = h2t[h2t>=tmin]
tw = tw[tw<=tmax]

tcku = sint.splrep(t, rhodot, s=0)
rhodot= sint.splev(tw, tcku, der=0)
tcku = sint.splrep(h2t, h2rhodot, s=0)
h2rhodot = sint.splev(tw, tcku, der=0)
tcku = sint.splrep(h5t, h5rhodot, s=0)
h5rhodot = sint.splev(tw, tcku, der=0)
tcku = sint.splrep(h10t, h10rhodot, s=0)
h10rhodot = sint.splev(tw, tcku, der=0)
tcku = sint.splrep(h15t, h15rhodot, s=0)
h15rhodot = sint.splev(tw, tcku, der=0)
tcku = sint.splrep(p2t, p2rhodot, s=0)
p2rhodot = sint.splev(tw, tcku, der=0)
tcku = sint.splrep(p5t, p5rhodot, s=0)
p5rhodot = sint.splev(tw, tcku, der=0)
tcku = sint.splrep(p10t, p10rhodot, s=0)
p10rhodot = sint.splev(tw, tcku, der=0)
tcku = sint.splrep(p15t, p15rhodot, s=0)
p15rhodot = sint.splev(tw, tcku, der=0)


del tcku, h2t, h5t, h10t, h15t, p2t, p5t, p10t, p15t, t, tmin, tmax
                     
import pandas as pd
A = pd.DataFrame(np.transpose([rhodot,p2rhodot,p5rhodot,p10rhodot,p15rhodot,h2rhodot,h5rhodot,h10rhodot,h15rhodot]),columns=['rhodot','2km path','5km path','10km path','15km path','2km simulation','5km simulation','10km simulation','15km simulation'])
A.corr().to_csv('Correlation_and_FLight_stats.csv',mode='w')
B=A.describe()
B.to_csv('Correlation_and_FLight_stats.csv',mode='a')
mean = B['rhodot'][1]
std = B['rhodot'][2]
#fig = plt.figure(1)
#scatter_matrix(A)
'''
fig = plt.figure()
#ax4=plt.plot(tw,abs(prhodot),color='y',label="Peter's virtual flight")
ax2=plt.plot(h2t,h2rhodot,color='b',label="Hunter's flight simulation 02km")
ax5=plt.plot(h5t,h5rhodot,color='g',label="Hunter's flight simulation 05km")
ax10=plt.plot(h10t,h10rhodot,color='m',label="Hunter's flight simulation 10km")
ax15=plt.plot(h15t,h15rhodot,color='c',label="Hunter's flight simulation 15km")
ax3=plt.plot(t,rhodot,color='k',label="Rhodot")
'''

fig = plt.figure(1)
ax=plt.plot(tw,rhodot,color='k',label="True Rhodot")
ax2=plt.plot(tw,h2rhodot,color='b',label="Simulated Flight Path, 2km Radius")
ax5=plt.plot(tw,h5rhodot,color='g',label="Simulated Flight Path, 5km Radius")
ax10=plt.plot(tw,h10rhodot,color='m',label="Simulated Flight Path, 10km Radius")
ax15=plt.plot(tw,h15rhodot,color='c',label="Simulated Flight Path, 15km Radius")
plt.axhline(mean,color='k')
plt.axhline(mean+std,color='k')
plt.axhline(mean-std,color='k')
plt.ylabel('hrs^{-1}')
plt.legend()
plt.autoscale(enable=True, axis='x', tight=True)
plt.ylim([-1.3,1.3])
plt.show()


fig = plt.figure(2)
ax=plt.plot(tw,rhodot,color='k',label="True Rhodot")
ax2=plt.plot(tw,p2rhodot,color='b',label="Idealized Flight Path, 2km Radius")
ax2=plt.plot(tw,p5rhodot,color='g',label="Idealized Flight Path, 5km Radius")
ax2=plt.plot(tw,p10rhodot,color='m',label="Idealized Flight Path, 10km Radius")
ax2=plt.plot(tw,p15rhodot,color='c',label="Idealized Flight Path, 15km Radius")
plt.axhline(mean,color='k')
plt.axhline(mean+std,color='k')
plt.axhline(mean-std,color='k')
plt.ylabel('hrs^{-1}')
plt.legend()
plt.autoscale(enable=True, axis='x', tight=True)
plt.ylim([-1.3,1.3])
plt.show()


fig = plt.figure(3)
s1 = plt.subplot(221)
#ax3=plt.plot(tw,rhodot,color='k',label="True Rhodot")
ax2=plt.plot(tw,p2rhodot,color='b',label="Idealized Flight Path, 2km Radius")
ax1=plt.plot(tw,h2rhodot,color='r',label="Simulated Flight Path, 2km Radius")
plt.ylabel('hrs^{-1}')
plt.legend()
plt.autoscale(enable=True, axis='x', tight=True)
plt.ylim([-1.3,1.3])

s2 = plt.subplot(222)
#ax3=plt.plot(tw,rhodot,color='k',label="True Rhodot")
ax2=plt.plot(tw,p5rhodot,color='b',label="Idealized Flight Path, 5km Radius")
ax1=plt.plot(tw,h5rhodot,color='r',label="Simulated Flight Path, 5km Radius")
plt.ylabel('hrs^{-1}')
plt.legend()
plt.autoscale(enable=True, axis='x', tight=True)
plt.ylim([-1.3,1.3])

s3 = plt.subplot(223)
#ax3=plt.plot(tw,rhodot,color='k',label="True Rhodot")
ax2=plt.plot(tw,p10rhodot,color='b',label="Idealized Flight Path, 10km Radius")
ax1=plt.plot(tw,h10rhodot,color='r',label="Simulated Flight Path, 10km Radius")
plt.ylabel('hrs^{-1}')
plt.legend()
plt.autoscale(enable=True, axis='x', tight=True)
plt.ylim([-1.3,1.3])

s4 = plt.subplot(224)
#ax3=plt.plot(tw,rhodot,color='k',label="True Rhodot")
ax2=plt.plot(tw,p15rhodot,color='b',label="Idealized Flight Path, 15km Radius")
ax1=plt.plot(tw,h15rhodot,color='r',label="Simulated Flight Path, 15km Radius")
plt.ylabel('hrs^{-1}')
plt.legend()
plt.autoscale(enable=True, axis='x', tight=True)
plt.ylim([-1.3,1.3])

plt.show()






