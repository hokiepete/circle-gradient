# -*- coding: utf-8 -*-
"""
Created on Wed Jun 13 15:59:44 2018

@author: pnola
"""
import h5py as hp
import numpy as np
import scipy.interpolate as sint

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

with hp.File('850mb_300m_10min_NAM_FTLE_Origin_t=0-211hrs_Sept2017_int=-1.hdf5','r') as data:
    ftle1 = 3600*data['ftle'][:].squeeze()
    t1 = data['t'][:].squeeze()
    data.close()
    
with hp.File('850mb_300m_10min_NAM_FTLE_Origin_t=0-211hrs_Sept2017_int=-2.hdf5','r') as data:
    ftle2 = 3600*data['ftle'][:].squeeze()
    t2 = data['t'][:].squeeze()
    data.close()
    
with hp.File('850mb_300m_10min_NAM_FTLE_Origin_t=0-211hrs_Sept2017_int=-3.hdf5','r') as data:
    ftle3 = 3600*data['ftle'][:].squeeze()
    t3 = data['t'][:].squeeze()
    data.close()
    
with hp.File('850mb_300m_10min_NAM_FTLE_Origin_t=0-211hrs_Sept2017_int=-4.hdf5','r') as data:
    ftle4 = 3600*data['ftle'][:].squeeze()
    t4 = data['t'][:].squeeze()
    data.close()

with hp.File('850mb_300m_10min_NAM_Rhodot_Origin_t=0-215hrs_Sept2017.hdf5','r') as data:
    rhodot = 3600*data['rhodot'][:].squeeze()
    s1 = 3600*data['s1'][:].squeeze()
    t = data['t'][:].squeeze()
    data.close()

with hp.File('simflightdata_2000.mat','r') as data:
    p2rhodot = 3600*data['rhodot'][:].squeeze()
    p2s1 = 3600*data['s1'][:].squeeze()
    p2t = 3600*data['timeout'][:].squeeze()
    p2theta = data['thetaout'][:].squeeze()
    data.close()

with hp.File('simflightdata_5000.mat','r') as data:
    p5rhodot = 3600*data['rhodot'][:].squeeze()
    p5s1 = 3600*data['s1'][:].squeeze()
    p5t = 3600*data['timeout'][:].squeeze()
    p5theta = data['thetaout'][:].squeeze()
    data.close()

with hp.File('simflightdata_10000.mat','r') as data:
    p10rhodot = 3600*data['rhodot'][:].squeeze()
    p10s1 = 3600*data['s1'][:].squeeze()
    p10t = 3600*data['timeout'][:].squeeze()
    p10theta = data['thetaout'][:].squeeze()
    data.close()
    
with hp.File('simflightdata_15000.mat','r') as data:
    p15rhodot = 3600*data['rhodot'][:].squeeze()
    p15s1 = 3600*data['s1'][:].squeeze()
    p15t = 3600*data['timeout'][:].squeeze()
    p15theta = data['thetaout'][:].squeeze()
    data.close()
    
with hp.File('hunterdata_r=02km.mat','r') as data:
    h2rhodot = 3600*data['rhodot'][:].squeeze()
    h2s1 = 3600*data['s1'][:].squeeze()
    h2t = data['timeout'][:].squeeze()
    h2theta = data['thetaout'][:].squeeze()
    data.close()

with hp.File('hunterdata_r=05km.mat','r') as data:
    h5rhodot = 3600*data['rhodot'][:].squeeze()
    h5s1 = 3600*data['s1'][:].squeeze()
    h5t = data['timeout'][:].squeeze()
    h5theta = data['thetaout'][:].squeeze()
    data.close()

with hp.File('hunterdata_r=10km.mat','r') as data:
    h10rhodot = 3600*data['rhodot'][:].squeeze()
    h10s1 = 3600*data['s1'][:].squeeze()
    h10t = data['timeout'][:].squeeze()
    h10theta = data['thetaout'][:].squeeze()
    data.close()

with hp.File('hunterdata_r=15km.mat','r') as data:
    h15rhodot = 3600*data['rhodot'][:].squeeze()
    h15s1 = 3600*data['s1'][:].squeeze()
    h15t = data['timeout'][:].squeeze()
    h15theta = data['thetaout'][:].squeeze()
    data.close()


h2rhodot, delete = circleaverage(h2rhodot,h2t,h2theta)
h2s1, h2t = circleaverage(h2s1,h2t,h2theta)
h5rhodot, delete = circleaverage(h5rhodot,h5t,h5theta)
h5s1, h5t = circleaverage(h5s1,h5t,h5theta)
h10rhodot, delete = circleaverage(h10rhodot,h10t,h10theta)
h10s1, h10t = circleaverage(h10s1,h10t,h10theta)
h15rhodot, delete = circleaverage(h15rhodot,h15t,h15theta)
h15s1, h15t = circleaverage(h15s1,h15t,h15theta)

p2rhodot, delete = circleaverage(p2rhodot,p2t,p2theta)    
p2s1, p2t = circleaverage(p2s1,p2t,p2theta)    
p5rhodot, delete = circleaverage(p5rhodot,p5t,p5theta)    
p5s1, p5t = circleaverage(p5s1,p5t,p5theta)    
p10rhodot, delete = circleaverage(p10rhodot,p10t,p10theta)    
p10s1, p10t = circleaverage(p10s1,p10t,p10theta)    
p15rhodot, delete = circleaverage(p15rhodot,p15t,p15theta)    
p15s1, p15t = circleaverage(p15s1,p15t,p15theta)    
del delete
    
tmin = np.max([t1.min(),t2.min(),t3.min(),t4.min(),t.min(),p2t.min(),p5t.min(),p10t.min(),p15t.min(),h2t.min(),h5t.min(),h10t.min(),h15t.min()])
tmax = np.min([t1.max(),t2.max(),t3.max(),t4.max(),t.max(),p2t.max(),p5t.max(),p10t.max(),p15t.max(),h2t.max(),h5t.max(),h10t.max(),h15t.max()])
tw = h2t[h2t>=tmin]
tw = tw[tw<=tmax]
tcku = sint.splrep(t1, ftle1, s=0)
ftle1 = sint.splev(tw, tcku, der=0)
tcku = sint.splrep(t2, ftle2, s=0)
ftle2 = sint.splev(tw, tcku, der=0)
tcku = sint.splrep(t3, ftle3, s=0)
ftle3 = sint.splev(tw, tcku, der=0)
tcku = sint.splrep(t4, ftle4, s=0)
ftle4 = sint.splev(tw, tcku, der=0)

tcku = sint.splrep(t, rhodot, s=0)
rhodot = sint.splev(tw, tcku, der=0)
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

tcku = sint.splrep(t, s1, s=0)
s1 = sint.splev(tw, tcku, der=0)
tcku = sint.splrep(h2t, h2s1, s=0)
h2s1 = sint.splev(tw, tcku, der=0)
tcku = sint.splrep(h5t, h5s1, s=0)
h5s1 = sint.splev(tw, tcku, der=0)
tcku = sint.splrep(h10t, h10s1, s=0)
h10s1 = sint.splev(tw, tcku, der=0)
tcku = sint.splrep(h15t, h15s1, s=0)
h15s1 = sint.splev(tw, tcku, der=0)
tcku = sint.splrep(p2t, p2s1, s=0)
p2s1 = sint.splev(tw, tcku, der=0)
tcku = sint.splrep(p5t, p5s1, s=0)
p5s1 = sint.splev(tw, tcku, der=0)
tcku = sint.splrep(p10t, p10s1, s=0)
p10s1 = sint.splev(tw, tcku, der=0)
tcku = sint.splrep(p15t, p15s1, s=0)
p15s1 = sint.splev(tw, tcku, der=0)


del tcku, h2t, h5t, h10t, h15t, p2t, p5t, p10t, p15t, t, tmin, tmax

with hp.File('PlottingData.hdf5','w') as savefile:
    savefile.create_dataset('tw',shape=tw.shape,data=tw)
    savefile.create_dataset('ftle1',shape=ftle1.shape,data=ftle1)
    savefile.create_dataset('ftle2',shape=ftle2.shape,data=ftle2)
    savefile.create_dataset('ftle3',shape=ftle3.shape,data=ftle3)
    savefile.create_dataset('ftle4',shape=ftle4.shape,data=ftle4)
    savefile.create_dataset('rhodot',shape=rhodot.shape,data=rhodot)
    savefile.create_dataset('h2rhodot',shape=h2rhodot.shape,data=h2rhodot)
    savefile.create_dataset('h5rhodot',shape=h5rhodot.shape,data=h5rhodot)
    savefile.create_dataset('h10rhodot',shape=h10rhodot.shape,data=h10rhodot)
    savefile.create_dataset('h15rhodot',shape=h15rhodot.shape,data=h15rhodot)
    savefile.create_dataset('p2rhodot',shape=p2rhodot.shape,data=p2rhodot)
    savefile.create_dataset('p5rhodot',shape=p5rhodot.shape,data=p5rhodot)
    savefile.create_dataset('p10rhodot',shape=p10rhodot.shape,data=p10rhodot)
    savefile.create_dataset('p15rhodot',shape=p15rhodot.shape,data=p15rhodot)
    savefile.create_dataset('s1',shape=s1.shape,data=s1)
    savefile.create_dataset('h2s1',shape=h2s1.shape,data=h2s1)
    savefile.create_dataset('h5s1',shape=h5s1.shape,data=h5s1)
    savefile.create_dataset('h10s1',shape=h10s1.shape,data=h10s1)
    savefile.create_dataset('h15s1',shape=h15s1.shape,data=h15s1)
    savefile.create_dataset('p2s1',shape=p2s1.shape,data=p2s1)
    savefile.create_dataset('p5s1',shape=p5s1.shape,data=p5s1)
    savefile.create_dataset('p10s1',shape=p10s1.shape,data=p10s1)
    savefile.create_dataset('p15s1',shape=p15s1.shape,data=p15s1)
    savefile.close()
#'''
