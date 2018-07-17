# -*- coding: utf-8 -*-
"""
Created on Tue Jul 17 12:28:31 2018

@author: pnolan86
"""
import matplotlib
matplotlib.use('Agg')
import seaborn as sns
import h5py as hp
import numpy as np
import scipy.interpolate as sint
sns.set_style('ticks')
matplotlib.rcParams['mathtext.fontset'] = 'cm'
matplotlib.rcParams['text.usetex']=True
import matplotlib.pyplot as plt
plt.rc('font', family='serif')


plt.close('all')
phi = 1.0/1.61803398875
#figheight = 4.5
figwidth = 6
FigSize=(figwidth, figwidth*phi)
def get_axis_limits(ax, scale=.9):
    return ax.get_xlim()[1]*scale, ax.get_ylim()[1]*scale

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

with hp.File('850mb_300m_10min_NAM_FTLE_Origin_t=4-215hrs_Sept2017_int=-1.hdf5','r') as data:
    ftle1 = 3600*data['ftle'][:].squeeze()
    t1 = data['t'][:].squeeze()
    data.close()
    
with hp.File('850mb_300m_10min_NAM_FTLE_Origin_t=4-215hrs_Sept2017_int=-2.hdf5','r') as data:
    ftle2 = 3600*data['ftle'][:].squeeze()
    t2 = data['t'][:].squeeze()
    data.close()
    
with hp.File('850mb_300m_10min_NAM_FTLE_Origin_t=4-215hrs_Sept2017_int=-3.hdf5','r') as data:
    ftle3 = 3600*data['ftle'][:].squeeze()
    t3 = data['t'][:].squeeze()
    data.close()
    
with hp.File('850mb_300m_10min_NAM_FTLE_Origin_t=4-215hrs_Sept2017_int=-4.hdf5','r') as data:
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

#'''
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



fig = plt.figure(1,figsize=FigSize)
ax=plt.plot(tw,rhodot,color='k',label="True Rhodot")
ax2=plt.plot(tw,h2rhodot,color='b',label="Simulated Flight Path, 2km Radius")
ax5=plt.plot(tw,h5rhodot,color='g',label="Simulated Flight Path, 5km Radius")
ax10=plt.plot(tw,h10rhodot,color='m',label="Simulated Flight Path, 10km Radius")
ax15=plt.plot(tw,h15rhodot,color='c',label="Simulated Flight Path, 15km Radius")
#plt.axhline(mean,color='k')
#plt.axhline(mean+std,color='k')
#plt.axhline(mean-std,color='k')
plt.ylabel('$hrs^{-1}$',fontsize=10)
plt.legend()
plt.autoscale(enable=True, axis='x', tight=True)
plt.ylim([-1.5,1.5])
plt.yticks(fontsize=8)
plt.xticks(fontsize=8)
plt.savefig('rhodot_idealized.tif', transparent=True, bbox_inches='tight',pad_inches=0)

fig = plt.figure(2,figsize=FigSize)
ax=plt.plot(tw,rhodot,color='k',label="True Rhodot")
ax2=plt.plot(tw,p2rhodot,color='b',label="Idealized Flight Path, 2km Radius")
ax2=plt.plot(tw,p5rhodot,color='g',label="Idealized Flight Path, 5km Radius")
ax2=plt.plot(tw,p10rhodot,color='m',label="Idealized Flight Path, 10km Radius")
ax2=plt.plot(tw,p15rhodot,color='c',label="Idealized Flight Path, 15km Radius")
#plt.axhline(mean,color='k')
#plt.axhline(mean+std,color='k')
#plt.axhline(mean-std,color='k')
plt.ylabel('$hrs^{-1}$',fontsize=10)
plt.legend()
plt.autoscale(enable=True, axis='x', tight=True)
plt.ylim([-1.5,1.5])
plt.yticks(fontsize=8)
plt.xticks(fontsize=8)
#plt.savefig('rhodot_simulated.eps', transparent=True, bbox_inches='tight',pad_inches=0)

fig = plt.figure(3,figsize=FigSize)
#fig = plt.figure(3,figsize=FigSize)
sub1 = plt.subplot(221)
#ax3=plt.plot(tw,rhodot,color='k',label="True Rhodot")
ax2=plt.plot(tw,p2rhodot,color='b',label="Idealized, 2km")
ax1=plt.plot(tw,h2rhodot,color='r',label="Simulated, 2km")
plt.tick_params(labelbottom='off')
#ax2.tick_params(labelbottom='off')

plt.ylabel('$hrs^{-1}$',fontsize=10)
plt.legend()
plt.autoscale(enable=True, axis='x', tight=True)
plt.yticks(fontsize=8)
plt.xticks(fontsize=8)
plt.ylim([-1.3,1.3])

sub2 = plt.subplot(222)
#ax3=plt.plot(tw,rhodot,color='k',label="True Rhodot")
ax2=plt.plot(tw,p5rhodot,color='b',label="Idealized, 5km")
ax1=plt.plot(tw,h5rhodot,color='r',label="Simulated, 5km")
#plt.ylabel('hrs^{-1}',fontsize=10)
plt.tick_params(labelbottom='off')
plt.tick_params(labelleft='off')
plt.legend()
plt.autoscale(enable=True, axis='x', tight=True)
plt.yticks(fontsize=8)
plt.xticks(fontsize=8)
plt.ylim([-1.3,1.3])

sub3 = plt.subplot(223)
#ax3=plt.plot(tw,rhodot,color='k',label="True Rhodot")
ax2=plt.plot(tw,p10rhodot,color='b',label="Idealized, 10km")
ax1=plt.plot(tw,h10rhodot,color='r',label="Simulated, 10km")
plt.ylabel('$hrs^{-1}$',fontsize=10)
#plt.xlabel('hrs',fontsize=10)
plt.legend()
plt.autoscale(enable=True, axis='x', tight=True)
plt.yticks(fontsize=8)
plt.xticks(fontsize=8)
plt.ylim([-1.3,1.3])

sub4 = plt.subplot(224)
#ax3=plt.plot(tw,rhodot,color='k',label="True Rhodot")
ax2=plt.plot(tw,p15rhodot,color='b',label="Idealized, 15km")
ax1=plt.plot(tw,h15rhodot,color='r',label="Simulated, 15km")
#plt.xlabel('hrs',fontsize=10)
#plt.ylabel('hrs^{-1}',fontsize=10)
plt.tick_params(labelleft='off')
plt.legend()
plt.autoscale(enable=True, axis='x', tight=True)
plt.yticks(fontsize=8)
plt.xticks(fontsize=8)
plt.ylim([-1.3,1.3])
#plt.savefig('rhodot_idealized_vs_simulated.eps', transparent=True, bbox_inches='tight',pad_inches=0)

fig = plt.figure(4,figsize=FigSize)
ax=plt.plot(tw,s1,color='k',label="True s1")
ax2=plt.plot(tw,h2s1,color='b',label="Simulated Flight Path, 2km Radius")
ax5=plt.plot(tw,h5s1,color='g',label="Simulated Flight Path, 5km Radius")
ax10=plt.plot(tw,h10s1,color='m',label="Simulated Flight Path, 10km Radius")
ax15=plt.plot(tw,h15s1,color='c',label="Simulated Flight Path, 15km Radius")
#plt.axhline(mean,color='k')
#plt.axhline(mean+std,color='k')
#plt.axhline(mean-std,color='k')
plt.ylabel('$hrs^{-1}$',fontsize=10)
plt.legend(loc=3)
plt.autoscale(enable=True, axis='x', tight=True)
plt.ylim([-2.5,0.5])
plt.yticks(fontsize=8)
plt.xticks(fontsize=8)
#plt.savefig('s1_idealized.eps', transparent=True, bbox_inches='tight',pad_inches=0)

fig = plt.figure(5,figsize=FigSize)
ax=plt.plot(tw,s1,color='k',label="True s1")
ax2=plt.plot(tw,p2s1,color='b',label="Idealized Flight Path, 2km Radius")
ax2=plt.plot(tw,p5s1,color='g',label="Idealized Flight Path, 5km Radius")
ax2=plt.plot(tw,p10s1,color='m',label="Idealized Flight Path, 10km Radius")
ax2=plt.plot(tw,p15s1,color='c',label="Idealized Flight Path, 15km Radius")
#plt.axhline(mean,color='k')
#plt.axhline(mean+std,color='k')
#plt.axhline(mean-std,color='k')
plt.ylabel('$hrs^{-1}$',fontsize=10)
plt.legend(loc=3)
plt.autoscale(enable=True, axis='x', tight=True)
plt.ylim([-2.5,0.5])
plt.yticks(fontsize=8)
plt.xticks(fontsize=8)
#plt.savefig('s1_simulated.eps', transparent=True, bbox_inches='tight',pad_inches=0)

fig = plt.figure(6,figsize=FigSize)
#fig = plt.figure(3,figsize=FigSize)
sub1 = plt.subplot(221)
#ax3=plt.plot(tw,rhodot,color='k',label="True Rhodot")
ax2=plt.plot(tw,p2s1,color='b',label="Idealized, 2km")
ax1=plt.plot(tw,h2s1,color='r',label="Simulated, 2km")
plt.tick_params(labelbottom='off')
#ax2.tick_params(labelbottom='off')
plt.ylabel('$hrs^{-1}$',fontsize=10)
plt.legend(loc=4)
plt.autoscale(enable=True, axis='x', tight=True)
plt.yticks(fontsize=8)
plt.xticks(fontsize=8)
plt.ylim([-2.1,0.5])

sub2 = plt.subplot(222)
#ax3=plt.plot(tw,rhodot,color='k',label="True Rhodot")
ax2=plt.plot(tw,p5s1,color='b',label="Idealized, 5km")
ax1=plt.plot(tw,h5s1,color='r',label="Simulated, 5km")
#plt.ylabel('hrs^{-1}',fontsize=10)
plt.tick_params(labelbottom='off')
plt.tick_params(labelleft='off')
plt.legend(loc=4)
plt.autoscale(enable=True, axis='x', tight=True)
plt.yticks(fontsize=8)
plt.xticks(fontsize=8)
plt.ylim([-2.1,0.5])

sub3 = plt.subplot(223)
#ax3=plt.plot(tw,rhodot,color='k',label="True Rhodot")
ax2=plt.plot(tw,p10s1,color='b',label="Idealized, 10km")
ax1=plt.plot(tw,h10s1,color='r',label="Simulated, 10km")
plt.ylabel('$hrs^{-1}$',fontsize=10)
#plt.xlabel('hrs',fontsize=10)
plt.legend(loc=4)
plt.autoscale(enable=True, axis='x', tight=True)
plt.yticks(fontsize=8)
plt.xticks(fontsize=8)
plt.ylim([-2.1,0.5])

sub4 = plt.subplot(224)
#ax3=plt.plot(tw,rhodot,color='k',label="True Rhodot")
ax2=plt.plot(tw,p15s1,color='b',label="Idealized, 15km")
ax1=plt.plot(tw,h15s1,color='r',label="Simulated, 15km")
sub4.annotate('A', xy=get_axis_limits(sub4,0.85))
#plt.xlabel('hrs',fontsize=10)
#plt.ylabel('hrs^{-1}',fontsize=10)
plt.tick_params(labelleft='off')
plt.legend(loc=4)
plt.autoscale(enable=True, axis='x', tight=True)
plt.yticks(fontsize=8)
plt.xticks(fontsize=8)
plt.ylim([-2.1,0.5])
#plt.savefig('s1_idealized_vs_simulated.eps', transparent=True, bbox_inches='tight',pad_inches=0)


fig = plt.figure(7,figsize=FigSize)
plt.axhline(0,color='k')
ax1=plt.plot(tw,rhodot,color='purple',label="Rhodot")
axf1=plt.plot(tw,-ftle1,color='b',label="FTLE -1hr")
axf4=plt.plot(tw,-ftle4,color='c',label="FTLE -4hr")
plt.autoscale(enable=True, axis='x', tight=True)
plt.legend(loc=4)
plt.yticks(fontsize=8)
plt.xticks(fontsize=8)
plt.ylim([-2.1,1.25])
#plt.savefig('rhodot_vs_FTLE.eps', transparent=True, bbox_inches='tight',pad_inches=0)

fig = plt.figure(8,figsize=FigSize)
plt.axhline(0,color='k')
ax1=plt.plot(tw,rhodot,color='purple',label="Rhodot")
axf1=plt.plot(tw,-ftle1,color='b',label="FTLE -1hr")
axf4=plt.plot(tw,-ftle4,color='c',label="FTLE -4hr")
plt.autoscale(enable=True, axis='x', tight=True)
plt.legend(loc=4)
plt.yticks(fontsize=8)
plt.xticks(fontsize=8)
plt.ylim([-2.1,1.25])
plt.xlim([tw.min(),70])
#plt.savefig('rhodot_vs_FTLE_closeup.eps', transparent=True, bbox_inches='tight',pad_inches=0)

fig = plt.figure(9,figsize=FigSize)
plt.axhline(0,color='k')
ax2=plt.plot(tw,s1,color='r',label="S1")
axf1=plt.plot(tw,-ftle1,color='b',label="FTLE -1hr")
axf4=plt.plot(tw,-ftle4,color='c',label="FTLE -4hr")
plt.autoscale(enable=True, axis='x', tight=True)
plt.legend(loc=4)
plt.yticks(fontsize=8)
plt.xticks(fontsize=8)
plt.ylim([-2.1,1.25])
#plt.savefig('s1_FTLE.eps', transparent=True, bbox_inches='tight',pad_inches=0)
#'''
fig = plt.figure(10,figsize=FigSize)
plt.axhline(0,color='k')
ax2=plt.plot(tw,s1,color='r',label="S1")
axf1=plt.plot(tw,-ftle1,color='b',label="FTLE -1hr")
axf4=plt.plot(tw,-ftle4,color='c',label="FTLE -4hr")
plt.autoscale(enable=True, axis='x', tight=True)
plt.legend(loc=4)
plt.yticks(fontsize=8)
plt.xticks(fontsize=8)
plt.ylim([-2.1,1.25])
plt.xlim([tw.min(),70])
#plt.savefig('s1_FTLE_closeup.png', transparent=True, bbox_inches='tight',pad_inches=0)



