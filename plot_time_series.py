# -*- coding: utf-8 -*-
"""
Created on Wed Jun 13 15:59:44 2018

@author: pnola
"""
import h5py as hp
#from matplotlib import use, rc
import matplotlib
#matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.close('all')
#matplotlib.rcParams['text.usetex']=True
#matplotlib.rcParams['mathtext.fontset'] = 'cm'
#plt.rc('font', **{'family': 'serif', 'serif': ['cmr10']})
import matplotlib.font_manager as font_manager
import cRidge_passing_times as cr
# Define font styles as dictionaries
'''
titlefont = {'fontsize':12,'family':'serif','fontname':'cmr10'}
labelfont = {'fontsize':10,'family':'serif','fontname':'cmr10'}
tickfont = {'fontsize':8,'family':'serif'}#,'fontname':'cmr10'}
font = font_manager.FontProperties(family='serif',style='normal', size=8)
'''
titlefont = {'fontsize':12}
labelfont = {'fontsize':10}
tickfont = {'fontsize':8}

font = font_manager.FontProperties(style='normal', size=8)
#'''
plt.close('all')
phi = 1.0/1.61803398875
#figheight = 4.5
figwidth = 6
FigSize=(figwidth, figwidth*phi)
def get_axis_limits(ax, scale=.9):
    return ax.get_xlim()[1]*scale, ax.get_ylim()[1]*scale

with hp.File('PlottingData.hdf5','r') as data:
    tw=data['tw'][:]
    ftle1=data['ftle1'][:]
    ftle2=data['ftle2'][:]
    ftle3=data['ftle3'][:]
    ftle4=data['ftle4'][:]
    rhodot=data['rhodot'][:]
    h2rhodot=data['h2rhodot'][:]
    h5rhodot=data['h5rhodot'][:]
    h10rhodot=data['h10rhodot'][:]
    h15rhodot=data['h15rhodot'][:]
    p2rhodot=data['p2rhodot'][:]
    p5rhodot=data['p5rhodot'][:]
    p10rhodot=data['p10rhodot'][:]
    p15rhodot=data['p15rhodot'][:]
    s1=data['s1'][:]
    h2s1=data['h2s1'][:]
    h5s1=data['h5s1'][:]
    h10s1=data['h10s1'][:]
    h15s1=data['h15s1'][:]
    p2s1=data['p2s1'][:]
    p5s1=data['p5s1'][:]
    p10s1=data['p10s1'][:]
    p15s1=data['p15s1'][:]
    data.close()
    

'''              
import pandas as pd
Alldata = pd.DataFrame(np.transpose([ftle1,ftle2,ftle3,ftle4,rhodot,p2rhodot,p5rhodot,p10rhodot,p15rhodot,h2rhodot,h5rhodot,h10rhodot,h15rhodot,s1,p2s1,p5s1,p10s1,p15s1,h2s1,h5s1,h10s1,h15s1]),columns=['FTLE int=-1','FTLE int=-2','FTLE int=-3','FTLE int=-4','rhodot','rd 2km path','rhodot 5km path','rhodot 10km path','rhodot 15km path','rhodot 2km simulation','rhodot 5km simulation','rhodot 10km simulation','rhodot 15km simulation','s1','s1 2km path','s1 5km path','s1 10km path','s1 15km path','s1 2km simulation','s1 5km simulation','s1 10km simulation','s1 15km simulation'])
Alldata.corr().to_csv('Correlation_and_FLight_stats.csv',mode='w')
B=Alldata.describe()
B.to_csv('Correlation_and_FLight_stats.csv',mode='a')
mean = B['rhodot'][1]
std = B['rhodot'][2]
rhodata = pd.DataFrame(np.transpose([rhodot,p2rhodot,p5rhodot,p10rhodot,p15rhodot,h2rhodot,h5rhodot,h10rhodot,h15rhodot]),columns=['rhodot','2km path','5km path','10km path','15km path','2km simulation','5km simulation','10km simulation','15km simulation'])
rhodata.corr().to_csv('Correlation_and_FLight_stats.csv',mode='a')
rhodata.describe().to_csv('Correlation_and_FLight_stats.csv',mode='a')
s1data = pd.DataFrame(np.transpose([s1,p2s1,p5s1,p10s1,p15s1,h2s1,h5s1,h10s1,h15s1]),columns=['s1','2km path','5km path','10km path','15km path','2km simulation','5km simulation','10km simulation','15km simulation'])
s1data.corr().to_csv('Correlation_and_FLight_stats.csv',mode='a')
s1data.describe().to_csv('Correlation_and_FLight_stats.csv',mode='a')
ftledata = pd.DataFrame(np.transpose([ftle1,ftle2,ftle3,ftle4]),columns=['FTLE int=-1','FTLE int=-2','FTLE int=-3','FTLE int=-4'])
ftledata.corr().to_csv('Correlation_and_FLight_stats.csv',mode='a')
ftledata.describe().to_csv('Correlation_and_FLight_stats.csv',mode='a')
#'''

'''
fig = plt.figure(1,figsize=FigSize)
ax=plt.plot(tw,rhodot,color='k',label="True $\dot{\\rho}$")
ax2=plt.plot(tw,h2rhodot,color='b',label="Simulated Flight Path, 2km Radius")
ax5=plt.plot(tw,h5rhodot,color='g',label="Simulated Flight Path, 5km Radius")
ax10=plt.plot(tw,h10rhodot,color='m',label="Simulated Flight Path, 10km Radius")
ax15=plt.plot(tw,h15rhodot,color='c',label="Simulated Flight Path, 15km Radius")
#plt.axhline(mean,color='k')
#plt.axhline(mean+std,color='k')
#plt.axhline(mean-std,color='k')
plt.ylabel('hr$^{-1}$',**labelfont)
plt.legend(prop = font)
plt.autoscale(enable=True, axis='x', tight=True)
plt.ylim([-1.5,1.5])
plt.yticks(**tickfont)
plt.xticks(**tickfont)
plt.savefig('rhodot_simulated.eps', transparent=True, bbox_inches='tight',pad_inches=0)

fig = plt.figure(2,figsize=FigSize)
ax=plt.plot(tw,rhodot,color='k',label="True $\dot{\\rho}$")
ax2=plt.plot(tw,p2rhodot,color='b',label="Idealized Flight Path, 2km Radius")
ax2=plt.plot(tw,p5rhodot,color='g',label="Idealized Flight Path, 5km Radius")
ax2=plt.plot(tw,p10rhodot,color='m',label="Idealized Flight Path, 10km Radius")
ax2=plt.plot(tw,p15rhodot,color='c',label="Idealized Flight Path, 15km Radius")
#plt.axhline(mean,color='k')
#plt.axhline(mean+std,color='k')
#plt.axhline(mean-std,color='k')
plt.ylabel('hr$^{-1}$',**labelfont)
plt.legend(prop = font)
plt.autoscale(enable=True, axis='x', tight=True)
plt.ylim([-1.5,1.5])
plt.yticks(**tickfont)
plt.xticks(**tickfont)
plt.savefig('rhodot_idealized.eps', transparent=True, bbox_inches='tight',pad_inches=0)

fig = plt.figure(3,figsize=FigSize)
#fig = plt.figure(3,figsize=FigSize)
sub1 = plt.subplot(221)
#ax3=plt.plot(tw,rhodot,color='k',label="True Rhodot")
ax2=plt.plot(tw,p2rhodot,color='b',label="Idealized, 2km")
ax1=plt.plot(tw,h2rhodot,color='r',label="Simulated, 2km")
plt.tick_params(labelbottom='off')
#ax2.tick_params(labelbottom='off')

plt.ylabel('hr$^{-1}$',**labelfont)
plt.legend(prop = font)
plt.autoscale(enable=True, axis='x', tight=True)
plt.yticks(**tickfont)
plt.xticks(**tickfont)
plt.ylim([-1.3,1.3])

sub2 = plt.subplot(222)
#ax3=plt.plot(tw,rhodot,color='k',label="True $\dot{\\rho}$")
ax2=plt.plot(tw,p5rhodot,color='b',label="Idealized, 5km")
ax1=plt.plot(tw,h5rhodot,color='r',label="Simulated, 5km")
#plt.ylabel('hrs^{-1}',**labelfont)
plt.tick_params(labelbottom='off')
plt.tick_params(labelleft='off')
plt.legend(prop = font)
plt.autoscale(enable=True, axis='x', tight=True)
plt.yticks(**tickfont)
plt.xticks(**tickfont)
plt.ylim([-1.3,1.3])

sub3 = plt.subplot(223)
#ax3=plt.plot(tw,rhodot,color='k',label="True $\dot{\\rho}$")
ax2=plt.plot(tw,p10rhodot,color='b',label="Idealized, 10km")
ax1=plt.plot(tw,h10rhodot,color='r',label="Simulated, 10km")
plt.ylabel('hr$^{-1}$',**labelfont)
#plt.xlabel('hrs',**labelfont)
plt.legend(prop = font)
plt.autoscale(enable=True, axis='x', tight=True)
plt.yticks(**tickfont)
plt.xticks(**tickfont)
plt.ylim([-1.3,1.3])

sub4 = plt.subplot(224)
#ax3=plt.plot(tw,rhodot,color='k',label="True $\dot{\\rho}$")
ax2=plt.plot(tw,p15rhodot,color='b',label="Idealized, 15km")
ax1=plt.plot(tw,h15rhodot,color='r',label="Simulated, 15km")
#plt.xlabel('hrs',**labelfont)
#plt.ylabel('hrs^{-1}',**labelfont)
plt.tick_params(labelleft='off')
plt.legend(prop = font)
plt.autoscale(enable=True, axis='x', tight=True)
plt.yticks(**tickfont)
plt.xticks(**tickfont)
plt.ylim([-1.3,1.3])
plt.savefig('rhodot_idealized_vs_simulated.eps', transparent=True, bbox_inches='tight',pad_inches=0)

fig = plt.figure(4,figsize=FigSize)
ax=plt.plot(tw,s1,color='k',label="True $s_{1}$")
ax2=plt.plot(tw,h2s1,color='b',label="Simulated Flight Path, 2km Radius")
ax5=plt.plot(tw,h5s1,color='g',label="Simulated Flight Path, 5km Radius")
ax10=plt.plot(tw,h10s1,color='m',label="Simulated Flight Path, 10km Radius")
ax15=plt.plot(tw,h15s1,color='c',label="Simulated Flight Path, 15km Radius")
#plt.axhline(mean,color='k')
#plt.axhline(mean+std,color='k')
#plt.axhline(mean-std,color='k')
plt.ylabel('hr$^{-1}$',**labelfont)
plt.legend(prop = font,loc=3)
plt.autoscale(enable=True, axis='x', tight=True)
plt.ylim([-2.5,0.5])
plt.yticks(**tickfont)
plt.xticks(**tickfont)
plt.savefig('s1_simulated.eps', transparent=True, bbox_inches='tight',pad_inches=0)

fig = plt.figure(5,figsize=FigSize)
ax=plt.plot(tw,s1,color='k',label="True $s_{1}$")
ax2=plt.plot(tw,p2s1,color='b',label="Idealized Flight Path, 2km Radius")
ax2=plt.plot(tw,p5s1,color='g',label="Idealized Flight Path, 5km Radius")
ax2=plt.plot(tw,p10s1,color='m',label="Idealized Flight Path, 10km Radius")
ax2=plt.plot(tw,p15s1,color='c',label="Idealized Flight Path, 15km Radius")
#plt.axhline(mean,color='k')
#plt.axhline(mean+std,color='k')
#plt.axhline(mean-std,color='k')
plt.ylabel('hr$^{-1}$',**labelfont)
plt.legend(prop = font,loc=3)
plt.autoscale(enable=True, axis='x', tight=True)
plt.ylim([-2.5,0.5])
plt.yticks(**tickfont)
plt.xticks(**tickfont)
plt.savefig('s1_idealized.eps', transparent=True, bbox_inches='tight',pad_inches=0)

fig = plt.figure(6,figsize=FigSize)
#fig = plt.figure(3,figsize=FigSize)
sub1 = plt.subplot(221)
#ax3=plt.plot(tw,rhodot,color='k',label="True $\dot{\\rho}$")
ax2=plt.plot(tw,p2s1,color='b',label="Idealized, 2km")
ax1=plt.plot(tw,h2s1,color='r',label="Simulated, 2km")
plt.tick_params(labelbottom='off')
#ax2.tick_params(labelbottom='off')
plt.ylabel('hr$^{-1}$',**labelfont)
plt.legend(prop = font,loc=4)
plt.autoscale(enable=True, axis='x', tight=True)
plt.yticks(**tickfont)
plt.xticks(**tickfont)
plt.ylim([-2.1,0.5])

sub2 = plt.subplot(222)
#ax3=plt.plot(tw,rhodot,color='k',label="True $\dot{\\rho}$")
ax2=plt.plot(tw,p5s1,color='b',label="Idealized, 5km")
ax1=plt.plot(tw,h5s1,color='r',label="Simulated, 5km")
#plt.ylabel('hrs^{-1}',**labelfont)
plt.tick_params(labelbottom='off')
plt.tick_params(labelleft='off')
plt.legend(prop = font,loc=4)
plt.autoscale(enable=True, axis='x', tight=True)
plt.yticks(**tickfont)
plt.xticks(**tickfont)
plt.ylim([-2.1,0.5])

sub3 = plt.subplot(223)
#ax3=plt.plot(tw,rhodot,color='k',label="True $\dot{\\rho}$")
ax2=plt.plot(tw,p10s1,color='b',label="Idealized, 10km")
ax1=plt.plot(tw,h10s1,color='r',label="Simulated, 10km")
plt.ylabel('hr$^{-1}$',**labelfont)
#plt.xlabel('hrs',**labelfont)
plt.legend(prop = font,loc=4)
plt.autoscale(enable=True, axis='x', tight=True)
plt.yticks(**tickfont)
plt.xticks(**tickfont)
plt.ylim([-2.1,0.5])

sub4 = plt.subplot(224)
#ax3=plt.plot(tw,rhodot,color='k',label="True $\dot{\\rho}$")
ax2=plt.plot(tw,p15s1,color='b',label="Idealized, 15km")
ax1=plt.plot(tw,h15s1,color='r',label="Simulated, 15km")
sub4.annotate('A', xy=get_axis_limits(sub4,0.85))
#plt.xlabel('hrs',**labelfont)
#plt.ylabel('hrs^{-1}',**labelfont)
plt.tick_params(labelleft='off')
plt.legend(prop = font,loc=4)
plt.autoscale(enable=True, axis='x', tight=True)
plt.yticks(**tickfont)
plt.xticks(**tickfont)
plt.ylim([-2.1,0.5])
plt.savefig('s1_idealized_vs_simulated.eps', transparent=True, bbox_inches='tight',pad_inches=0)


fig = plt.figure(7,figsize=FigSize)
plt.axhline(0,color='k')
ax1=plt.plot(tw,rhodot,color='purple',label="$\dot{\\rho}$")
axf1=plt.plot(tw,-ftle1,color='b',label="FTLE -1hr")
axf4=plt.plot(tw,-ftle4,color='c',label="FTLE -4hr")
plt.autoscale(enable=True, axis='x', tight=True)
plt.legend(prop = font,loc=4)
plt.yticks(**tickfont)
plt.xticks(**tickfont)
plt.ylabel('hr$^{-1}$',**labelfont)
plt.ylim([-2.1,1.25])
plt.savefig('rhodot_vs_FTLE.eps', transparent=True, bbox_inches='tight',pad_inches=0)

fig = plt.figure(8,figsize=FigSize)
plt.axhline(0,color='k')
ax1=plt.plot(tw,rhodot,color='purple',label="$\dot{\\rho}$")
axf1=plt.plot(tw,-ftle1,color='b',label="FTLE -1hr")
axf4=plt.plot(tw,-ftle4,color='c',label="FTLE -4hr")
plt.autoscale(enable=True, axis='x', tight=True)
plt.legend(prop = font,loc=4)
plt.yticks(**tickfont)
plt.xticks(**tickfont)
plt.ylabel('hr$^{-1}$',**labelfont)
plt.ylim([-2.1,1.25])
plt.xlim([tw.min(),70])
plt.savefig('rhodot_vs_FTLE_closeup.eps', transparent=True, bbox_inches='tight',pad_inches=0)

fig = plt.figure(9,figsize=FigSize)
plt.axhline(0,color='k')
ax2=plt.plot(tw,s1,color='r',label="$s_{1}$")
axf1=plt.plot(tw,-ftle1,color='b',label="FTLE -1hr")
axf4=plt.plot(tw,-ftle4,color='c',label="FTLE -4hr")
plt.autoscale(enable=True, axis='x', tight=True)
plt.legend(prop = font,loc=4)
plt.yticks(**tickfont)
plt.xticks(**tickfont)
plt.ylabel('hr$^{-1}$',**labelfont)
plt.ylim([-2.1,1.25])
plt.savefig('s1_FTLE.eps', transparent=True, bbox_inches='tight',pad_inches=0)

fig = plt.figure(10,figsize=FigSize)
plt.axhline(0,color='k')
ax2=plt.plot(tw,s1,color='r',label="$s_{1}$")
axf1=plt.plot(tw,-ftle1,color='b',label="FTLE -1hr")
axf4=plt.plot(tw,-ftle4,color='c',label="FTLE -4hr")
plt.autoscale(enable=True, axis='x', tight=True)
plt.legend(prop = font,loc=4)
plt.yticks(**tickfont)
plt.xticks(**tickfont)
plt.ylabel('hr$^{-1}$',**labelfont)
plt.ylim([-2.1,1.25])
plt.xlim([tw.min(),70])
plt.savefig('s1_FTLE_closeup.eps', transparent=True, bbox_inches='tight',pad_inches=0)
#'''

import numpy as np
ridges = cr.hr_4_percentile_90()
li4 = np.percentile(ftle4,90,axis=None)
li1 = np.percentile(ftle1,50,axis=None)
lis1 = -np.percentile(-s1,90,axis=None)
lird = -np.percentile(-h2rhodot[h2rhodot<=0],90,axis=None)
fig = plt.figure(11,figsize=FigSize)
[plt.axvline(_x, linewidth=1,alpha=0.3, color='b') for _x in ridges]
plt.axhline(0,color='k')
ax1=plt.plot(tw,h2s1,color='r',label="$s_{1}$, 2km")
plt.axhline(lis1,color='r',linestyle='dashed')
ax2=plt.plot(tw[h2rhodot<=0],h2rhodot[h2rhodot<=0],color='purple',label="$\dot{\\rho}$, 2km")
plt.axhline(lird,color='purple',linestyle='dashed')
axf4=plt.plot(tw,-ftle4,color='c',label="FTLE -4hr")
plt.axhline(-li4,color='c',linestyle='dashed')
plt.autoscale(enable=True, axis='x', tight=True)
plt.legend(prop = font,loc=4)
plt.yticks(**tickfont)
plt.xticks(**tickfont)
plt.ylabel('hr$^{-1}$',**labelfont)
plt.ylim([-2.1,0.25])
plt.savefig('s1_rhodot_FTLE.eps', transparent=True, bbox_inches='tight',pad_inches=0)


fig = plt.figure(12,figsize=FigSize)
plt.axhline(0,color='k')
ax1=plt.plot(tw,h2s1,color='r',label="$s_{1}$, 2km")
plt.axhline(lis1,color='r',linestyle='dashed')
ax2=plt.plot(tw[h2rhodot<=0],h2rhodot[h2rhodot<=0],color='purple',label="$\dot{\\rho}$, 2km")
plt.axhline(lird,color='purple',linestyle='dashed')
axf1=plt.plot(tw,-ftle1,color='b',label="FTLE -1hr")
plt.axhline(-li1,color='b',linestyle='dashed')
plt.autoscale(enable=True, axis='x', tight=True)
plt.legend(prop = font,loc=4)
plt.yticks(**tickfont)
plt.xticks(**tickfont)
plt.ylabel('hr$^{-1}$',**labelfont)
plt.ylim([-2.1,0.25])
plt.savefig('s1_rhodot_FTLE.eps', transparent=True, bbox_inches='tight',pad_inches=0)

'''
fig = plt.figure(12,figsize=FigSize)
plt.axhline(0,color='k')
plt.axhline(-li4,color='g')
ax1=plt.plot(tw,h2s1,color='r',label="$s_{1}$, 2km")
ax2=plt.plot(tw,h2rhodot,color='purple',label="$\dot{\\rho}$, 2km")
axf1=plt.plot(tw,-ftle1,color='b',label="FTLE -1hr")
axf4=plt.plot(tw,-ftle1,color='c',label="FTLE -4hr")
plt.autoscale(enable=True, axis='x', tight=True)
plt.legend(prop = font,loc=4)
plt.yticks(**tickfont)
plt.xticks(**tickfont)
plt.ylabel('hr$^{-1}$',**labelfont)
plt.ylim([-2.1,1.25])
plt.xlim([tw.min(),70])
plt.savefig('s1_rhodot_FTLE_closeup.eps', transparent=True, bbox_inches='tight',pad_inches=0)
'''
plt.show()
