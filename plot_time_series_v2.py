# -*- coding: utf-8 -*-
"""
Created on Wed Jun 13 15:59:44 2018

@author: pnola
"""
import h5py as hp
import matplotlib
#matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.close('all')
matplotlib.rcParams['text.usetex']=True
matplotlib.rcParams['mathtext.fontset'] = 'cm'
plt.rc('font', **{'family': 'serif', 'serif': ['cmr10']})
matplotlib.rcParams['lines.linewidth'] = 1
import matplotlib.font_manager as font_manager
import matplotlib.gridspec as gridspec
# Define font styles as dictionaries
titlefont = {'fontsize':12}
labelfont = {'fontsize':10}
tickfont = {'fontsize':8}

font = font_manager.FontProperties(style='normal', size=8)
#'''
plt.close('all')
phi = 1.0/1.61803398875
#figheight = 4.5
figwidth = 6
FigSize=(figwidth, 5)
FigSize2=(figwidth, 4)
def get_axis_limits(ax, scale=.9):
    return ax.get_xlim()[1]*scale, ax.get_ylim()[1]*scale

with hp.File('PlottingData.hdf5','r') as data:
    tw=data['tw'][:]
    ftle05=data['ftle05'][:]
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
    

ylim = [-1.6,1.6]
fig = plt.figure(1,figsize=FigSize)
gs1 = gridspec.GridSpec(2, 2)
gs1.update(wspace=0.05, hspace=0.05)
sub1 = plt.subplot(gs1[0])
ax3=plt.plot(tw,rhodot,color='C3',label="$True \\dot{\\rho}$")
ax1=plt.plot(tw,h2rhodot,color='k',label="2km flight")
ax2=plt.plot(tw,p2rhodot,color='C0',label="2km circle")
plt.tick_params(labelbottom='off')
plt.ylabel('hr$^{-1}$',**labelfont)
plt.autoscale(enable=True, axis='x', tight=True)
plt.yticks(**tickfont)
plt.xticks([])#**tickfont)
plt.ylim(ylim)
plt.annotate('2km', xy=(0.86, 0.03), xycoords='axes fraction')


sub2 = plt.subplot(gs1[1])
ax3=plt.plot(tw,rhodot,color='C3',label="True $\\dot{\\rho}$")
ax1=plt.plot(tw,h5rhodot,color='k',label="flight")
ax2=plt.plot(tw,p5rhodot,color='C0',label="circle")
plt.tick_params(labelbottom='off')
plt.tick_params(labelleft='off')
plt.legend(prop = font)
plt.autoscale(enable=True, axis='x', tight=True)
plt.yticks([])#**tickfont)
plt.xticks([])#**tickfont)
plt.ylim(ylim)
plt.annotate('5km', xy=(0.86, 0.03), xycoords='axes fraction')
sub2.legend(prop=font, bbox_to_anchor=(0.575, 1.18), shadow=False, ncol=3)

sub3 = plt.subplot(gs1[2])
ax3=plt.plot(tw,rhodot,color='C3',label="$True \\dot{\\rho}$")
ax1=plt.plot(tw,h10rhodot,color='k',label="10km flight")
ax2=plt.plot(tw,p10rhodot,color='C0',label="10km circle")
plt.ylabel('hr$^{-1}$',**labelfont)
plt.autoscale(enable=True, axis='x', tight=True)
plt.yticks(**tickfont)
plt.xticks(**tickfont)
plt.ylim(ylim)
plt.xlabel('Hours')
plt.annotate('10km', xy=(0.82, 0.03), xycoords='axes fraction')


sub4 = plt.subplot(gs1[3])
ax3=plt.plot(tw,rhodot,color='C3',label="$True \\dot{\\rho}$")
ax1=plt.plot(tw,h15rhodot,color='k',label="15km flight")
ax2=plt.plot(tw,p15rhodot,color='C0',label="15km circle")
plt.tick_params(labelleft='off')
plt.autoscale(enable=True, axis='x', tight=True)
plt.yticks([])#**tickfont)
plt.xticks(**tickfont)
plt.ylim(ylim)
plt.annotate('15km', xy=(0.82, 0.03), xycoords='axes fraction')

plt.xlabel('Hours')
plt.savefig('rhodot_idealized_vs_simulated.eps', transparent=True, bbox_inches='tight',pad_inches=0)
plt.savefig('rhodot_idealized_vs_simulated.png', transparent=True, bbox_inches='tight',pad_inches=0)

ylim=[-2.1,1.25]
fig = plt.figure(2,figsize=FigSize)
gs1 = gridspec.GridSpec(2, 2)
gs1.update(wspace=0.05, hspace=0.05)
sub1 = plt.subplot(gs1[0])
ax3=plt.plot(tw,s1,color='C3',label="True s$_{1}$")
ax1=plt.plot(tw,h2s1,color='k',label="2km flight")
ax2=plt.plot(tw,p2s1,color='C0',label="2km circle")
plt.tick_params(labelbottom='off')
plt.ylabel('hr$^{-1}$',**labelfont)
plt.autoscale(enable=True, axis='x', tight=True)
plt.yticks(**tickfont)
plt.xticks([])#**tickfont)
plt.ylim(ylim)
plt.annotate('2km', xy=(0.86, 0.03), xycoords='axes fraction')
sub2 = plt.subplot(gs1[1])
ax3=plt.plot(tw,s1,color='C3',label="True s$_{1}$")
ax1=plt.plot(tw,h5s1,color='k',label="flight")
ax2=plt.plot(tw,p5s1,color='C0',label="circle")
plt.tick_params(labelbottom='off')
plt.tick_params(labelleft='off')
plt.autoscale(enable=True, axis='x', tight=True)
plt.yticks([])#**tickfont)
plt.xticks([])#**tickfont)
plt.ylim(ylim)
plt.annotate('5km', xy=(0.86, 0.03), xycoords='axes fraction')
sub2.legend(prop=font, bbox_to_anchor=(0.575, 1.18), shadow=False, ncol=3)

sub3 = plt.subplot(gs1[2])
ax3=plt.plot(tw,s1,color='C3',label="True s$_{1}$")
ax1=plt.plot(tw,h10s1,color='k',label="10km flight")
ax2=plt.plot(tw,p10s1,color='C0',label="10km circle")
plt.ylabel('hr$^{-1}$',**labelfont)
plt.autoscale(enable=True, axis='x', tight=True)
plt.yticks(**tickfont)
plt.xticks(**tickfont)
plt.ylim(ylim)
plt.xlabel('Hours')
plt.annotate('10km', xy=(0.82, 0.03), xycoords='axes fraction')

sub4 = plt.subplot(gs1[3])
ax3=plt.plot(tw,s1,color='C3',label="True s$_{1}$")
ax1=plt.plot(tw,h15s1,color='k',label="15km flight")
ax2=plt.plot(tw,p15s1,color='C0',label="15km circle")
#sub4.annotate('A', xy=get_axis_limits(sub4,0.85))
plt.tick_params(labelleft='off')
plt.autoscale(enable=True, axis='x', tight=True)
plt.yticks([])#**tickfont)
plt.xticks(**tickfont)
plt.xlabel('Hours')
plt.ylim(ylim)
plt.annotate('15km', xy=(0.82, 0.03), xycoords='axes fraction')

plt.savefig('s1_idealized_vs_simulated.eps', transparent=True, bbox_inches='tight',pad_inches=0)
plt.savefig('s1_idealized_vs_simulated.png', transparent=True, bbox_inches='tight',pad_inches=0)

ylim = [-2.1,1.25]
fig = plt.figure(3,figsize=FigSize2)
plt.axhline(0,color='k')
ax1=plt.plot(tw,rhodot,color='purple',label="$\\dot{\\rho}$")
axf05=plt.plot(tw,-ftle05,color='g',label="$\\sigma$, T=0.5hr")
axf1=plt.plot(tw,-ftle1,color='b',label="$\\sigma$, T=1hr")
axf4=plt.plot(tw,-ftle2,color='c',label="$\\sigma$, T=2hr")
plt.autoscale(enable=True, axis='x', tight=True)
plt.legend(prop = font)
plt.yticks(**tickfont)
plt.xticks(**tickfont)
plt.ylabel('hr$^{-1}$',**labelfont)
plt.xlabel('Hours')
plt.ylim(ylim)
plt.savefig('rhodot_vs_FTLE.eps', transparent=True, bbox_inches='tight',pad_inches=0)
plt.savefig('rhodot_vs_FTLE.png', transparent=True, bbox_inches='tight',pad_inches=0)

ylim = [-2.1,1.25]
fig = plt.figure(4,figsize=FigSize2)
plt.axhline(0,color='k')
ax2=plt.plot(tw,s1,color='r',label="$s_{1}$")
axf05=plt.plot(tw,-ftle05,color='g',label="$\\sigma$, T=0.5hr")
axf1=plt.plot(tw,-ftle1,color='b',label="$\\sigma$, T=1hr")
axf4=plt.plot(tw,-ftle2,color='c',label="$\\sigma$, T=2hr")
plt.autoscale(enable=True, axis='x', tight=True)
plt.legend(prop = font)
plt.yticks(**tickfont)
plt.xticks(**tickfont)
plt.ylabel('hr$^{-1}$',**labelfont)
plt.xlabel('Hours')
plt.ylim([-2.1,1.25])
plt.savefig('s1_FTLE.eps', transparent=True, bbox_inches='tight',pad_inches=0)
plt.savefig('s1_FTLE.png', transparent=True, bbox_inches='tight',pad_inches=0)

#'''