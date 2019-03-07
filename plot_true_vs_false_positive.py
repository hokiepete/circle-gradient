# -*- coding: utf-8 -*-
"""
Created on Sun Aug 26 16:46:36 206

@author: pnola
"""

import numpy as np
import h5py as hp
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from mpl_toolkits.basemap import Basemap
from scipy.integrate import trapz
import sys

epsilon = sys.float_info.epsilon
matplotlib.rcParams['text.usetex']=True
matplotlib.rcParams['mathtext.fontset'] = 'cm'
matplotlib.rcParams['lines.linewidth']=1
matplotlib.rcParams['lines.markersize']=2
plt.rc('font', **{'family': 'serif', 'serif': ['cmr10']})
titlefont = {'fontsize':12}
labelfont = {'fontsize':10}
tickfont = {'fontsize':8}
time_step_offset = 24

with hp.File('850mb_300m_10min_NAM_Rhodot_Origin_t=0-215hrs_Sept2017.hdf5','r') as data:
#with hp.File('hunterdata_r=02km_interpolated_2_cridges.hdf5','r') as data:
    rhodot = data['rhodot'][:].squeeze()
    s1 = data['s1'][:].squeeze()
    time = data['t'][:].squeeze()
    data.close()

#for radius in [400,800,1200,1600,2000,3000,5000,7500,10000]:#[200,500,800,1000,2000,3500,5000,7500,10000]:#[200,300,400]:#np.append(np.linspace(100,10000,37),np.array([1,10,100,500,1000,5000,10000,15000])):

radius=5000#int(radius)
#for percent in np.linspace(100,0,101):
s1_percent = 85
print(radius,s1_percent)

passing_times = np.load('passing_files/passing_times_{0:03d}th_percentile_radius={1:05d}_int=-1.npy'.format(90,radius))+time_step_offset          
#passing_times = np.load('passing_files/passing_times_{0:03d}th_percentile_radius={1:05d}.npy'.format(percent,radius))+24            
#thresh_rhodot = -np.percentile(-rhodot[rhodot<0],percent)
#thresh_s1 = -np.percentile(-s1[s1<0],percent)
thresh_s1 = -np.percentile(-s1,s1_percent)

s1_true_positive = []
s1_false_positive = []
s1_true_negative = []
s1_false_negative = []
#for tt in range(1267):
for tt in range(24,1291):#1267):
    if len([x for x in passing_times if x == tt])!=0:
        if s1[tt]<thresh_s1:    
            s1_true_positive.append(tt)
        else:
            s1_false_negative.append(tt)
    else:
        if s1[tt]<thresh_s1:    
            s1_false_positive.append(tt)
        else:
            s1_true_negative.append(tt)


integration = '-1'
with hp.File('850mb_300m_10min_NAM_FTLE_Origin_t=4-215hrs_Sept2017_int='+integration+'.hdf5','r') as data:
    ftle1 = data['ftle'][:].squeeze()
    data.close()

with hp.File('850mb_300m_10min_NAM_LCS_t=4-215hrs_Sept2017_int='+integration+'.hdf5','r') as loadfile:
    ftle=loadfile['ftle'][:]
    dirdiv=loadfile['directionalderivative'][:]
    concav=loadfile['concavity'][:]
    loadfile.close()

dirdiv= np.ma.masked_where(concav>0,dirdiv)
thresh=np.percentile(ftle1[ftle1>0],90,axis=None)
dirdiv_plot = np.ma.masked_where(ftle<=thresh,dirdiv) 
xlen = 259
ylen = 257
tlen = 1267
integration = '-1'
dim = [tlen,ylen,xlen]

with hp.File('850mb_300m_10min_NAM_Rhodot_t=0-215hrs_Sept2017.hdf5','r') as data:
    s1 = data['s1'][:]#[time_step_offset:,:,:]
    #ftle1 = data['ftle'][:].squeeze()
    data.close()
    
smax = s1.max()
smin = s1.min()
s1= np.ma.masked_where(s1>thresh_s1,s1)


star = [37.19838, -80.57834]
origin = [37.208, -80.5803]
print("Begin Map")
plt.close('all')

fig,ax = plt.subplots(figsize=(6,3))

m = Basemap(width=77700,height=76800,\
    rsphere=(6378137.00,6356752.3142),\
    resolution='c',area_thresh=0.,projection='lcc',\
    lat_1=35.,lat_0=origin[0],lon_0=origin[1])#,ax=ax)


x = np.linspace(0, m.urcrnrx, dim[2])
y = np.linspace(0, m.urcrnry, dim[1])
xx, yy = np.meshgrid(x, y)

ax = plt.subplot(121)
t=261
print(time[t])
m.pcolormesh(xx,yy,s1[t,:,:])#,cmap='jet')#'Wistia')
m.contour(xx,yy,dirdiv_plot[t-24,:,:],levels =[0],linewidths=2,cmap='winter')
x, y = m(star[1],star[0])
circle1 = plt.Circle((x, y), radius, color='darkgray',fill=False,linewidth=1,linestyle='--')
ax.add_patch(circle1)

ax = plt.subplot(122)
t=124
print(time[t])
m.pcolormesh(xx,yy,s1[t,:,:])#,cmap='jet')#'Wistia')#,shading='gouraud')
#plt.colorbar()
m.contour(xx,yy,dirdiv_plot[t-24,:,:],levels =[0],linewidths=2,cmap='winter')
x, y = m(star[1],star[0])
circle1 = plt.Circle((x, y), radius, color='darkgray',fill=False,linewidth=1,linestyle='--')
ax.add_patch(circle1)

plt.show()

plt.savefig('true_vs_false_positive.png', transparent=True, bbox_inches='tight')






