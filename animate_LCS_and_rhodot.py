# -*- coding: utf-8 -*-
"""
Created on Sun Jul 22 17:59:06 2018

@author: pnola
"""

#Domain [-300 300]^2 km
#Origin (41.3209371228N, 289.46309961W)
#Projection Lambert
import matplotlib
#matplotlib.use('Agg')
matplotlib.rcParams['text.usetex']=True
matplotlib.rcParams['mathtext.fontset'] = 'cm'

import matplotlib.pyplot as plt
plt.close('all')
plt.rc('font', **{'family': 'serif', 'serif': ['cmr10']})

from mpl_toolkits.basemap import Basemap
from mpl_toolkits.axes_grid1 import make_axes_locatable
import numpy as np

#import matplotlib.animation as animation
import h5py as hp
import scipy.ndimage.filters as filter
from cyanorangecmap import co
initday = 4
inittime = 00-4+4 #Run Initialization(UTC) - 4hrs for EDT Conversion + 4hrs for integration time
#inittime = 12-4+4 #Run Initialization(UTC) - 4hrs for EDT Conversion + 4hrs for integration time
stepsize = 10 #minutes
'''
amaxits = 7
rmaxits = 7
astd = 2
rstd = 2
atol =-0.002
rtol =-0.002
'''
amaxits = 9
rmaxits = 9
astd = 1
rstd = 1
atol =-0.005
rtol =-0.005
parallelres = 7
meridianres = 7
dx = 300
dy = 300
star = [37.19838, -80.57834]
plt.register_cmap(name='CO', data=co())

#xlen = 201
#ylen = 197
xlen = 259
ylen = 257
tlen = 1267
dim = [tlen,ylen,xlen]
fig, ax = plt.subplots(figsize=(16, 12), dpi=100, facecolor='w', edgecolor='k')
origin = [37.208, -80.5803]

#plt.rcParams.update({'axes.titlesize': 'large'})

print("Begin Map")
''
m = Basemap(width=77700,height=76800,\
    rsphere=(6378137.00,6356752.3142),\
    resolution='f',area_thresh=0.,projection='lcc',\
    lat_1=35.,lat_0=origin[0],lon_0=origin[1])#,ax=ax)

#m.drawcoastlines(linewidth=2.0)
#m.drawcountries()
m.drawparallels(np.linspace(m.llcrnrlat,m.urcrnrlat,parallelres),labels=[True,False,False,False])
m.drawmeridians(np.linspace(m.llcrnrlon,m.urcrnrlon,meridianres),labels=[False,False,False,True])
m.drawstates()
m.drawrivers()

with hp.File('850mb_300m_10min_NAM_Rhodot_t=0-215hrs_Sept2017.hdf5','r') as loadfile:
    rhodot=3600*loadfile['rhodot'][24:,:,:]
    loadfile.close()

with hp.File('850mb_300m_10min_NAM_LCS_t=4-215hrs_Sept2017_int=-1.hdf5','r') as loadfile:
    ftle=loadfile['ftle'][:]
    dirdiv=loadfile['directionalderivative'][:]
    concav=loadfile['concavity'][:]
    loadfile.close()

colormin = rhodot.min(axis=None)
colormax = rhodot.max(axis=None)
colorlevel = 1/3.0*np.min(np.fabs([colormin,colormax]))

#thresh=np.percentile(ftle,95,axis=None)
#thresh = 0.3000400059223705/3600 #90th percentile for the -4hr FTLE timeseries, see plot_time_series.py
thresh = 0
#thresh = 0.5571152601155729/3600 #90th percentile for the -1hr FTLE timeseries, see plot_time_series.py
#thresh = 0.1742656513095688/3600 #50th percentile for the -1hr FTLE timeseries, see plot_time_series.py
dirdiv = np.ma.masked_where(concav>0,dirdiv)
dirdiv = np.ma.masked_where(ftle<=thresh,dirdiv)

#colorlevel = 2
x = np.linspace(0, m.urcrnrx, dim[2])
y = np.linspace(0, m.urcrnry, dim[1])
xx, yy = np.meshgrid(x, y)
'''
repulsion = m.contourf(xx, yy, rhodot[t,:,:],vmin=-colorlevel,\
        vmax=colorlevel,levels=np.linspace(colormin,colormax,301),cmap='CO')
'''
repulsion = m.pcolormesh(xx, yy, rhodot[0,:,:],vmin=-colorlevel,\
        vmax=colorlevel,shading='gouraud',cmap='CO')
clb = plt.colorbar(repulsion,fraction=0.037, pad=0.02) #shrink=0.5,pad=0.2,aspect=10)
clb.ax.set_title('$\dot{\\rho}$',size = 18)


ridge = m.contour(xx,yy,dirdiv[0,:,:],levels =[0])#,latlon=True)
#m.scatter(star[1],star[0],marker='*',s=20*20,latlon=True)


print("Begin Loop")
for t in range(dim[0]):
    for c in ridge.collections:
        #if ind == 0
         #   print c.get_segments()
         #   ind += 1
         c.remove()
    repulsion.set_array(np.ravel(rhodot[t,:,:]))
    ridge = m.contour(xx,yy,dirdiv[t,:,:],levels =[0])#,latlon=True)
    #velquiver.set_UVC(u[t,::30,::30],v[t,::30,::30])
    #qk = plt.quiverkey(velquiver, 0, 0, 12000, r'$2 \frac{m}{s}$', labelpos='E', coordinates='figure')
    #plt.quiverkey(velquiver, 0.9, 0.9, 2, r'$2 \frac{m}{s}$', labelpos='N', coordinates='figure')
    minute = stepsize * (t + 24)
    h, minute = divmod(minute,60)
    x, y = m(star[1],star[0])
    m.scatter(x,y,marker='o',color='g',s=20*16)
    plt.annotate('Kentland Farm',xy=(x-0.1*x,y+0.05*y),size=15)
    plt.title("Repulsion Rate, 9-{0}-2017 {1:02d}{2:02d} GMT".format(initday+(inittime+h)//24, (inittime+h)%24, minute),fontsize=18)
    #plt.autoscale(tight=True)
    plt.savefig('Kentland_lcs_{0:04d}.tif'.format(t), transparent=False, bbox_inches='tight')

"""
npp = nridge.collections[0].get_paths()
wpp = wridge.collections[0].get_paths()
nindex = [104,75,31,32,26] #no windage
windex = [78,54,16,17,10,19] # windage = 0.019
nv = np.empty([0,2])
wv = np.empty([0,2])
plt.close('all')
for i in range(len(nindex)):
    if i<3:
        nv = np.concatenate((nv,list(reversed(npp[nindex[i]].vertices))))
    else:
        nv = np.concatenate((nv,npp[nindex[i]].vertices))

for i in range(len(windex)):
    if i<3:
        wv = np.concatenate((wv,list(reversed(wpp[windex[i]].vertices))))
    else:
        wv = np.concatenate((wv,wpp[windex[i]].vertices))
        
nx,ny = m(nv[:,0],nv[:,1],inverse=True)
wx,wy = m(wv[:,0],wv[:,1],inverse=True)
m.plot(nx,ny, latlon=True,color='r')
m.plot(wx,wy, latlon=True,color='b')
m.drawcoastlines()
parallels = np.arange(round(lat_min,1),lat_max+0.1,0.1)
meridians = np.arange(round(lon_max,1),lon_min-0.1,-0.1)
m.drawparallels(parallels,labels=[1,0,0,0],fontsize=10)
m.drawmeridians(meridians,labels=[0,0,0,1],fontsize=10)

ax = plt.gca()
def format_coord(x, y):
    return 'x=%.4f, y=%.4f'%(m(x, y, inverse = True))
ax.format_coord = format_coord
plt.show()
'''
'''
plt.figure(2)
plt.subplot(121)
m.pcolormesh(wlon,wlat,wftle,latlon=True,vmin=0, vmax=wftle.max(),cmap='Blues')#,alpha=0.4)
m.pcolormesh(nlon,nlat,nftle,latlon=True,vmin=0, vmax=nftle.max(),cmap='Reds')#,alpha=0.4)
#lon, lat = np.meshgrid(lon,lat,indexing='ij')
#m.pcolormesh(lon,lat,ftle,latlon=True,shading='gourand')
#m.contourf(lon,lat,ftle,latlon=True,levels=np.linspace(0,ftle.max(),3001))
#plt.colorbar()
m.drawcoastlines()
#m.drawrivers()
#m.drawstates()
parallels = np.arange(round(lat_min,1),lat_max+0.1,0.1)
meridians = np.arange(round(lon_max,1),lon_min-0.1,-0.1)
m.drawparallels(parallels,labels=[1,0,0,0],fontsize=10)
# draw meridians
m.drawmeridians(meridians,labels=[0,0,0,1],fontsize=10)
plt.title("Red: No windage, Blue: windage = 0.019; both fields masked under 3 days^{-1}")

plt.subplot(122)
m.pcolormesh(nlon,nlat,nftle,latlon=True,vmin=0, vmax=nftle.max(),cmap='Reds')#,alpha=0.4)
m.pcolormesh(wlon,wlat,wftle,latlon=True,vmin=0, vmax=wftle.max(),cmap='Blues')#,alpha=0.4)
#lon, lat = np.meshgrid(lon,lat,indexing='ij')
#m.pcolormesh(lon,lat,ftle,latlon=True,shading='gourand')
#m.contourf(lon,lat,ftle,latlon=True,levels=np.linspace(0,ftle.max(),3001))
#plt.colorbar()
m.drawcoastlines()
#m.drawrivers()
#m.drawstates()
parallels = np.arange(round(lat_min,1),lat_max+0.1,0.1)
meridians = np.arange(round(lon_max,1),lon_min-0.1,-0.1)
m.drawparallels(parallels,labels=[1,0,0,0],fontsize=10)
# draw meridians
m.drawmeridians(meridians,labels=[0,0,0,1],fontsize=10)
'''

'''
plt.figure(1)
plt.subplot(111)

m.drawcoastlines()
nridge = m.contour(nlon,nlat,ndirdiv,levels =[0],colors='blue',latlon=True,alpha=0.6)
wridge = m.contour(wlon,wlat,wdirdiv,levels =[0],colors='red',latlon=True,alpha=0.6)
parallels = np.arange(round(lat_min,1),lat_max+0.1,0.1)
meridians = np.arange(round(lon_max,1),lon_min-0.1,-0.1)
m.drawparallels(parallels,labels=[1,0,0,0],fontsize=10)
# draw meridians
m.drawmeridians(meridians,labels=[0,0,0,1],fontsize=10)
ax = plt.gca()
def format_coord(x, y):
    return 'x=%.4f, y=%.4f'%(m(x, y, inverse = True))
ax.format_coord = format_coord
plt.show()
'''

















"""