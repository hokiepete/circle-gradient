import matplotlib.pyplot as plt
plt.close('all')
from mpl_toolkits.basemap import Basemap
import os.path
import numpy as np
import h5py as hp
star = [37.19838, -80.57834]
xlen = 259
ylen = 257
tlen = 1267

integration = '-0,5'
dim = [tlen,ylen,xlen]
origin = [37.208, -80.5803]
print("Begin Map")

x = np.linspace(0, 77700, dim[2])-77700/2
y = np.linspace(0, 76800, dim[1])-76800/2
tt = 746
with hp.File('850mb_300m_10min_velocity.hdf5','r') as data:
    u = data['u'][tt,:,:].squeeze()
    v = data['v'][tt,:,:].squeeze()
    data.close()

#x+=77700/2
#y+=76800/2
cutoff=12000
xx = x[abs(x)<cutoff]+77700/2
yy = y[abs(y)<cutoff]+76800/2
print(max(xx)-min(xx))
fig,ax = plt.subplots(figsize=(6,6))
m = Basemap(width=max(xx)-min(xx),height=max(yy)-min(yy),\
    rsphere=(6378137.00,6356752.3142),\
    resolution='c',area_thresh=0.,projection='lcc',\
    lat_1=35.,lat_0=origin[0],lon_0=origin[1])#,ax=ax)

#xxc, yyc = np.meshgrid(xx, yy)
xx, yy = np.meshgrid(xx-min(xx), yy-min(yy))
#uu = np.empty(xx.shape)
#vv = np.empty(xx.shape)
uu=[]
vv=[]
index1=0
for i in range(u.shape[0]):
    for j in range(u.shape[1]):
        if abs(x[j])<cutoff and abs(y[i])<cutoff:
            uu.append(u[i,j])
            vv.append(v[i,j])
            
    
uu = np.reshape(uu,xx.shape,order='c')
vv = np.reshape(vv,xx.shape,order='c')

x, y = m(star[1],star[0])
sizing = 7
#for tt in range(1291):
#    print(tt)
m.quiver(xx[::sizing,::sizing],yy[::sizing,::sizing],uu[::sizing,::sizing],vv[::sizing,::sizing],color='lightgrey')
#m.scatter(xx,yy)

circle1 = plt.Circle((x, y), 100, color='black',fill=True)
ax.add_patch(circle1)

circle1 = plt.Circle((x, y), 800, color='black',fill=False)
ax.add_patch(circle1)

circle1 = plt.Circle((x, y), 2000, color='black',fill=False)
ax.add_patch(circle1)

circle1 = plt.Circle((x, y), 7500, color='black',fill=False)
ax.add_patch(circle1)

plt.show()
#plt.title('quiver{0:04d}.png'.format(tt))
plt.savefig('flight_schematic.eps'.format(tt), transparent=False, bbox_inches='tight',pad_inches=0)
#plt.close('all')