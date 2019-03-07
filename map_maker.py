from mpl_toolkits.basemap import Basemap
import matplotlib.pyplot as plt
# setup Lambert Conformal basemap.
# set resolution=None to skip processing of boundary datasets.
plt.close('all')
width = 5+3/8
plt.figure(1,[width,0.75*width])
m = Basemap(width=1200000,height=900000,projection='lcc',
            resolution='i',area_thresh=1000.0,lat_1=30.,lat_2=60,lat_0=37.189722,lon_0=-80.576389)
#m.bluemarble()
#m.etopo()
m.shadedrelief()
m.drawstates(color='grey')
m.drawcoastlines(color='grey')
x,y=m(-80.576389,37.189722)
plt.annotate('Kentland Farm', xy=(x, y),  xycoords='data',
                xytext=(-40, 10), textcoords='offset points',
                color='black',
                #arrowprops=dict(arrowstyle="fancy", color='g')
                )
m.scatter(-80.576389,37.189722,color='red',latlon=True)
plt.savefig('map.eps', transparent=True, bbox_inches='tight')
