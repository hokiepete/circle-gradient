from mpl_toolkits.basemap import Basemap
import matplotlib.pyplot as plt
# setup Lambert Conformal basemap.
# set resolution=None to skip processing of boundary datasets.
plt.close('all')
plt.figure(1,[6,4.5])
m = Basemap(width=1200000,height=900000,projection='lcc',
            resolution='i',area_thresh=1000.0,lat_1=30.,lat_2=60,lat_0=37.189722,lon_0=-80.576389)
m.bluemarble()
m.drawstates(color='black')
m.drawcoastlines(color='black')
x,y=m(-80.576389,37.189722)
plt.annotate('Kentland Farm', xy=(x, y),  xycoords='data',
                xytext=(-40, 10), textcoords='offset points',
                color='white',
                #arrowprops=dict(arrowstyle="fancy", color='g')
                )
m.scatter(-80.576389,37.189722,color='red',latlon=True)
plt.savefig('map.eps', transparent=True, bbox_inches='tight')
