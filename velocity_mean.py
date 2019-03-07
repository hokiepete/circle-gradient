import h5py as hp
import numpy as np
#Domain Lat Lon
Lat = 37.208
Lon =-80.5803
gridspace = 3000
with hp.File('850mb_NAMvel_10min_3km.hdf5','r') as loadfile:
    u = loadfile['u'][::6,:,:]
    v = loadfile['v'][::6,:,:]
    loadfile.close()
dim = u.shape

speed = np.sqrt(u**2+v**2)
speed_mean = np.mean(speed,axis=None)


