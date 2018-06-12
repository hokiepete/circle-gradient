import h5py as hp
with hp.File('850mb_NAMvel_10min_3km.hdf5','r') as readfile:
    u = readfile['/u'][:]
    readfile.close()