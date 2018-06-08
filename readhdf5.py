import h5py as hp
with hp.File('NAM_Velocity_t=0-215hrs_Sept2017_300m_15min_Res.hdf5','r') as readfile:
    u = readfile['/u'][0,:,:,:]
    readfile.close()