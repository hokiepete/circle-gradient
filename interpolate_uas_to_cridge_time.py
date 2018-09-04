# -*- coding: utf-8 -*-
"""
Created on Wed Jun 13 15:59:44 2018

@author: pnola
"""
import h5py as hp
import numpy as np
import scipy.interpolate as sint
from functions import circleaverage
    
with hp.File('hunterdata_r=02km.mat','r') as data:
    h2rhodot = 3600*data['rhodot'][:].squeeze()
    h2s1 = 3600*data['s1'][:].squeeze()
    h2t = data['timeout'][:].squeeze()
    h2theta = data['thetaout'][:].squeeze()
    data.close()


h2rhodot, delete = circleaverage(h2rhodot,h2t,h2theta)
h2s1, h2t = circleaverage(h2s1,h2t,h2theta)
    
tmin = np.max([h2t.min()])
tmax = np.min([h2t.max()])
tw = h2t[h2t>=tmin]
tw = tw[tw<=tmax]
tcku = sint.splrep(h2t, h2rhodot, s=0)
h2rhodot = sint.splev(tw, tcku, der=0)

tcku = sint.splrep(h2t, h2s1, s=0)
h2s1 = sint.splev(tw, tcku, der=0)

with hp.File('PlottingData.hdf5','w') as savefile:
    savefile.create_dataset('tw',shape=tw.shape,data=tw)
    savefile.create_dataset('h2rhodot',shape=h2rhodot.shape,data=h2rhodot)
    savefile.create_dataset('h2s1',shape=h2s1.shape,data=h2s1)
    savefile.close()
