# -*- coding: utf-8 -*-
"""
Created on Mon Jul 23 12:53:58 2018

@author: pnola
"""
import numpy as np
import h5py as hp

with hp.File('FTLEOutput_int=-4.mat','r') as loadfile:
    ftlein = np.swapaxes(loadfile['F'][:,::3,:],1,2)
    eig1in = np.swapaxes(loadfile['F'][:,1::3,:],1,2)
    eig2in = np.swapaxes(loadfile['F'][:,2::3,:],1,2)
    loadfile.close()
dy=300
dx=300
dim =  ftlein.shape
dirdiv = np.ma.empty(dim)
concav = np.ma.empty(dim)
    
for t in range(dim[0]):
    print(t)
    ftle=ftlein[t,:,:]
    eig1=eig1in[t,:,:]
    eig2=eig2in[t,:,:]
    
    dfdy,dfdx = np.gradient(ftle,dy,dx,edge_order=2)
    dfdydy,dfdydx = np.gradient(dfdy,dy,dx,edge_order=2)
    dfdxdy,dfdxdx = np.gradient(dfdx,dy,dx,edge_order=2)
    
    for i in range(dim[1]):
        for j in range(dim[2]):
            if (dfdx[i,j] and dfdy[i,j] and dfdxdy[i,j] and dfdydy[i,j] and dfdxdx[i,j] and dfdydx[i,j]) is not np.ma.masked:    
                dirdiv[t,i,j] = np.dot([dfdx[i,j],dfdy[i,j]],[eig1[i,j],eig2[i,j]])
                concav[t,i,j] = np.dot(np.dot([[dfdxdx[i,j],dfdxdy[i,j]],[dfdydx[i,j],dfdydy[i,j]]],[eig1[i,j],eig2[i,j]]),[eig1[i,j],eig2[i,j]])
            else:
                dirdiv[t,i,j] = np.ma.masked
                concav[t,i,j] = np.ma.masked


with hp.File('850mb_300m_10min_NAM_LCS_t=4-215hrs_Sept2017_int=-4.hdf5','w') as savefile:
        savefile.create_dataset('ftle',shape=ftlein.shape,data=ftlein)
        savefile.create_dataset('concavity',shape=concav.shape,data=concav)
        savefile.create_dataset('directionalderivative',shape=dirdiv.shape,data=dirdiv)
        savefile.close()