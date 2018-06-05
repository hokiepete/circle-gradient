# -*- coding: utf-8 -*-
"""
Created on Tue Apr 17 10:48:20 2018

@author: pnola
"""

import h5py as hp
import numpy as np
import scipy.interpolate as sinp
import matplotlib.pyplot as plt

dim=3

if dim == 2:

    xlen = 43
    ylen = 41
    
    with hp.File('850mb_NAMvel_1hr_3km.hdf5','r') as loadfile:
        uin = loadfile['u'][:]
        vin = loadfile['v'][:]
        loadfile.close()
    
    
    tstep =1.0/6.0
    timein = np.linspace(0,72,73)
    timewant = np.arange(0,72+tstep,tstep)                    
    if timewant[-1]>72:
        timewant = timewant[0:-1]
        
    uwant = np.empty([np.size(timewant),ylen,xlen])
    vwant = np.empty([np.size(timewant),ylen,xlen])
    for i in range(ylen):
        for j in range(xlen):
            tcku = sinp.splrep(timein, uin[:,i,j], s=0)
            uwant[:,i,j] = sinp.splev(timewant, tcku, der=0)
            tckv = sinp.splrep(timein, vin[:,i,j], s=0)
            vwant[:,i,j] = sinp.splev(timewant, tckv, der=0)
            
    with hp.File('850mb_NAMvel_15min_3km.hdf5','w') as savefile:
        savefile.create_dataset('u',shape=uwant.shape,data=uwant)
        savefile.create_dataset('v',shape=vwant.shape,data=vwant)
        savefile.close()
    plt.close('all')  
    plt.plot(timewant,uwant[:,0,0])
    plt.plot(timein,uin[:,0,0])
    
elif dim ==3:
    xlen = 43
    ylen = 41
    zlen = 42
    tlen = 216
    gridspacing = 3000
    with hp.File('3D_NAMvel_1hr_3km.hdf5','r') as loadfile:
        uin = loadfile['u'][:]
        vin = loadfile['v'][:]
        win = loadfile['w'][:]
        pressurein = loadfile['pressure'][:]
        heightin = loadfile['height'][:]
        groundin = loadfile['ground'][:]
        #xin = loadfile['x'][:]
        #yin = loadfile['y'][:]
        loadfile.close()
    
    xin = np.linspace(-(xlen-1)/2*gridspacing,(xlen-1)/2*gridspacing,xlen)
    yin = np.linspace(-(ylen-1)/2*gridspacing,(ylen-1)/2*gridspacing,ylen)
    tstep =1.0/6.0
    timein = np.linspace(0,(tlen-1),tlen)
    timewant = np.arange(0,(tlen-1)+tstep,tstep)                    
    if timewant[-1]>(tlen-1):
        timewant = timewant[0:-1]
        
    uwant = np.empty([np.size(timewant),zlen,ylen,xlen])
    vwant = np.empty([np.size(timewant),zlen,ylen,xlen])
    wwant = np.empty([np.size(timewant),zlen,ylen,xlen])
    heightwant = np.empty([np.size(timewant),zlen,ylen,xlen])
    pressurewant = np.empty([np.size(timewant),zlen,ylen,xlen])
    for k in range(zlen):
        for i in range(ylen):
            for j in range(xlen):
                tcku = sinp.splrep(timein, uin[:,k,i,j], s=0)
                uwant[:,k,i,j] = sinp.splev(timewant, tcku, der=0)
                
                tckv = sinp.splrep(timein, vin[:,k,i,j], s=0)
                vwant[:,k,i,j] = sinp.splev(timewant, tckv, der=0)
                
                tckw = sinp.splrep(timein, win[:,k,i,j], s=0)
                wwant[:,k,i,j] = sinp.splev(timewant, tckw, der=0)
                
                tckheight = sinp.splrep(timein, heightin[:,k,i,j], s=0)
                heightwant[:,k,i,j] = sinp.splev(timewant, tckheight, der=0)
                
                tckpressure = sinp.splrep(timein, pressurein[:,k,i,j], s=0)
                pressurewant[:,k,i,j] = sinp.splev(timewant, tckpressure, der=0)
                
    with hp.File('3D_NAMvel_10min_3km.hdf5','w') as savefile:
        savefile.create_dataset('u',shape=uwant.shape,data=uwant)
        savefile.create_dataset('v',shape=vwant.shape,data=vwant)
        
        savefile.create_dataset('w',shape=wwant.shape,data=wwant)
        savefile.create_dataset('height',shape=heightwant.shape,data=heightwant)
        
        savefile.create_dataset('pressure',shape=pressurewant.shape,data=pressurewant)
        savefile.create_dataset('ground',shape=groundin.shape,data=groundin)
        
        savefile.create_dataset('x',shape=xin.shape,data=xin)
        savefile.create_dataset('y',shape=yin.shape,data=yin)
        
        savefile.close()
        
        
    plt.close('all')  
    plt.plot(timewant,uwant[:,0,0,0])
    plt.plot(timein,uin[:,0,0,0])
    
    
