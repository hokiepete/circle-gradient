# -*- coding: utf-8 -*-
"""
Created on Tue Feb 20 10:36:17 2018

@author: pnola
"""
import numpy as np
from scipy.io import savemat
xlen = 259
ylen = 257
tlen = 845
dim = [tlen,xlen,ylen]

mydata = np.genfromtxt('myfile.csv', delimiter=',')
print "Data is in"
# Organize velocity data
uvar = mydata[:,0]
vvar = mydata[:,1]
del mydata
u = np.empty(dim)
v = np.empty(dim)
index = 0
for t in range(dim[0]):
    for y in range(dim[2]):
        for x in range(dim[1]):
            u[t,x,y] = uvar[index]
            v[t,x,y] = vvar[index]
            index+=1
del uvar, vvar
x = np.linspace(-38700,38700,xlen)
y = np.linspace(-38400,38400,ylen)
t = np.linspace(0,211,tlen)
print t
mdict = {'x': x,'y': y,'time': t,'u': u,'v': v}
savemat('KentlandFarmVelocityData.mat',mdict)