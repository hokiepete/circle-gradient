import numpy as np
import h5py as hp
import pandas as pd
#import matplotlib.pyplot as plt
#from numpy import genfromtxt
print("Begin")
'''
xlen = 259
ylen = 257
tlen = 1291
xmin =-38850
xmax = 38850
ymin =-38400 
ymax = 38400 
t0=0
tf=215
'''
xlen = 130
ylen = 129
zlen = 29
tlen = 1291
t0 = 0
tf = 215
xmin =-19350
xmax = 19350
ymin =-19200 
ymax = 19200 
zmin = 0
zmax = 5600
#'''
dimension = 3
data = 'vel'
job = 'verybig' #'big', ''
#job = ''
if data == 'vel' and dimension == 3 and job =='verybig':
    import sys
    savefile = hp.File('NAM_Velocity_t='+str(t0)+'-'+str(tf)+'hrs_Sept2017_300m_15min_Res.hdf5','w')
    mydata = pd.read_csv('myfile_u.csv', delimiter=',',usecols=[0],names=['uvar'])
    print("Data is in")
    uvar = mydata['uvar']
    del mydata
    u = np.empty([tlen,xlen,ylen,zlen])
    print('ReArrange')
    sys.stdout.flush()
    index = 0
    for t in range(tlen):
        print(t)
        for z in range(zlen):
            for y in range(ylen):
                for x in range(xlen):
                    u[t,x,y,z] = uvar[index]
                    index+=1
    
    del uvar
    savefile.create_dataset('u',shape=u.shape,data=u)        
    del u

    mydata = pd.read_csv('myfile_v.csv', delimiter=',',usecols=[0],names=['vvar'])
    print("Data is in")
    vvar = mydata['vvar']
    del mydata
    v = np.empty([tlen,xlen,ylen,zlen])
    print('ReArrange')
    sys.stdout.flush()
    index = 0
    for t in range(tlen):
        print(t)
        for z in range(zlen):
            for y in range(ylen):
                for x in range(xlen):
                    v[t,x,y,z] = vvar[index]
                    index+=1
    
    del vvar
    savefile.create_dataset('v',shape=v.shape,data=v)        
    del v
    
    mydata = pd.read_csv('myfile_w.csv', delimiter=',',usecols=[0],names=['wvar'])
    print("Data is in")
    wvar = mydata['wvar']
    del mydata
    w = np.empty([tlen,xlen,ylen,zlen])
    print('ReArrange')
    sys.stdout.flush()
    index = 0
    for t in range(tlen):
        print(t)
        for z in range(zlen):
            for y in range(ylen):
                for x in range(xlen):
                    w[t,x,y,z] = wvar[index]
                    index+=1
    
    del wvar
    savefile.create_dataset('w',shape=w.shape,data=w)        
    del w
    
    savefile.close()
    xx = np.linspace(xmin,xmax,xlen)
    yy = np.linspace(ymin,ymax,ylen)
    zz = np.linspace(zmin,zmax,zlen)
    y, x, z = np.meshgrid(yy,xx,zz)
    t = np.linspace(t0,tf,tlen)
    print(x.shape)
    print(t)
    print('Save')
    with hp.File('NAM_gridpoints.hdf5','w') as savefile:
        savefile.create_dataset('t',shape=t.shape,data=t)
        savefile.create_dataset('x',shape=x.shape,data=x)
        savefile.create_dataset('y',shape=y.shape,data=y)
        savefile.create_dataset('z',shape=z.shape,data=z)
        savefile.close()
        
elif data == 'vel' and dimension == 3 and job =='big':
    savefile = hp.File('NAM_Velocity_t='+str(t0)+'-'+str(tf)+'hrs_Sept2017_300m_15min_Res.hdf5','w')
    mydata = pd.read_csv('myfile.csv', delimiter=',',names=['uvar','vvar','wvar','na'])
    print("Data is in")
    uvar = mydata['uvar']
    del mydata
    u = np.empty([tlen,xlen,ylen,zlen])
    print('ReArrange')
    index = 0
    for t in range(tlen):
        print(t)
        for z in range(zlen):
            for y in range(ylen):
                for x in range(xlen):
                    u[t,x,y,z] = uvar[index]
                    index+=1
    
    del uvar
    savefile.create_dataset('u',shape=u.shape,data=u)        
    del u

    mydata = pd.read_csv('myfile.csv', delimiter=',',names=['uvar','vvar','wvar','na'])
    print("Data is in")
    vvar = mydata['vvar']
    del mydata
    v = np.empty([tlen,xlen,ylen,zlen])
    print('ReArrange')
    index = 0
    for t in range(tlen):
        print(t)
        for z in range(zlen):
            for y in range(ylen):
                for x in range(xlen):
                    v[t,x,y,z] = vvar[index]
                    index+=1
    
    del vvar
    savefile.create_dataset('v',shape=v.shape,data=v)        
    del v
    
    mydata = pd.read_csv('myfile.csv', delimiter=',',names=['uvar','vvar','wvar','na'])
    print("Data is in")
    wvar = mydata['wvar']
    del mydata
    w = np.empty([tlen,xlen,ylen,zlen])
    print('ReArrange')
    index = 0
    for t in range(tlen):
        print(t)
        for z in range(zlen):
            for y in range(ylen):
                for x in range(xlen):
                    w[t,x,y,z] = wvar[index]
                    index+=1
    
    del wvar
    savefile.create_dataset('w',shape=w.shape,data=w)        
    del w
    
    savefile.close()
    xx = np.linspace(xmin,xmax,xlen)
    yy = np.linspace(ymin,ymax,ylen)
    zz = np.linspace(zmin,zmax,zlen)
    y, x, z = np.meshgrid(yy,xx,zz)
    t = np.linspace(t0,tf,tlen)
    print(x.shape)
    print(t)
    print('Save')
    with hp.File('NAM_gridpoints.hdf5','w') as savefile:
        savefile.create_dataset('t',shape=t.shape,data=t)
        savefile.create_dataset('x',shape=x.shape,data=x)
        savefile.create_dataset('y',shape=y.shape,data=y)
        savefile.create_dataset('z',shape=z.shape,data=z)
        savefile.close()
    
elif data == 'vel' and dimension == 3:
    mydata = pd.read_csv('myfile.csv', delimiter=',',names=['uvar','vvar','wvar','na'])
    print("Data is in")
    uvar = mydata['uvar']
    vvar = mydata['vvar']
    wvar = mydata['wvar']
    del mydata
    u = np.empty([tlen,xlen,ylen,zlen])
    v = np.empty([tlen,xlen,ylen,zlen])
    w = np.empty([tlen,xlen,ylen,zlen])
    print('ReArrange')
    index = 0
    for t in range(tlen):
        print(t)
        for z in range(zlen):
            for y in range(ylen):
                for x in range(xlen):
                    u[t,x,y,z] = uvar[index]
                    v[t,x,y,z] = vvar[index]
                    w[t,x,y,z] = wvar[index]
                    index+=1
    
    del uvar, vvar, wvar
    with hp.File('NAM_Velocity_t='+str(t0)+'-'+str(tf)+'hrs_Sept2017_300m_15min_Res.hdf5','w') as savefile:
        savefile.create_dataset('u',shape=u.shape,data=u)
        savefile.create_dataset('v',shape=v.shape,data=v)
        savefile.create_dataset('w',shape=w.shape,data=w)
        savefile.close()
    del u, v, w


    xx = np.linspace(xmin,xmax,xlen)
    yy = np.linspace(ymin,ymax,ylen)
    zz = np.linspace(zmin,zmax,zlen)
    y, x, z = np.meshgrid(yy,xx,zz)
    t = np.linspace(t0,tf,tlen)
    print(x.shape)
    print(t)
    print('Save')
    with hp.File('NAM_gridpoints.hdf5','w') as savefile:
        savefile.create_dataset('t',shape=t.shape,data=t)
        savefile.create_dataset('x',shape=x.shape,data=x)
        savefile.create_dataset('y',shape=y.shape,data=y)
        savefile.create_dataset('z',shape=z.shape,data=z)
        savefile.close()

if data == 'pressure' and dimension == 3:
    
    mydata = pd.read_csv('myfile.csv', delimiter=',',names=['pvar','na'])
    print("Data is in")
    pvar = mydata['pvar']
    del mydata
    p = np.empty([tlen,xlen,ylen,zlen])
    print('ReArrange')
    index = 0
    for t in range(tlen):
        print(t)
        for z in range(zlen):
            for y in range(ylen):
                for x in range(xlen):
                    p[t,x,y,z] = pvar[index]
                    index+=1
    
    del pvar
    with hp.File('NAM_Pressure_t='+str(t0)+'-'+str(tf)+'hrs_Sept2017_300m_10min_Res.hdf5','w') as savefile:
        savefile.create_dataset('Pressure',shape=p.shape,data=p)
        savefile.close()
    del p
        
elif dimension == 2:
    #mydata = np.genfromtxt('NAMvelocityData.csv', delimiter=',')
    #mydata = pd.read_csv('850mb_300m_10min.csv', delimiter=',',names=['uvar','vvar','na'])
    
    readfile = pd.read_csv('myfile.csv', delimiter=',',names=['uvar','vvar','na'])
    print("Data is in")
    uvar = readfile['uvar']
    vvar = readfile['vvar']
    
    u = np.empty([tlen,ylen,xlen])
    v = np.empty([tlen,ylen,xlen])
    print('ReArrange')
    index = 0
    for t in range(tlen):
        print(t)
        for y in range(ylen):
            for x in range(xlen):
                u[t,y,x] = uvar[index]
                v[t,y,x] = vvar[index]
                index+=1
    
    del uvar, vvar
    with hp.File('850mb_300m_10min.hdf5','w') as savefile:
    #with hp.File('NAM_Pressure_t=46-62hrs_Sept2017.hdf5','w') as savefile:
        #savefile.create_dataset('Pressure',shape=u.shape,data=u)
        savefile.create_dataset('u',shape=u.shape,data=u)
        savefile.create_dataset('v',shape=v.shape,data=v)
        savefile.close()
    del u, v

    xx = np.linspace(-38700,38700,xlen)
    yy = np.linspace(-38400,38400,ylen)
    x, y = np.meshgrid(xx,yy)
    t = np.linspace(t0,tf,tlen)
    print(x.shape)
    print('Save')
    with hp.File('850mb_NAM_gridpoints.hdf5','w') as savefile:
        savefile.create_dataset('t',shape=t.shape,data=t)
        savefile.create_dataset('x',shape=x.shape,data=x)
        savefile.create_dataset('y',shape=y.shape,data=y)
        savefile.close()
 
    
'''
ds=0.3
[DUy, DUx] = np.gradient(u, ds, ds, edge_order=2)
[DVy, DVx] = np.gradient(v, ds, ds, edge_order=2)
J = np.array([[0, 1], [-1, 0]])
A = np.empty(tlen,xlen,ylen)
  
for m in range(len(x1)):
    for n in range(len(x2)):
      Utemp = np.array([U[n, m], V[n, m]])
      
      S = 0.5*(Grad + np.transpose(Grad))
      A[n, m] = np.dot(Utemp, np.dot(np.dot(np.transpose(J), np.dot(S, J)), Utemp))/np.dot(Utemp, Utemp)
            
fig = plt.Figure()
plt.pcolormesh(xx,yy,np.transpose(A[0,:,:]))
'''