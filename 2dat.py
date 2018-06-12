import h5py as hp

#Domain Lat Lon
Lat = 37.208
Lon =-80.5803
gridspace = 3000
with hp.File('850mb_NAMvel_10min_3km.hdf5','r') as loadfile:
    u = loadfile['u'][:]
    v = loadfile['v'][:]
    loadfile.close()
dim = u.shape

for t in range(dim[0]):
    f = open('roms%04d.dat' % t, 'w')
    f.write("Surface Velocity ROMS data (m/s)\n")	
    #f.write("Domain Center 36.7964N,-120.822E\n")
    f.write("Domain Center "+str(Lat)+"N,"+str(Lon)+"E\n")
    f.write("#Data_XMin = "+str(-1*(dim[2]-1)/2*gridspace)+"\n")
    f.write("#Data_XMax = "+str((dim[2]-1)/2*gridspace)+"\n")
    f.write("#Data_XRes = "+str(dim[2])+"\n")
    f.write("#Data_YMin = "+str(-1*(dim[1]-1)/2*gridspace)+"\n")
    f.write("#Data_YMax = "+str((dim[1]-1)/2*gridspace)+"\n")
    f.write("#Data_YRes = "+str(dim[1])+"\n")
    f.write("ZONE T=\"%04d\" I=" % (t+1) +str(dim[1])+" J="+str(dim[2])+"\n")
    for i in range(dim[1]):
        for j in range(dim[2]):
            f.write(str(u[t,i,j])+" "+str(v[t,i,j])+"\n")
    f.close()