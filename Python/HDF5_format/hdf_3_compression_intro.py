import h5py as hp
import numpy as np
#now we will create group into hdf5 file

hd=hp.File("data1.hdf5",'w')
arr = np.ones((5,2))
grp1=hd.create_group("train")
grp1.create_dataset('bbox',data=arr,compression='gzip',compression_opts=9)#this is for compression the hdf5 data

grp2=hd.create_group("test")
grp2.create_dataset('bbox', (5,2), dtype=np.int64,fillvalue=34,compression='gzip',compression_opts=9)
hd.close()