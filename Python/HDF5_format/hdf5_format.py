import h5py
import numpy as np
###########################################################
# h5py.File.attrs
# h5py.File.get
# h5py.File.close
# h5py.File.id
# h5py.File.copy
# h5py.File.items
# h5py.File.create_dataset h5py.File.iteritems
# h5py.File.create_group
# h5py.File.iterkeys
# h5py.File.driver
# h5py.File.itervalues
# h5py.File.fid
# h5py.File.keys
# h5py.File.file
# h5py.File.libver
# h5py.File.filename
# h5py.File.mode
# h5py.File.flush
# h5py.File.mro
# h5py.File.name
# h5py.File.parent
# h5py.File.ref
# h5py.File.require_dataset
# h5py.File.require_group
# h5py.File.userblock_size
# h5py.File.values
# h5py.File.visit
# h5py.File.visititems
temperature = np.random.random(1024)
print (temperature)
# this is for creating hdf5 file or reading it if already created
f = h5py.File("new2.hdf5")
f["temp"] = temperature
f["temp"].attrs["dt"] = 10.0
f["temp"].attrs["start_time"] = 1375204299
print(f)

# Loading the content of datasets in data
data = f["temp"]
# for key, value in data.attrs.iteritems():
#     print ("%s: %s" % (key, value))
a=1
print(data[2:4])
print(data.att)

# big_dataset = f.create_dataset("big", shape=(4, 4),dtype='float32',compression='gzip')
# print(big_dataset)
####################################################################3
# Type 2

dat=np.arange(10,dtype="int32")
print(dat)
print(dat.dtype)
print(h5py.File.attrs)