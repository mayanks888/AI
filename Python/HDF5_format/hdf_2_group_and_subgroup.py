import h5py as hp
import numpy as np

data=hp.File("data.hdf5",'r')

val=data.items()

np1=list(val)#need to convert it into list so that you can vies data inside item
print(val)
print(np1)
#this is for reading the datasets from group train
input_val=data.get('train')
#read value inside bbox
find_dat=input_val.get('bbox')
#need to convert the value into numpy arrat so that you can view it
print(np.array(find_dat))

#this is for reading the datasets from group test
input_val=data.get('test')
find_dat=input_val.get('bbox')
print(np.array(find_dat))


data.close()
