import h5py as hp
import numpy as np
#now we will create group into hdf5 file

'''hd=hp.File("data1.hdf5",'w')
arr = np.ones((5,2))
grp1=hd.create_group("train")
grp1.create_dataset('bbox',data=arr)#this is for compression the hdf5 data

grp2=hd.create_group("test")
grp2.create_dataset('bbox', shape=(5,2), dtype=np.int64, fillvalue=34)

#creating attributes for group 1
grp1.attrs['class']='dataMatrix'
grp1.attrs['version']='1.1'
hd.close()'''


# Now to access attribuutes

data=hp.File('data1.hdf5','r')
new_val=data.get("train")
at_val=new_val.attrs.keys()
at_item=new_val.attrs.values()

print(list(at_item),list(at_val))#dont forget to convert into list first