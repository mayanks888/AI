import h5py
import numpy as np
# # create hdf5 file
# f=h5py.File("name.hdf5",'r')
# print(f.filename)
# f.close()

'''# f=f=h5py.File("name.hdf5",'w')# New file overwriting any existing file
# f=h5py.File("name.hdf5",'r')# Open read-only (must exist)
# f=h5py.File("name.hdf5",'r+')# Open read-write (must exist)
# f=h5py.File("name.hdf5",'a')# Open read-write (create if doesn't exist)

# now Ill write something on name hdf5 file its didnt work as of writing this code,i'll work on it later
f=h5py.File('name.hdf5','w')
with h5py.File("name.hdf5", "w") as f:
    f.write("Hello!")

f=h5py.File("test_data.hdf5",'w')
arr = np.ones((5,2))
f["cool_dataset_name"] = arr
f.create_dataset('newDAtasets',data=arr)
dset = f["cool_dataset_name"]#this is also a h5py format
print(dset[1:4])
print(dset.shape)

dat = f.create_dataset('empty', (5,2), dtype=np.int64,fillvalue=34)#or this method can be used to creae datasets


f.close()

#now let us read hdf5 fil2
hdf=h5py.File("test_data.hdf5",'r')
li=list(hdf.keys())#this is to read the list of datasets inside the hdf5 file
print(li)
get_Data=hdf.get('empty')
print(get_Data[:])#way to access data
#if you want it out save it in numpy array
data_return=np.array(get_Data)#we will convert it baack to np to voewin variable explorer
hdf.close()

#now we will create group into hdf5 file
hd=h5py.File("data.hdf5",'w')
arr = np.ones((5,2))
grp1=hd.create_group("train")
grp1.create_dataset('bbox',data=arr)

grp2=hd.create_group("test")
grp2.create_dataset('bbox', (5,2), dtype=np.int64,fillvalue=34)
hd.close()'''


#now reading groups in hdfc data file created above with groups
hdf=h5py.File("data.hdf5",'r')
gorup_read=hdf.items()
print(list(gorup_read))#very imp to convert it into list other wise will not be able to read it
get_Data=hdf.get('train')
print((get_Data))
bbox_val=get_Data.items()
print(list(bbox_val))
# data_return=np.array((bbox_val.get('bbox')))
# print(data_return)
data_return=get_Data.get('bbox')
print(np.array(data_return))
