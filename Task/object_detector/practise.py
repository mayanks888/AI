import datetime
import time
import numpy as np
ts = time.time()
st = datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d %H:%M:%S')
st = datetime.datetime.fromtimestamp(ts).strftime('%d_%m_%Y_%H_%M_%S')
print (st)

import h5py

f = h5py.File('foo.hdf5', 'w')
uint8_dt = h5py.special_dtype(vlen=np.dtype('uint8'))
d = f.create_dataset('data', (1, 1), maxshape=(None,1 ), dtype=uint8_dt)#, chunks=True)

d.resize((6,1 ))
h5py.__version__
# Out[5]: '2.2.1'