import lzma

import numpy as np
import lzma
file_path='/home/mayank_sati/codebase/python/lidar/mmdetection_ignit/tools/3950bd41f74548429c0f7700ff3d8269.xz'
bindata=lzma.open(file_path).read()
obj = lzma.LZMADecompressor(format=lzma.FORMAT_XZ)
print('len(bindata) ', len(bindata))
# binstr = b''.join(bindata)
new_data=obj.decompress(bindata)
print('len(new_data) ', len(new_data))
y = np.frombuffer(new_data, dtype=np.float32)
# final_data=np.reshape(y, (1,64, 400,400))
final_data=np.reshape(y, ((-1,4)) )


##########################################################

1