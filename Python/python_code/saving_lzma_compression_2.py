import lzma
import numpy as np

#################################################3
# comp = lzma.LZMACompressor()
# decomp = lzma.LZMADecompressor()
# # a = comp.compress(b'alpha') + comp.flush()
# a = comp.compress(complete_pc_colors[:,-1].astype(int).tobytes()) + comp.flush()
# print(a)
# with lzma.open("file.xz", "w") as f:
#     f.write(a)
# b = decomp.decompress(a)
# print(b)
# y = np.frombuffer(b, dtype=np.int)


#########################################################3

import numpy as np
from io import BytesIO
import lzma
# file_path='/home/mayank_sati/codebase/python/lidar/mmdetection_ignit/tools/3950bd41f74548429c0f7700ff3d8269.xz'
file_path='/home/mayank_sati/Downloads/v1.0-mini/pointpainting/9e6ce5a26025462fbd9bdac2d18c07eb.xz'
bindata=lzma.open(file_path).read()
decomp = lzma.LZMADecompressor()
b=decomp.decompress(bindata)
# b = decomp.decompress(a)
print(b)
y = np.frombuffer(b, dtype=np.int)

nb_classes = 11
one_hot_targets = np.eye(nb_classes)[y]
one_hot_targets[:,0]=0
1
