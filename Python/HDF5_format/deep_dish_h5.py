import deepdish as dd
import numpy as np
import time
d = {
    'foo': np.ones((10, 20)),
    'sub': {
        'bar': 'a string',
        'baz': 1.23,
    },
}
dd.io.save('test.h5', d)
t=time.time()
range_val=5
for _ in range(range_val):
    d = dd.io.load('/home/mayank_sati/codebase/python/lidar/Centerpoint/CenterPoint_tyanwey_old_only_preprocessor/tools/test.h5')
print("total_time",((time.time() - t)*1000)/range_val)