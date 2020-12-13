

import tensorflow as tf

sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))
import os

print(os.system('nvcc --version'))
tf.test.is_gpu_available(
    cuda_only=False,
    min_cuda_compute_capability=None
)


with tf.device('/gpu:0'):
    a = tf.constant([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], shape=[2, 3], name='a')
    b = tf.constant([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], shape=[3, 2], name='b')
    c = tf.matmul(a, b)

with tf.Session() as sess:
    print (sess.run(c))

from tensorflow.python.client import device_lib

print(device_lib.list_local_devices())
