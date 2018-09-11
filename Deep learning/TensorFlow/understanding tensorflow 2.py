import numpy as np
import tensorflow as tf
from tensorflow.python import debug as tf_debug

np.random.seed(101)
tf.set_random_seed(101)

rand_a=np.random.uniform(0,100,(5,5))
rand_b=np.random.uniform(0,100,(5,1))

a=tf.placeholder(tf.float32)
b=tf.placeholder(tf.float32)
mul_op=a*b
add_op=a+b
sess=tf.Session()
sess = tf_debug.TensorBoardDebugWrapperSession(sess, "localhost:7000")
add_result=sess.run(add_op,feed_dict={a:rand_a,b:rand_b})
mul_result=sess.run(mul_op,feed_dict={a:rand_a,b:rand_b})
print (add_result)