import numpy as np
import tensorflow as tf

feature=2
neuron=3
rand_feature_val=np.random.random([1,feature])
x=tf.placeholder(tf.float32,(None,feature))
w=tf.Variable(tf.random_normal([feature,neuron]))
bias=tf.Variable(tf.ones([neuron]))
xw=tf.matmul(x,w)
z=tf.add(xw,bias)
a=tf.sigmoid(z)
init=tf.global_variables_initializer()

sess=tf.Session()
sess.run(init)
layer_out=sess.run(a,feed_dict={x:rand_feature_val})
print (sess.run(w))