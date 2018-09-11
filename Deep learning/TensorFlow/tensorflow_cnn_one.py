import tensorflow as tf
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data
mnist=input_data.read_data_sets('MNIST_data/',one_hot=1)


def init_weight(shape):
    init_random_di=tf.truncated_normal(shape,stddev=0.1)
    return tf.Variable(init_random_di)


def init_bias(shape):
    init_random_bias=tf.constant(0.1,shape=shape)
    return init_random_bias

def conv2d(x,W):
    #x=input data[batch H,W, (RGB]#dimensionals
    #W=filter H, filter Wm channel In, channel out
    return( tf.nn.conv2d(x,W,strides=[1,1,1,1],padding='SAME'))

def max_pool_2x2(x):
    ##x=input data{batch H,W, (RGB))
    return (tf.nn.pool(x,window_shape=[1,2,2,1],strides=[1,2,2,1],padding='SAME'))

#convolution layer
def convolution_layer(input_x,shape):
    W=init_weight(shape)
    b=init_bias(shape[3])
    return tf.nn.relu(conv2d(input_x,W)+b)

#normal (fully connected)
'''def normal_full_layer(input_layer,size):
    input_size=int(input_layer.get_shape()[1])
    W=init_weight([input_size,size])
    b=init_bias([size])
    return tf.matmul(W,input_layer)+b'''


#placeholder
x=tf.placeholder(tf.float32,shape=[None,784])
y_true=tf.placeholder(tf.float32,shape=[None,10])
x_image=tf.reshape(x,[-1,28,28,1])#[batch size,height,width , channel)
convo1=convolution_layer(x_image,shape=[5,5,1,32])
convo_1_pooling=max_pool_2x2(convo1)


'''convo2=convolution_layer(convo_1_pooling,shape=[5,5,32,64])
convo_2_pooling=max_pool_2x2(convo2)

convo_2_flat=tf.reshape(convo_2_pooling,[-1,7*7*64])
full_layer=tf.nn.relu(normal_full_layer(convo_2_flat,1024))

#dropout
hold_prob=tf.placeholder(tf.float32)
full_one_dropout=tf.nn.dropout(full_layer,keep_prob=hold_prob)

y_pred=normal_full_layer(full_one_dropout,10)'''
