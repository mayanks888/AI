import tensorflow as tf

tf.enable_eager_execution()

# tf.executing_eagerly()

import numpy as np


def normalize(X):
    """ Normalizes the array X """
    mean = np.mean(X)
    std = np.std(X)
    X = (X - mean) / std
    return X


def append_bias_reshape(features, labels):
    m = features.shape[0]
    n = features.shape[1]
    x = np.reshape(np.c_[np.ones(m), features], [m, n + 1])
    y = np.reshape(labels, [m, 1])
    return x, y


# Data
boston = tf.contrib.learn.datasets.load_dataset('boston')
X_train, Y_train = boston.data, boston.target
X_train = normalize(X_train)
X_train, Y_train = append_bias_reshape(X_train, Y_train)
m = len(X_train)  # Number of training examples
n = 13 + 1  # Number of features + bias


class Regressor(tf.keras.Model):

    def __init__(self):
        super(Regressor, self).__init__()
        self.dense = tf.keras.layers.Dense(1)

    def call(self, inputs, training=None, mask=None):
        output = self.dense(inputs)
        # output=tf.matmul(x,y)
        return output


model = Regressor()
output = model.call(x_train)
loss = tf.reduce_mean(tf.square(Y - Y_hat, name='loss'))
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01).minimize(loss)
