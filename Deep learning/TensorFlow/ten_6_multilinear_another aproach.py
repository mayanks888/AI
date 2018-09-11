import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

def normalize(X):
    """ Normalizes the array X """
    mean = np.mean(X)
    std = np.std(X)
    X = (X - mean)/std
    return X

def append_bias_reshape(features,labels):
    m = features.shape[0]
    n = features.shape[1]
    x = np.reshape(np.c_[np.ones(m),features],[m,n + 1])
    y = np.reshape(labels,[m,1])
    return x, y

# Data
boston = tf.contrib.learn.datasets.load_dataset('boston')
X_train, Y_train = boston.data, boston.target
X_train = normalize(X_train)
X_train, Y_train = append_bias_reshape(X_train, Y_train)
m = len(X_train)  #Number of training examples
n = 13 + 1   # Number of features + bias

# Placeholder for the Training Data
X = tf.placeholder(tf.float32, name='X', shape=[m,n])
Y = tf.placeholder(tf.float32, name='Y')
# We create TensorFlow variables for weight and bias. This time, weights are initialized with random numbers:
# Variables for coefficients
w = tf.Variable(tf.random_normal([n,1]))
# Define the linear regression model to be used for prediction. Now we need matrix multiplication to do the task:
# The Linear Regression Model
Y_hat = tf.matmul(X, w)
# For better differentiation, we define the loss function:
# Loss function
loss = tf.reduce_mean(tf.square(Y - Y_hat, name='loss'))
# Choose the right optimizer:
# Gradient Descent with learning rate of 0.01 to minimize loss
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01).minimize(loss)

# Initializing Variables
init_op = tf.global_variables_initializer()
total = []

with tf.Session() as sess:
    # Initialize variables
    sess.run(init_op)
    # writer = tf.summary.FileWriter('graphs', sess.graph)
     # train the model for 100 epcohs
    for i in range(2):
        _, l = sess.run([optimizer, loss], feed_dict={X: X_train, Y: Y_train})
        total.append(l)
        print('Epoch {0}: Loss {1}'.format(i, l))
         # writer.close()
# w_value, b_value = sess.run([w, b])

plt.plot(total)
plt.show()