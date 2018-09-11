import tensorflow as tf
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data
mnist=input_data.read_data_sets('MNIST_data/',one_hot=1)
# pd.read_csv('/PycharmProjects/datasets/MNIST_data')
# print(mnist)
# print(mnist.train.num_examples)
my_image=mnist.train.images[5].reshape(28,28)
plt.imshow(my_image,cmap="gist_gray")
# plt.show()

#softmax model approach)
#define variable
neuron=10
feature=784
labels=10
x=tf.placeholder(tf.float32,shape=[None,feature])
#loss function
y_true=tf.placeholder(tf.float32,[None,10])
# w=tf.Variable(tf.random_normal([feature,neuron]))
# bias=tf.Variable(tf.ones([neuron]))
w=tf.Variable(tf.ones([feature,neuron]))

bias=tf.Variable(tf.ones([neuron]))
# trial=tf.Variable(initial_value=3)
trial=tf.Variable(tf.ones([neuron]))
val_default=trial*3
# val_default=tf.multiply(trial,3)

#crete graph function
y=tf.matmul(x,w)+bias



cross_entropy=tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_true,logits=y))

#optimiser
optimiser=tf.train.GradientDescentOptimizer(learning_rate=0.3)
train_data=optimiser.minimize(cross_entropy,trial)
init=tf.global_variables_initializer()

sess=tf.Session()
sess.run(init)
training_steps=100

for i in range(training_steps):
    batch_x,batch_y=mnist.train.next_batch(100)
    my_comp_data=sess.run([y,train_data,],feed_dict={x:batch_x,y_true:batch_y})
    mycross_entropy=sess.run([cross_entropy],feed_dict={x:batch_x,y_true:batch_y})
   # print(sess.run(w))
    print (np.max(sess.run(w)))
    print (np.max(sess.run(bias)))
    print (np.max(sess.run(trial)))
    # print (sess.run(val_default))
    # print(my_comp_data)
   # print(mycross_entropy)

#evaluate my model
my_prediction=tf.equal(tf.argmax(y,1),tf.argmax(y_true,1))
#cast convert true to 1 an false to 0
accuray=tf.reduce_mean(tf.cast(my_prediction,tf.float32))
print(sess.run(accuray,feed_dict={x:mnist.test.images,y_true:mnist.test.labels}))