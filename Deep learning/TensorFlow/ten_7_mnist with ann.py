import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.python import debug as tf_debug
from tensorflow.examples.tutorials.mnist import input_data
mnist=input_data.read_data_sets('MNIST_data/',one_hot=1)
# my_image=mnist.train.images[26].reshape(28,28)#printing any image at index image[index] and shaping it in28X28
# plt.imshow(my_image,cmap="gist_gray")
# plt.show()

total_pixel_input=784#28X28 matrix
total_neuron=10#this does not have hidded layer as of now so we are considering 10 output sinceoutput can be 0 to 9

#creating placeholder
input_matrix=tf.placeholder(dtype=tf.float32,shape=[None,total_pixel_input])#(no of rows*728)
output_matrix=tf.placeholder(dtype=tf.float32,shape=[None,total_neuron])

#creating variable
# w=tf.Variable(tf.random_normal(shape=[total_pixel_input,total_neuron]))#(728*10)
w=tf.Variable(tf.zeros(shape=[total_pixel_input,total_neuron]))#(728*10)
b=tf.Variable(tf.ones([10]))
matrix_multiply=tf.matmul(input_matrix,w)+b #output should be a matrix of (10 colum and batch side rows)

#apply softmax  as aactivation function for all ten neuron
#remember here only one softmax is applied to ouput of all 10 neuron
softmax_out=tf.nn.softmax(matrix_multiply)

all_cross_entropy=tf.nn.softmax_cross_entropy_with_logits(labels=output_matrix, logits=softmax_out)

loss = tf.reduce_mean(all_cross_entropy,name="loss_Calculation")
    # tf.summary.scalar('cross-entropy', loss)

optimiser=tf.train.GradientDescentOptimizer(learning_rate=.001).minimize(loss)

init=tf.global_variables_initializer()


sess=tf.Session()

sess.run(init)
# writer = tf.summary.FileWriter("/home/mayank-s/PycharmProjects/output_graph/1", sess.graph)
sess = tf_debug.TensorBoardDebugWrapperSession(sess, "localhost:7000")
training_steps=100

for i in range(training_steps):
    batch_x,batch_y=mnist.train.next_batch(100)
    # my_comp_data=sess.run([y,train_data,],feed_dict={input_matrix:batch_x,output_matrix:batch_y})
    mycross_entropy,my_softmax_val,all_cross_ent,_=sess.run([loss,softmax_out,all_cross_entropy,optimiser],feed_dict={input_matrix:batch_x,output_matrix:batch_y})
    # print (mycross_entropy,my_softmax_val)

#evaluate my model
my_prediction=tf.equal(tf.argmax(softmax_out,1),tf.argmax(output_matrix,1))#comparing max index of softmax output and true y output matrix
#cast convert true to 1 an false to 0
accuray=tf.reduce_mean(tf.cast(my_prediction,tf.float32),name="Accuracy")
prediction_val,final_acuracy=(sess.run([my_prediction,accuray],feed_dict={input_matrix:mnist.test.images,output_matrix:mnist.test.labels}))
print ("predicted value are", prediction_val )
print('accuracy ', final_acuracy*100)