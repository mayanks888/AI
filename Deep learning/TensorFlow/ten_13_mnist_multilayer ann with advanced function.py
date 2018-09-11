import tensorflow as tf
import  numpy as np
import matplotlib.pyplot as plt

from tensorflow.examples.tutorials.mnist import input_data
mnist=input_data.read_data_sets('MNIST_data/',one_hot=1)
# my_image=mnist.train.images[26].reshape(28,28)#printing any image at index image[index] and shaping it in28X28
# plt.imshow(my_image,cmap="gist_gray")
# plt.show()

total_pixel_input=784#28X28 matrix
hidden_layer_neuron=100
output=10#this does not have hidded layer as of now so we are considering 10 output sinceoutput can be 0 to 9
seed_val=10
epochs=100
batch=1000
input_matrix=tf.placeholder(dtype=tf.float32,shape=[None,total_pixel_input])#(no of rows*728)
output_matrix=tf.placeholder(dtype=tf.float32,shape=[None,output])

def mymultilayer_perceptron(my_input_matrix):#one function will take care of all weight and bias thing
    ful_con_layer_1 = tf.contrib.layers.fully_connected(my_input_matrix, hidden_layer_neuron, activation_fn=tf.nn.relu, scope='fc1')
    out_ful_conected_layer=tf.contrib.layers.fully_connected(ful_con_layer_1, output, activation_fn=tf.sigmoid, scope='ol')
    return out_ful_conected_layer


y_hat = mymultilayer_perceptron(input_matrix)
cross_entropy_val=tf.nn.softmax_cross_entropy_with_logits(logits=y_hat,labels=output_matrix)
loss=tf.reduce_mean(cross_entropy_val)

optimiser=tf.train.AdamOptimizer(learning_rate=.05).minimize(loss)

init=tf.global_variables_initializer()

sess=tf.Session()
sess.run(init)


for i in range(epochs):
    batch_x, batch_y = mnist.train.next_batch(batch)
    # actual_weight, actual_bias=sess.run([weight,bias])
    # print (actual_weight)
    mycross_entropy, my_softmax_val, all_cross_ent, _ = sess.run([loss, y_hat, cross_entropy_val, optimiser],
                                                                 feed_dict={input_matrix: batch_x,
                                                                                    output_matrix: batch_y})

    my_prediction_train = tf.equal(tf.argmax(my_softmax_val, 1), tf.argmax(batch_y,1))  # comparing max index of softmax output and true y output matrix
    # cast convert true to 1 an false to 0
    accuray = tf.reduce_mean(tf.cast(my_prediction_train, tf.float32))
    prediction_val, final_acuracy = (sess.run([my_prediction_train, accuray]))
    print ('epoch no ', i)
    #print("predicted value are", prediction_val)
    print('accuracy ', final_acuracy * 100)

    # loss= sess.run([loss], feed_dict={input_matrix: batch_x, output_matrix: batch_y})

my_prediction=tf.equal(tf.argmax(y_hat,1),tf.argmax(output_matrix,1))#comparing max index of softmax output and true y output matrix
#cast convert true to 1 an false to 0
accuray=tf.reduce_mean(tf.cast(my_prediction,tf.float32))
prediction_val,final_acuracy=(sess.run([my_prediction,accuray],feed_dict={input_matrix:mnist.test.images,output_matrix:mnist.test.labels}))
print ("all predicted value are", prediction_val )
print('Test accuracy ', final_acuracy*100)