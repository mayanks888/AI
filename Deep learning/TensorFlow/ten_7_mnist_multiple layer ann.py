import tensorflow as tf
import  numpy as np
import matplotlib.pyplot as plt
from tensorflow.python import debug as tf_debug
from tensorflow.examples.tutorials.mnist import input_data
mnist=input_data.read_data_sets('MNIST_data/',one_hot=1)
# my_image=mnist.train.images[26].reshape(28,28)#printing any image at index image[index] and shaping it in28X28
# plt.imshow(my_image,cmap="gist_gray")
# plt.show()

total_pixel_input=784#28X28 matrix
hidden_layer_neuron=100
output=10#this does not have hidded layer as of now so we are considering 10 output sinceoutput can be 0 to 9
seed_val=10
epochs=200
batch=100
input_matrix=tf.placeholder(dtype=tf.float32,shape=[None,total_pixel_input])#(no of rows*728)
output_matrix=tf.placeholder(dtype=tf.float32,shape=[None,output])

# w=tf.Variable(tf.zeros(shape=[total_pixel_input,total_neuron]))#(728*10)
# # b=tf.Variable(tf.ones([10]))
weight={"ih":tf.Variable(tf.random_normal([total_pixel_input,hidden_layer_neuron],seed=seed_val)),
        'ho':tf.Variable(tf.random_normal([hidden_layer_neuron,output],seed=seed_val))}
# weight={"ih":tf.Variable(tf.zeros(shape=[total_pixel_input,hidden_layer_neuron])),
#         'ho':tf.Variable(tf.zeros(shape=[hidden_layer_neuron,output]))}
# weight={"ih":tf.Variable(tf.truncated_normal([total_pixel_input,hidden_layer_neuron])),
#         'ho':tf.Variable(tf.truncated_normal([hidden_layer_neuron,output]))}
bias={'ih':tf.ones(hidden_layer_neuron),'oh':tf.ones(output)}


# bias = {
#      'ih': tf.Variable(tf.random_normal([1, hidden_layer_neuron], seed = seed_val)),
#      'oh': tf.Variable(tf.random_normal([1, output], seed = seed_val))}
#

hl_input=tf.add(tf.matmul(input_matrix,weight['ih']),bias['ih'])
hl_output=tf.sigmoid(hl_input)
# hl_output=tf.nn.relu(hl_input)

ol_input=tf.add(tf.matmul(hl_output,weight['ho']),bias['oh'])
ol_output=tf.nn.softmax(ol_input,name="output_Value_of_softmax")
# ol_output=tf.sigmoid(ol_input)

cross_entropy_val=tf.nn.softmax_cross_entropy_with_logits_v2(logits=ol_output,labels=output_matrix)
loss=tf.reduce_mean(cross_entropy_val,name="loss")
Actual_output=output_matrix
# optimiser=tf.train.GradientDescentOptimizer(learning_rate=.01).minimize(loss)
optimiser=tf.train.AdamOptimizer(learning_rate=.01).minimize(loss)

my_prediction_train = tf.equal(tf.argmax(ol_output,1), tf.argmax(output_matrix,1))
accuray = tf.reduce_mean(tf.cast(my_prediction_train, tf.float32),name='accuracy')

# tf.summary.scalar(name='Accuracy',accuray)

init=tf.global_variables_initializer()

sess=tf.Session()
sess.run(init)
# sess = tf_debug.TensorBoardDebugWrapperSession(sess, "localhost:7000")

for i in range(epochs):
    batch_x, batch_y = mnist.train.next_batch(batch)
    # actual_weight, actual_bias=sess.run([weight,bias])
    # print (actual_weight)
    # mycross_entropy, my_softmax_val, all_cross_ent, _ = sess.run([loss, ol_output, cross_entropy_val, optimiser],
    #                                                              feed_dict={input_matrix: batch_x,
    #                                                                                 output_matrix: batch_y})
    mycross_entropy, myaccuracy, all_cross_ent, _,Sof_output,y_input = sess.run([loss, accuray, cross_entropy_val, optimiser,ol_output,Actual_output],
                                                                 feed_dict={input_matrix: batch_x,
                                                                            output_matrix: batch_y})


    '''my_prediction_train = tf.equal(tf.argmax(my_softmax_val, 1), tf.argmax(batch_y,1))  # comparing max index of softmax output and true y output matrix
    # cast convert true to 1 an false to 0
    accuray = tf.reduce_mean(tf.cast(my_prediction_train, tf.float32),name='accuracy')
    prediction_val, final_acuracy = sess.run([my_prediction_train, accuray],feed_dict={input_matrix: batch_x,output_matrix: batch_y})'''
    # print (Sof_output)
    # print (y_input)
    # print(np.argmax(Sof_output,axis=1), np.argmax(y_input,axis=1))
    print(np.sum(Sof_output))
    print ('epoch no ', i)
    #print("predicted value are", prediction_val)
    print('accuracy ', myaccuracy * 100)

    # loss= sess.run([loss], feed_dict={input_matrix: batch_x, output_matrix: batch_y})

# my_prediction=tf.equal(tf.argmax(ol_output,1),tf.argmax(output_matrix,1))#comparing max index of softmax output and true y output matrix
# #cast convert true to 1 an false to 0
# accuray=tf.reduce_mean(tf.cast(my_prediction,tf.float32))
# prediction_val,final_acuracy=(sess.run([my_prediction,accuray],feed_dict={input_matrix:mnist.test.images,output_matrix:mnist.test.labels}))
# print ("all predicted value are", prediction_val )
# print('Test accuracy ', final_acuracy*100)