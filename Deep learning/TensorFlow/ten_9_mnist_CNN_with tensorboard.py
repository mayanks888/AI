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
epochs=50
batch=100
learning_rate=.03
drop_out=0.85
input_matrix=tf.placeholder(dtype=tf.float32,shape=[None,total_pixel_input],name='input_matrix')#(no of rows*728)
output_matrix=tf.placeholder(dtype=tf.float32,shape=[None,output],name="outpu_matrix")
drop_out_val = tf.placeholder(tf.float32)

def create_conv2d(input_matrix_converted,filter_weight,bias,strides=1):
    val=tf.nn.conv2d(input=input_matrix_converted,filter=filter_weight,strides=[1,strides,strides,1],padding='SAME')
    new_val=tf.nn.bias_add(val,bias)
    return tf.nn.relu(new_val)


def max_pool(input_maxpol,k=2):
    max_pol_val=tf.nn.max_pool(value=input_maxpol,ksize=[1,k,k,1],strides=[1,k,k,1],padding="SAME")
    return max_pol_val

def convolution_layers(input_data,weight,bias,dropouts):
    input_data1=tf.reshape(input_data,shape=[-1,28,28,1])
    conv1=create_conv2d(input_data1,weight['wc1'],bias['bc1'])#first convolution layer
    maxp1=max_pool(conv1,k=2)
    conv2=create_conv2d(maxp1,weight['wc2'],bias['bc2'])#second layer
    maxp2 = max_pool(conv2,k=2)
    # Reshape conv2 output to match the input of fully connected layer
    fc1 = tf.reshape(maxp2, [-1, weight['wd1'].get_shape().as_list()[0]])
    fcl_input=tf.add(tf.matmul(fc1,weight['wd1']),bias['bd1'])
    fcl_output=tf.nn.relu(fcl_input)
    fc1_dropout = tf.nn.dropout(fcl_output, dropouts)#giving drop to to reduce overfitting
    #final_out=tf.add(tf.matmul(fc1_dropout,weight['out']),bias['out'])
    final_out=tf.add(tf.matmul(fcl_output,weight['out']),bias['out'])
    return final_out

#creating filters
fil_weights = {
# 5x5 conv, 1 input, and 32 outputs
'wc1': tf.Variable(tf.random_normal([5, 5, 1, 32])),
# 5x5 conv, 32 inputs, and 64 outputs
'wc2': tf.Variable(tf.random_normal([5, 5, 32, 64])),
# fully connected, 7*7*64 inputs, and 1024 outputs
'wd1': tf.Variable(tf.random_normal([7*7*64, 1024])),
# 1024 inputs, 10 outputs for class digits
'out': tf.Variable(tf.random_normal([1024, output]))
}

biases = {
'bc1': tf.Variable(tf.random_normal([32])),
'bc2': tf.Variable(tf.random_normal([64])),
'bd1': tf.Variable(tf.random_normal([1024])),
'out': tf.Variable(tf.random_normal([output]))
}


pred_value=convolution_layers(input_data=input_matrix,weight=fil_weights,bias=biases,dropouts=drop_out_val)
softmax_output=tf.nn.softmax(pred_value)
total_cross_entropy=tf.nn.softmax_cross_entropy_with_logits_v2(logits=softmax_output,labels=output_matrix)
loss=tf.reduce_mean(total_cross_entropy)
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)
# correct_prediction = tf.equal(tf.argmax(pred_value, 1), tf.argmax(output_matrix, 1))
# accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
init = tf.global_variables_initializer()

sess=tf.Session()
sess.run(init)
sess = tf_debug.TensorBoardDebugWrapperSession(sess, "localhost:7000")


for i in range(epochs):
    batch_x, batch_y = mnist.train.next_batch(batch)
    # actual_weight, actual_bias=sess.run([weight,bias])
    # print (actual_weight)
    layer_output,mycross_entropy, my_softmax_val, all_cross_ent, _ = sess.run([pred_value,loss, softmax_output,total_cross_entropy, optimizer],
                                                                 feed_dict={input_matrix: batch_x,output_matrix: batch_y,drop_out_val:drop_out})

    # feed_dicy=({input_matrix:input_data1, bias: bias['bc1']})
    # conv1_val=sess.run(create_conv2d)
    my_prediction_train = tf.equal(tf.argmax(my_softmax_val, 1), tf.argmax(batch_y,1))  # comparing max index of softmax output and true y output matrix
    # cast convert true to 1 an false to 0
    accuray = tf.reduce_mean(tf.cast(my_prediction_train, tf.float32))
    prediction_val, final_acuracy = (sess.run([my_prediction_train, accuray]))
    print ('epoch no ', i)
    #print("predicted value are", prediction_val)
    print('accuracy ', final_acuracy * 100)

# writer = tf.summary.FileWriter("../../../../output_graph", sess.graph)