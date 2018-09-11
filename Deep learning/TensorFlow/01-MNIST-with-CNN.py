import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
from tensorflow.python import debug as tf_debug
mnist = input_data.read_data_sets("MNIST_data/",one_hot=True)

# Function to help intialize random weights for fully connected or convolutional layers, we leave the shape attribute as a parameter for this.
def init_weights(shape):
    init_random_dist = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(init_random_dist)

# Same as init_weights, but for the biases
def init_bias(shape):
    init_bias_vals = tf.constant(0.1, shape=shape)
    return tf.Variable(init_bias_vals)
# Create a 2D convolution using builtin conv2d from TF. From those docs:
# 
# Computes a 2-D convolution given 4-D `input` and `filter` tensors.
# 
# Given an input tensor of shape `[batch, in_height, in_width, in_channels]`
# and a filter / kernel tensor of shape
# `[filter_height, filter_width, in_channels, out_channels]`, this op
# performs the following:
# 
# 1. Flattens the filter to a 2-D matrix with shape
#    `[filter_height * filter_width * in_channels, output_channels]`.
# 2. Extracts image patches from the input tensor to form a *virtual*
#    tensor of shape `[batch, out_height, out_width,
#    filter_height * filter_width * in_channels]`.
# 3. For each patch, right-multiplies the filter matrix and the image patch
#    vector.
#
def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')
# Create a max pooling layer, again using built in TF functions:
# 
# Performs the max pooling on the input.
# 
#     Args:
#       value: A 4-D `Tensor` with shape `[batch, height, width, channels]` and
#         type `tf.float32`.
#       ksize: A list of ints that has length >= 4.  The size of the window for
#         each dimension of the input tensor.
#       strides: A list of ints that has length >= 4.  The stride of the sliding
#         window for each dimension of the input tensor.
#       padding: A string, either `'VALID'` or `'SAME'`. 

def max_pool_2by2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                          strides=[1, 2, 2, 1], padding='SAME')

# Using the conv2d function, we'll return an actual convolutional layer here that uses an ReLu activation.
def convolutional_layer(input_x, shape):
    W = init_weights(shape)
    b = init_bias([shape[3]])
    return tf.nn.relu(conv2d(input_x, W) + b)

# This is a normal fully connected layer
def normal_full_layer(input_layer, size):
    input_size = int(input_layer.get_shape()[1])
    W = init_weights([input_size, size])
    b = init_bias([size])
    return tf.matmul(input_layer, W) + b

# ### Placeholders
x = tf.placeholder(tf.float32,shape=[None,784])
y_true = tf.placeholder(tf.float32,shape=[None,10])
# ### Layers
x_image = tf.reshape(x,[-1,28,28,1])

# Using a 6by6 filter here, used 5by5 in video, you can play around with the filter size
# You can change the 32 output, that essentially represents the amount of filters used
# You need to pass in 32 to the next input though, the 1 comes from the original input of 
# a single image.
convo_1 = convolutional_layer(x_image,shape=[6,6,1,32])
convo_1_pooling = max_pool_2by2(convo_1)

# Using a 6by6 filter here, used 5by5 in video, you can play around with the filter size
# You can actually change the 64 output if you want, you can think of that as a representation
# of the amount of 6by6 filters used.
convo_2 = convolutional_layer(convo_1_pooling,shape=[6,6,32,64])
convo_2_pooling = max_pool_2by2(convo_2)

# Why 7 by 7 image? Because we did 2 pooling layers, so (28/2)/2 = 7
# 64 then just comes from the output of the previous Convolution
convo_2_flat = tf.reshape(convo_2_pooling,[-1,7*7*64])
full_layer_one = tf.nn.relu(normal_full_layer(convo_2_flat,1024))

# NOTE THE PLACEHOLDER HERE!
hold_prob = tf.placeholder(tf.float32)
full_one_dropout = tf.nn.dropout(full_layer_one,keep_prob=hold_prob)

y_pred = normal_full_layer(full_one_dropout,10)
# ### Loss Function
cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_true,logits=y_pred))
# ### Optimizer
optimizer = tf.train.AdamOptimizer(learning_rate=0.0001)
train = optimizer.minimize(cross_entropy)


init = tf.global_variables_initializer()


#_________________________________________________
#
# my_prediction_train = tf.equal(tf.argmax(ol_output,1), tf.argmax(output_matrix,1))
# accuray = tf.reduce_mean(tf.cast(my_prediction_train, tf.float32),name='accuracy')
#
# tf.summary.histogram("Softmax", ol_output)
# # variable_summaries(weight['ih'])
# # variable_summaries(bias['ih'])
# # tf.summary.histogram("prediction", my_prediction_train)
# # tf.summary.scalar('cross entropy', cross_entropy_val)
# tf.summary.scalar("Acuracy", accuray)
# tf.summary.scalar("Loss", loss)
# summary_op = tf.summary.merge_all()
# #_________________________________________________


steps = 1500

with tf.Session() as sess:
    
    sess.run(init)
    # sess = tf_debug.TensorBoardDebugWrapperSession(sess, "localhost:7000")
    for i in range(steps):
        
        batch_x , batch_y = mnist.train.next_batch(100)
        
        sess.run(train,feed_dict={x:batch_x,y_true:batch_y,hold_prob:0.5})
        
        # PRINT OUT A MESSAGE EVERY 100 STEPS
        if i%10 == 0:
            
            print('Currently on step {}'.format(i))
            print('Accuracy is:')
            # Test the Train Model
            matches = tf.equal(tf.argmax(y_pred,1),tf.argmax(y_true,1))

            acc = tf.reduce_mean(tf.cast(matches,tf.float32))

            print(sess.run(acc,feed_dict={x:mnist.test.images,y_true:mnist.test.labels,hold_prob:1.0}))
            print('\n')

