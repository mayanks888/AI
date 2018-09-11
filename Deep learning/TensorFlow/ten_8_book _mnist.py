import tensorflow as tf
# Import MNIST data
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
# Parameters
learning_rate = 0.001
training_iters = 500
batch_size = 128
display_step = 10
# Network Parameters
n_input = 784
# MNIST data input (img shape: 28*28)

n_classes = 10
# MNIST total classes (0-9 digits)
dropout = 0.85
# Dropout, probability to keep units

x = tf.placeholder(tf.float32, [None, n_input])
y = tf.placeholder(tf.float32, [None, n_classes])
keep_prob = tf.placeholder(tf.float32)

def conv2d(x, W, b, strides=1):
    x = tf.nn.conv2d(x, W, strides=[1, strides, strides, 1], padding='SAME')
    x = tf.nn.bias_add(x, b)
    return tf.nn.relu(x)
def maxpool2d(x, k=2):
    return tf.nn.max_pool(x, ksize=[1, k, k, 1], strides=[1, k, k, 1], padding='SAME')

def conv_net(x, weights, biases, dropout):
    # reshape the input picture
    x = tf.reshape(x, shape=[-1, 28, 28, 1])
    # First convolution layer
    conv1 = conv2d(x, weights['wc1'], biases['bc1'])
    # Max Pooling used for downsampling
    conv1 = maxpool2d(conv1, k=2)
    # Second convolution layer
    conv2 = conv2d(conv1, weights['wc2'], biases['bc2'])
    # Max Pooling used for downsampling
    conv2 = maxpool2d(conv2, k=2)
    # Reshape conv2 output to match the input of fully connected layer
    fc1 = tf.reshape(conv2, [-1, weights['wd1'].get_shape().as_list()[0]])
    # Fully connected layer
    fc1 = tf.add(tf.matmul(fc1, weights['wd1']), biases['bd1'])
    fc1 = tf.nn.relu(fc1)
    # Dropout
    fc1 = tf.nn.dropout(fc1, dropout)
    # Output the class prediction
    out = tf.add(tf.matmul(fc1, weights['out']), biases['out'])
    return out

weights = {
# 5x5 conv, 1 input, and 32 outputs
'wc1': tf.Variable(tf.random_normal([5, 5, 1, 32])),
# 5x5 conv, 32 inputs, and 64 outputs
'wc2': tf.Variable(tf.random_normal([5, 5, 32, 64])),
# fully connected, 7*7*64 inputs, and 1024 outputs
'wd1': tf.Variable(tf.random_normal([7*7*64, 1024])),
# 1024 inputs, 10 outputs for class digits
'out': tf.Variable(tf.random_normal([1024, n_classes]))
}
biases = {
'bc1': tf.Variable(tf.random_normal([32])),
'bc2': tf.Variable(tf.random_normal([64])),
'bd1': tf.Variable(tf.random_normal([1024])),
'out': tf.Variable(tf.random_normal([n_classes]))
}

pred = conv_net(x, weights, biases, keep_prob)
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)
correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
init = tf.global_variables_initializer()

train_loss = []
train_acc = []
test_acc = []
sess= tf.Session()
step = 1
while step <= training_iters:
    batch_x, batch_y = mnist.train.next_batch(batch_size)
    sess.run(optimizer, feed_dict={x: batch_x, y: batch_y,
    keep_prob: dropout})
    if step % display_step == 0:
        loss_train, acc_train = sess.run([cost, accuracy],
        feed_dict={x: batch_x,
        y: batch_y,keep_prob: 1.})
        print( "Iter " + str(step) + ", Minibatch Loss= " + \
        "{:.2f}".format(loss_train) + ", Training Accuracy= " + \
        "{:.2f}".format(acc_train))
        # Calculate accuracy for 2048 mnist test images.
        # Note that in this case no dropout
        acc_test = sess.run(accuracy,
        feed_dict={x: mnist.test.images,
        y: mnist.test.labels,
        keep_prob: 1.})
        print ("Testing Accuracy:" + \
        "{:.2f}".format(acc_train))
        train_loss.append(loss_train)
        train_acc.append(acc_train)
        test_acc.append(acc_test)
    step += 1