import tensorflow as tf

# input_batch =tf.constant[[1,2],[3,4]]
input_batch=tf.fill((2,2),3.0)
kernel = tf.fill((2,2),2.0)

conv2d = tf.nn.conv2d(input_batch, kernel, strides=[1, 1, 1, 1], padding='SAME')

sess=tf.Session()
print(sess.run(input_batch))
print(sess.run(conv2d))
# lower_right_kernel_pixel = sess.run(conv2d)[1][0]
# print (lower_right_kernel_pixel)