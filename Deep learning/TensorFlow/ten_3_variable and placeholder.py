import tensorflow as tf

x = tf.placeholder("float")
y = 2 * x
data = tf.random_uniform([4,5],10)
sess=tf.Session()
x_data=sess.run(data)#here we have to run data first since then only it will be calculated before giving to placeholder
print(sess.run(y,feed_dict={x:x_data}))
'''with tf.Session() as sess:#another way of running session
    x_data = sess.run(data)
     print(sess.run(y, feed_dict = {x:data}))'''