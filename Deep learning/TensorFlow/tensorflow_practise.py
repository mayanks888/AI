import tensorflow as tf
# $ tensorboard --logdir /tmp/logdir --debugger_port 7000
from tensorflow.python import debug as tf_debug
x = tf.placeholder("float", [None, 3])
y=tf.Variable(initial_value=[[2,2,2],
                            [2,2,2],])

y = x * 2
z = y + 3
init=tf.global_variables_initializer()
sess=tf.Session()
sess.run(init)
# sess = tf_debug.TensorBoardDebugWrapperSession(sess, "MAYANK_PC:7000")
sess = tf_debug.TensorBoardDebugWrapperSession(sess, "localhost:7000")

# sess = tf_debug.TensorBoardDebugWrapperSession(sess, "MAYANK_PC:6064")
# sess.run(my_fetches)
# sess = tf_debug.TensorBoardDebugWrapperSession(sess, "http://0.0.0.0:6006/#debugger")

# sess.run(my_fetches)
x_data = [[1, 2, 3],
         [4, 5, 6],]
result = sess.run(z, feed_dict={x: x_data})
# sess = tf_debug.TensorBoardDebugWrapperSession(sess, "MAYANK_PC:7000")
# sess.run(my_fetches)
# new_result=sess.run(y,feed_dict={x: x_data})
# print(sess.run(x))

print(result)
# print (new_result)

# python "Deep learning\TensorFlow\tensorflow_practise.py"