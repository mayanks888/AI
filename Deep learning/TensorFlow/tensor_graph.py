import tensorflow as tf
import numpy as np
#this is part one of tensor graph
'''a=tf.constant(5,name='input_a')
b=tf.constant(3,name='input_b')
c=tf.multiply(a,b,name='input_c')
d=tf.add(a,b,name='input_d')
e=tf.add(c,d,name= 'input_e')'''

#this is part two of tensor grapg
a=tf.constant([5,3])
b=tf.reduce_prod(a)#this function multiply a value amoung itself
c=tf.reduce_sum(a)
e=tf.add(c,b)
sess=tf.Session()
# writer = tf.summary.FileWriter('./my_graph', graph=sess.graph)
# print(sess.run(e))
# writer = tf.train.summary.FileWriter('./my_graph', sess.graph)

writer = tf.summary.FileWriter("/home/mayank-s/PycharmProjects/output_graph", sess.graph)
# writer.add_graph(sess.graph)
print(sess.run(e))

#this part will explain us feed dictionary concepts
a=tf.multiply(2,3)
b=tf.add(a,4)
sess=tf.Session()
out=(sess.run(b,feed_dict={a:5}))#basically feed dictionary overite tensor value
print(out)

#this will be used for placeholder

a=tf.placeholder(tf.float32,shape=None)
b=tf.reduce_sum(a)
c=tf.reduce_prod(b)

sess=tf.Session()
# out=sess.run(c,feed_dict={a:[2,3]})
out=sess.run(c,feed_dict={a:np.array([2,3], dtype=np.int32)})#feedinf 2d tensor as input
print (out)

#this part we are dealing with thw variable

a=tf.Variable(initial_value=3)
b=a.assign(a*2)#reinitialise value of variable and setting it to b
init=tf.global_variables_initializer()
sess=tf.Session()
sess.run(init)
print(sess.run(b))
print(sess.run(b))
# Variable.assign_add() Variable.assign_sub() this function is used to chage variable values by one
