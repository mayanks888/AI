import  tensorflow as tf

'''a=tf.constant(5,name="input_name")
b=tf.constant(3,name='input_b')
c=tf.multiply(a,b,name='input_c')
d=tf.add(a,b,name="input_d")
e=tf.add(c,d,name='input_e')

# sess=tf.Session()
# print(sess.run(c))
# print (sess.run(e))

with tf.Session() as sess:
	writer = tf.summary.FileWriter("output", sess.graph)
	print(sess.run(e))
	writer.close()'''

#this session deals with definig variable and scope in the graph

with tf.name_scope('Scope A'):
    a=tf.multiply(2,3,name="multiply_a")
    b=tf.add(3,4,name='add b')

with tf.name_scope('Scope B'):
    c=tf.div(4,2)
    d=tf.Variable(initial_value=3)
    j=tf.add(d,c)

init=tf.global_variables_initializer()
sess=tf.Session()
sess.run(init)
print(sess.run(j))

print (a)
