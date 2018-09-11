import tensorflow as tf
hello=tf.constant("hello_ sandy")#this is how you define constant in tensorflow

wel=tf.constant('wel to myworld')
print (hello)
# with tf.Session as sess:

a=tf.constant(2)
b=tf.constant(3)

matrix_val=tf.fill((4,4),3)
norm_dist=tf.random_normal((3,3),mean=0,stddev=5)
uniform_dist=tf.random_uniform((4,4),minval=0,maxval=9)
mylist=[matrix_val,norm_dist,uniform_dist]#this is way to run multiple data in one go in session
# playing with variable and placeholder
my_var=tf.Variable(initial_value=5)
print (my_var)
init=tf.global_variables_initializer()
sess=tf.Session()#need to create this session to create tensorflow environment
data=sess.run(hello)
print (data)
print (sess.run(a+b))
print (sess.run(matrix_val))
print (uniform_dist.eval(session=sess))#another way to define run session

for dat in mylist:
    print (sess.run(dat))
    print('\n')


sess.run(init)#this is for declaring variables
print (sess.run(my_var))