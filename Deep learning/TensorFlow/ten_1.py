import numpy as np
import  tensorflow as tf

# '______________________________________________________________________________________
#this is numpy matrix understanding
a=[[1,2,3],[2,5,3]] #creating a 2 d matrix  of (2,3)
dm_2=[[[1],[2],[3]],[[4],[5],[6]]]
dm_arr=np.array(dm_2)
k= np.array(a)
print (k, k.shape)
print(dm_arr,dm_arr.shape)
dm_3=np.array([[[1,5],[2,7],[3,9]],[[4,7],[5,63],[6,78]]])#creating 3 d matrix where dim=2, row=3,colm=2(here dim is like depth of matrix)
print(dm_3,dm_3.shape)
print(dm_3[0,2,1])#print element of (0 dim arrat,3 row, 2nd column)(remember in matrix address start from 0'''
# '______________________________________________________________________________________
# tenor (rank ,shape, datatype understanding)
val=tf.constant([1,2])#one dimensional matrix or vector(tensor of rank 1)
val2=tf.constant([[1,2,3],[7,5,3]])#2d matrix( tenosor of rank 2
val3=tf.constant([[[1,5],[2,7],[3,9]],[[4,7],[5,63],[6,78]]],dtype=tf.float32)#(3 dimsion matrix)(tensor of rank 3
con_data=tf.constant(.1,shape=(3,4),dtype=tf.float32)#another way to create a array of constant
sess=tf.Session()
print (sess.run(val[1]))
print (sess.run(val2[1,2]))
print (sess.run(val3[1,1,0]))
print (val,val2,val3)#this shows shape of tensor
# '______________________________________________________________________________________
# convert np array to tensor
a=np.array([[1,2,3],[2,5,3]]) #creating a 2 d matrix  of (2,3)
tenc=tf.convert_to_tensor(a,dtype=tf.float32)#convert array to tensor
print (sess.run(tenc),tenc)
# '______________________________________________________________________________________
#creating your own graph

g=tf.Graph()
with g.as_default():
    var1=tf.Variable(tf.ones(5),dtype=tf.float32)
    var3=tf.constant(1.0)
    var4 = tf.constant(4.0)
    var2=tf.Variable(tf.constant(2.3),dtype=tf.float32)
    # var2=tf.Variable(tf.constant([2.0,3,4,5,6]),dtype=tf.float32)
    mul=tf.multiply(var4,var3)
    sess=tf.Session()
    print(sess.run(mul))

# print (g.as_graph_element())
# '______________________________________________________________________________________
#martrix multiply

mat1=tf.constant([7,8,9,12,34,21],dtype=tf.float32,shape=[2,3])

mat2=tf.constant([1,2,3,4,5,6],dtype=tf.float32,shape=[3,2])#shape matter a lot for matrix multiplication
mat3=tf.linspace(start=1.0,stop=10,num=6)#to create you own list of constant
mul=tf.matmul(mat1,mat2)
new_dat=tf.range(10)#print range of value just like in python
shuffle=tf.random_shuffle(new_dat)
sess=tf.Session()
print(sess.run(mul))
print (sess.run(mat3))
print(sess.run(new_dat))
print(sess.run(shuffle))

# '______________________________________________________________________________________
#further function on martrix
identy=tf.eye(3)#create idendity matrix

b = tf.Variable(tf.random_uniform([5,10], 0, 2, dtype= tf.int32))#create a random matrix of 1s and 0s, size 5x10
init=tf.global_variables_initializer()
sess=tf.Session()
sess.run(init)
print(sess.run(identy))
print(sess.run(b))


