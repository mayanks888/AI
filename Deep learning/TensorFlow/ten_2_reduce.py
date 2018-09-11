import tensorflow as tf

'''x = [1., 2., 3.]
y = [1., 2., 3.]
z = [0., 1., 3.]
# __________________________________________-
# reduce_all or any
result1 = tf.equal(x, y)
result2 = tf.equal(y, z)

# Use reduce_all and reduce_any to test the results of equal.
result3 = tf.reduce_all(result1)
result4 = tf.reduce_all(result2)
result5 = tf.reduce_any(result1)
result6 = tf.reduce_any(result2)

sess = tf.Session()
print("EQUAL     ", sess.run(result1))
print("EQUAL     ", sess.run(result2))
print("REDUCE ALL", sess.run(result3))
print("REDUCE ALL", sess.run(result4))
print("REDUCE ANY", sess.run(result5))
print("REDUCE ANY", sess.run(result6))

q = tf.constant([[True,  True], [False, False]])
res7=tf.reduce_all(q)  # False
res8=tf.reduce_all(q, 0)  # [False, False],reduce in term of row,axis=0,vertically (DEFAULT: axis=0)
res9=tf.reduce_all(q, 1)  # [True, False],reduce interm of colum,axis=1,horizontally (axis=1)
print("REDUCE ALL", sess.run(res7))
print("REDUCE all", sess.run(res8))
print("REDUCE all", sess.run(res9))
# __________________________________________-
#reduce dimeiosnality by product
print (sess.run(tf.reduce_prod(x)))

mymatrix=tf.constant([1,-2,6,-4,9,2,6,7,2],shape=[3,3],dtype=tf.float32)
sess=tf.Session()
print(sess.run(tf.reduce_mean(mymatrix,axis=1)))
print(sess.run(tf.reduce_max(mymatrix,axis=1)))

# __________________________________________-

# segmention of  matrix
segdata=tf.constant([1,2,3,4,5,6,7,8,9,14,24,34,49,55,46,7,82,9,32,43],shape=[5,4],dtype=tf.float32)#creating matrix
sess=tf.Session()
print(sess.run(segdata))
segid=[0,0,1,1,2]#this is uninque way dor dimension reduction it segment your matrix into ( index of 0,1,2)
# so 0 inder matrix will be calculated in ine group and similar to index 1 and 3
print(sess.run(tf.segment_max(segdata,segment_ids=segid)))#check result for better understanding
print(sess.run(tf.segment_min(segdata,segment_ids=segid)))#check result for better understanding
print(sess.run(tf.segment_mean(segdata,segment_ids=segid)))#check result for better understanding
# __________________________________________-
# intercahange option in matrix
segdata=tf.constant([1,2,3,4,5,6,7,8,9,14,24,34,49,55,46,7,82,9,32,43],shape=[5,4],dtype=tf.float32)#creating matrix
seg2=tf.constant([[True,True],[False,True]])
seg2=tf.constant([True,True,False,True])
sess=tf.Session()
print(sess.run(segdata))
print(sess.run(tf.argmax(segdata,axis=1)))#this will give us output of indix of maximum value in horizontal axis
print(sess.run(tf.argmax(segdata,axis=0)))#max indiex in vertival or column wise
print(sess.run(tf.where(seg2)))
# __________________________________________-
#reading shape and size , and reshaping tensor

segdata=tf.constant([1,2,3,4,5,6,7,8,9,14,24,34,49,55,46,7,82,9,32,43],shape=[5,4],dtype=tf.float32)#creating matrix
seg2=tf.constant([[True,True],[False,True]])

sess=tf.Session()

print(sess.run(segdata))
print (tf.size(segdata).eval(session=sess))#another way to run session,
print(sess.run((tf.rank(segdata))))#this will the dimesion of tenor
print(sess.run(tf.reshape(segdata,shape=[10,2])))#reshape you tensor
print(sess.run(tf.squeeze(segdata)))#this will reset your tensor matrix into best suited rows and column
change_dim=(tf.expand_dims(segdata,1))#increase the diminsion of tensor(now row ,column and depth can be treated as x axis y axis and z axis)

print (change_dim,sess.run(change_dim))
print(sess.run(tf.shape(change_dim)))#printing shape of tensor'''



#checking reduce means
sum_check=tf.constant([1,2,3,4,5,6,7,8,9,14,24,34,49,55,46,7,82,9,32,43],shape=[5,4],dtype=tf.float32)
data1=tf.reduce_sum(sum_check,axis=1)
data2=tf.reduce_sum(sum_check,[1])

sess=tf.Session()
print(sess.run(sum_check))

print ('\n',sess.run(data1))

print ('\n',sess.run(data2))