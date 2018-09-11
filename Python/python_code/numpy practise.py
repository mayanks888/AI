import numpy as np
# its is basically one dimensional vector or multidimensional array of matrix

mylist=[1,2,3]
arr=np.array(mylist)
print (arr)
# uses range to create a array

arr2=np.arange(0,10)
print (arr2)

# print (np.linspace(0,5,50))

# print (np.identity(5))#create idendity matrix

# print (np.random.randint(0,10,5)) #create random integer array of value

# print (np.random.rand(5,5)) #create random integer array of value


#this is to convery one dimendsional array to 2 d array
# we will arr2
# print (arr2.reshape(2,5))# we will arr2 to change to 2d keeping item to fit inside 2 d

print (arr2.max())#find max value in arary
print (arr2.argmax())#find index for maximum value
print (arr2.shape)#find shape
print(arr2.dtype)

#numpy indexing and selection

arr=np.arange(0,11)
print (arr[5])
import numpy as np
#numpy indexing and selection

arr=np.arange(0,11)
print (arr[5])#find the value at index 5
print (arr[2:6])#same as slicing as in string
print (arr[:])#remember this it will be useful in future data science
#this is important info np arary does not copy of array that is it will always create same reference
# example
myaray=arr
myaray[:]=99
print (myaray)
print (arr)#check original arr also changed
# or else use np.copy

myaray=np.copy(arr)
myaray[:]=45
print (myaray)
print (arr)


#now lets work with 2 d array
arr2=np.arange(0,9)
newarry=arr2.reshape(3,3)
print(newarry)
print (newarry[1,2])
print (newarry[1][1])#check indexing at 2 aray
print (newarry[1:,1:])#understand this concepts *

#cindition searching in array
print (newarry[newarry>3])

# numpy operations
arr2=np.arange(0,11)
print (arr2+arr2)
print (arr2/arr2)#this one is good to understand use it
print (np.sqrt(arr2))