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