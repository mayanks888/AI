from functools import reduce
import numpy as np
import random
mylist=[1,2,3,4]

sum= lambda a, b: a+b

mysum=reduce(sum, mylist)
print(mysum)

val=np.random.randint(0,3,20)
val=list(val)
print(val)

print("no of time 0 occered", val.count(0))

val2=random.sample(range(0,30), 30)
print(val2)
# or

val3= [random.randint(0, 3) for iter in range(30)]
print(val3)

val4=[random.randint(0,1) for _ in range(20)]
print(val4)

new_list=[1,2,3,4,5,6,7,8,9]
print("slicedlist",new_list[0:-1:2])


lst =list(range(1, 11))
print (lst)
