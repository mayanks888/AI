a = 4
print(a + 59)
print(float(20 / 7))
print(20 // 5)  # python modules is somewhat different
print(2 ** 5)  # this is exponential
print(pow(2, 5))
# variable
a = 5  # python is case sensitive
A = 8
print(a, A)
print(dir(__builtins__))
print(help(max))
print(reversed([1, 5, 5, 6, 8]))

import math

print(math.sqrt(9))

# string
print("hi my \\ name is \" cool")
print(len("lenght of my string"))
randVariable = "mayank"
print(randVariable * 5)

# list
myList = [1, 2, "mayamk", 5]  # list can contain anything
print(myList)
print("forward_index", myList[0], myList[1])
print("backward_index", myList[-1], myList[-2])
myList.append(58)
for lis in myList:
    print(lis)
mySecdList = ["a", 78, "$"]
myList.extend(mySecdList)  # adding two list together
print("extended lsit os =", myList)
print(myList.remove(78))
print(myList.remove("a"))
print(myList)
print("after remove", myList)
thirdList = [1, 8, 5, 4, 3, 9, 8, 47, 96, 63]
thirdList.sort()
print("after sorting", thirdList)
print("sum of all list {}, sum of sliced list is {}".format(sum(thirdList),sum(thirdList[2:4])))
print("sum of all list {a}, sum of sliced list is {b}".format(a=sum(thirdList),b=sum(thirdList[2:4])))
# print("sum of sliced list",sum(thirdList[2:4]))
# this kind of function run on itself object
thirdList.reverse()
print("ater reversing", thirdList)
print("occurnace of no is", thirdList.count(8))
print("indexing", thirdList[0])
print("index with jump values",thirdList[0:-1:2])#here 2 repsent the jump value in list
thirdList.pop()
print("slicing", thirdList[1:4])
print("poping last", thirdList)
forthList = [[1, 2, 3], [5, 6, 7]]
print("slicing", forthList[1:][:])

print("count fothlist {}".format(forthList.count(2)))
print(len(forthList))


#creat list of 10 vales

myforthlist=[9]*6
print(myforthlist)

#copy function
myfifthList=["a","b","c"]
new_fifthList=myfifthList
new_fifthList.append("j")
print(new_fifthList)
print(myfifthList)
# to create a
new_fifthList=myfifthList.copy()
# or
new_fifthList=myfifthList[:]
new_fifthList.append("j")

print(new_fifthList)
print(myfifthList)
# to create a

#######################33
# get the index of any value in list

print(myfifthList.index("b"))



lst =list(range(1, 11))
print (lst)
