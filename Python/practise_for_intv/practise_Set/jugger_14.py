'''Problem 15:

Write a program that computes the value of a+aa+aaa+aaaa with
a given digit as the value of a.

Suppose the following input is supplied to the program:

9
Then, the output should be:

11106'''
from functools import reduce
val=9

fun=lambda a,b:(a+(10*a))

# value=reduce(fun, list(range(1,val)))
# print(value)

# val=99
limit=9
print((val*10)+val)
fist_Val=limit
finalval=0
sum=0
for val in range(4):
    finalval=(finalval * 10+limit)
    sum=sum+finalval
print(sum)


#
#
# a = str(9)
# total,tmp = 0,str()        # initialing an integer and empty string
#
# for i in range(4):
#     tmp+=a               # concatenating 'a' to 'tmp'
#     total+=int(tmp)      # converting string type to integer type
#
# print(total)