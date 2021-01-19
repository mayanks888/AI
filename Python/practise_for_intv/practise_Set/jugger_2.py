'''Problem 2:

Write a program which can compute the factorial of a given numbers.
The results should be printed in a comma-separated sequence on a single
line.Suppose the following input is supplied to the program: 8 Then,
the output should be:40320
'''
from functools import reduce
mul_func=lambda a,b:a*b

lis=list(range(1,9))
sum =reduce(mul_func,lis)
print(sum)
