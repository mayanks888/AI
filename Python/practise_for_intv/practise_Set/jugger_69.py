'''Problem 70:

Please write a program to output a random even number
between 0 and 10 inclusive using random module and
list comprehension.
'''

import random

val =random.randint(0,10)
print(val)

import random
resp = [i for i in range(0,11,2)]
print(random.choice(resp))

'''Problem 71: 

Please write a program to output a random number, 
which is divisible by 5 and 7, between 10 and 150 inclusive 
using random module and list comprehension.
'''

import random
rval=[val for val in range(10,151) if (val%5==0 and val%7==0)]
print(rval)

print(random.choice(rval))

'''Problem 72: 

Please write a program to generate a list with 5 random numbers 
between 100 and 200 inclusive.
'''

val=[random.randint(100,200) for _ in range(5)]
print(val)

'''Problem 73: 

Please write a program to randomly generate a list with 5 even 
numbers between 100 and 200 inclusive.
'''
resp = random.sample(range(100,201,3),10)
print(resp)