'''Problem 35:

Define a function which can generate a list where the values are
square of numbers between 1 and 20 (both included).
Then the function needs to print the last 5 elements in the list.
'''


fun =lambda a:a*a

out=map(fun, list(range(21)))
out=list(out)
print(out[-5:])

'''Problem 38: 

With a given tuple (1,2,3,4,5,6,7,8,9,10), 
write a program to print the first half values in one line and 
the last half values in one line.
'''

data=tuple(range(21))

cool=[val for val in data if val%2==0]
print(cool)

unicodeString = u"hello world!"
print (unicodeString)