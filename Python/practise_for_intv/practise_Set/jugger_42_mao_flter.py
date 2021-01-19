'''Problem 42:

Write a program which can map() and filter() to make a list whose
elements are square of even number in [1,2,3,4,5,6,7,8,9,10].
'''

mylist=list(range(11))

def sqr(val):
    if val%2==0:
        return True
    else:
        return False
    # return val*val

func=lambda val:val*val

cool=list(filter(sqr, mylist))
final_vale=list(map(func,cool))
print(final_vale)