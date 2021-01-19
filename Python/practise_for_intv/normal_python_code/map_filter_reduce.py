def getting_square(a):
    return (a * a)

def isgreate(a):
    # return a>5,0:
    if a>5:
        return True
    else:
        return False

mylist = [1, 2, 3, 4,8,9]

squarelist = [val * val for val in mylist]
print(squarelist)
# with map fucntion

map_convert = list(map(getting_square, mylist))
map_convert = tuple(map(getting_square, mylist))
print(map_convert)

#filter
filter_conver=list(filter(isgreate,mylist))
print(filter_conver)

#reduce
from functools import reduce

sun_func=lambda a,b: a+b
mul_function=lambda a,b :a*b
sum_val =reduce(sun_func,mylist)
#this is the best function to multiple value of the list
mul_val =reduce(mul_function,mylist)
print(sum_val,mul_val)


mythird_list=["may","june","july"]

playstrint=lambda a,b :a+ " and "+b

print(reduce(playstrint,mythird_list))