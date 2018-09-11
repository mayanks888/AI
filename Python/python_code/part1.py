print ('hello mayan"k')
#new print format
print ('hello {} my name is {}'.format('sandy','mayank'))

list_val=[2,3,3,4,5]
(list_val.append(8))
print (list)
#nested list
list2=[1,2,3,[5,6]]
print (list2[3][1])
#sets they removed the repeating value
set_data={9,1,2,2,3,3,5,5,6,1}
print (set_data)#remove the repeating value

#running a different type of for loop
out =[num**2 for num in list_val]#converting a for loop to list

print (out)

#I will use map function as its seems interesting
x=[2,3,3,4,5]
def sqrare(value):
    return value**2

t=list(map(sqrare,x))
print (t)

#now using lambda function instead of def function

data= lambda value:value**2
print (data(2))#assign the value as like a function

# now using map and lambda together
j=map(lambda value:value**2,x)
print (list(j))#converting j object to list

# lets play with filter now its is used for removing item from list
dat= filter(lambda value:value%2==0,x)
print (list(dat))


dis='hi my name is mayank'
list_call=(dis.split())
print(len(list_call))

# print property
print ( 'the radius of {a} is {n}'.format(a='circle',n=80 ))