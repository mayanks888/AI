

for data in range(5,50,5):
    print(data)

# assert(2==4)
myList=[1,5,7,8,6,2,2,8,4,5,5]

for val in myList:
    print(val)
loop=0
while(loop<10):
    print(loop)
    loop+=1


cool=[val+2 for val in myList]
print("coolVAl is",cool)

def make_square(a):
    return (a**2)

final_val=list(map(make_square, myList))
print("this is final list",final_val)

newList=[]
cool=[newList.append(make_square(val)) for val in myList]
# or
cool=[make_square(val) for val in myList]
print(newList)

#more advanced
new=[val for val in myList if val%2==0]
# myList.
print("more advanced",new)



myString="Mayank"
cool=myString.center(0)
print(cool)