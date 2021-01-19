'''Problem 65:

Please write assert statements to verify that every number
in the list [2,4,6,8] is even.'''
mylist=[2,4,6,9,8,]

for val in mylist:
    try:
        print(val)
        assert (val%2==0) ,print("it has odd value")
    except:
        print("cool")