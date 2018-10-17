# this is made for my understanding about python built in function


a = 'Real Python'
b = ['Real', 'Python']

# Here both the len function will perform same function
print(len(a))

print(a.__len__())

print(b[0])

print(b.__getitem__(0))

print(dir(a))

class checkbuilt():
    def __init__(self):
        self.data=[1,2,3,4,5,6]


    def __len__(self):
        return len(self.data)

    def find_length(self):
        return len(self.data)

    def __str__(self):
        return (str(self.data))

    '''The[] operator is called the indexing operator and is used in various contexts 
    in Python such as getting the value at an index in sequences, getting the value 
    associatedwith a key in dictionaries, or obtaining a part of a sequence through 
    slicing.You can change its behavior using the __getitem__() special method.'''

    def __getitem__(self, key):
        print('hello i getting an index now')
        return self.data[key]

    def __setitem__(self, key, value):
        self.data[key]=value

    def __iter__(self):  # iterate over all keys
        for x in (self.data):
            yield x
            # dat.append()

    def __call__(self,a,b):
        sum = a+b
        return sum



thor=checkbuilt()
# Here you can see that by using(__len__) we were able to use thor as to find the lenth of list directly)
print("by built in method",len(thor))

""" you should keep in mind that Python requires the function to return an integer.
If your method were to return anything other than an integer, you would get a 
TypeError.This most probably, is to keep it consistent  # with the fact that len() 
is generally used to obtain the length of a sequence, which can only be an integer:"""

print("by normal orthodox method",thor.find_length())


print("by built in method string present is ",str(thor))

print("by built in method to extract index values is ",thor[1])
# print("by built in method to change index values is ",thor[1,89])

# this buit in function  (__call__) is used to run the class object direcat by directly calling the instance of class check example
print("by built in method to call the function(__call__) ",thor(2,4))

# this is how you will use __iter__ whoch is like running the for loop in the in list or anything
for item in thor :
    print("by built in method to use iteration (__iter__) ",item)




# ***********************************************************************************
class Order:
    def __init__(self, cart, customer):
        self.cart = list(cart)
        self.customer = customer

    def __add__(self, other):
        new_cart = self.cart.copy()
        new_cart.append(other)
        return Order(new_cart, self.customer)

order = Order(['banana', 'apple'], 'Real Python')

# print((order + 'orange')) # New Order instance
print((order + 'orange').cart ) # New Order instance
print(order.cart ) # Original instance unchanged
order = order + 'mango'  # Changing the original instance
print(order.cart)




import collections.abc

class MyMap(collections.abc.Mapping):
    def __init__(self, n):
        self.n = n

    def __getitem__(self, key): # given a key, return it's value
        if 0 <= key < self.n:
            return key * key
        else:
            raise KeyError('Invalid key')

    def __iter__(self): # iterate over all keys
        for x in range(self.n):
            yield x

    def __len__(self):
        return self.n

m = MyMap(5)
for k, v in m.items():
    print(k, '->', v)