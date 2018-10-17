#very important concepts of iterator which can be used to better ulitlise and save memory

favorite_numbers = [6, 57, 4, 7, 68, 95]
iter_obj=iter(favorite_numbers)
# print(next (iter_obj))
# print(next (iter_obj))
# print(next (iter_obj))
# this mean that we can call next(iter_obj as many times to go to the next iteration)
for _ in range(len(favorite_numbers)):
    print('the next value in list is :',next(iter_obj))
    # print(next(iter_obj))


# &*********************************************************************************
# one more great example for the benfit of itereators
# This is the first line in a giant file

# noticed this practise save memory and time in loading data
print(next(open('/home/mayank-s/PycharmProjects/Datasets/housing_data.csv')))
print(next(open('/home/mayank-s/PycharmProjects/Datasets/README.txt')))

data=open('/home/mayank-s/PycharmProjects/Datasets/housing_data.csv')
iter_obj2=iter(data)

for _ in range(20):
    mylinedata=next(data)
    cool=mylinedata.split(',')
    print('the next line in csv file is :',mylinedata)

# *****************************************************************************************
# intrucing iteration in class
'''class Count:

    """Iterator that counts upward forever."""

    def __init__(self, start=0):
        self.num = start

    def __iter__(self):
        return self

    def __next__(self):
        num = self.num
        self.num += 1
        return num


c=Count()
print(next(c))
print(next(c))
print(next(c))'''


# *******************************************************************************8
class Point:
    def __init__(self, x, y):
        self.x, self.y = x, y

    def __iter__(self):
        yield self.x
        yield self.y

p=Point(3,4)
print(p)
x, y = p
print(x,y)
for item in p :
    print("by built in method to use iteration (__iter__) ",item)