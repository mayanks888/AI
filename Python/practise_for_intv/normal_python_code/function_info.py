

def add_val(a, b):
    return a+b

#pasing multiple parameter with *values
def sum_of_all(name, *values):
    for val in values:
        print(val)
    print(values)

def make_square(a):
    return (a**2)



if __name__ == '__main__': #just type main funciton in pycharm
    print(add_val(4,5))
    print(sum_of_all("mayan",3,5,7,87))

    myList = [1, 5, 8, 9, 7, 33, 4, 5, 8, 2]
    final_val = list(map(make_square, myList))
    print("this is final list", final_val)

    newList = []
    cool = [newList.append(make_square(val)) for val in myList]
    # or
    cool = [make_square(val) for val in myList]
    print(newList)

    # **********************
    # lamda

    add_func = lambda v: v ** 2

    print("lambada function outpus is ", add_func(2))