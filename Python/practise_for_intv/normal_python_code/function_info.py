

def add_val(a, b):
    return a+b

#pasing multiple parameter with *values
def sum_of_all(name, *values):
    for val in values:
        print(val)
    print(values)

if __name__ == '__main__':
    print(add_val(4,5))
    print(sum_of_all("mayan",3,5,7,87))