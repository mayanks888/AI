import os
def checkingGeneator(num):
    print("starting")
    while(True):
        yield num
        num+=1

cd=checkingGeneator(4)
print(cd)
print(next(cd))
print(next(cd))
print(next(cd))
print(next(cd))

#fibonachi in generator

def myfibonachi(limit):
    a, b=0,1
    while(a<limit):
        yield a
        a,b=b,b+a

if __name__ == '__main__':
    fs = myfibonachi(10)
    for i in fs:
        print(i)
