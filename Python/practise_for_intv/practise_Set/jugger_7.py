'''Problem 7:

Write a program which takes 2 digits, X,Y as input and generates a 2-dimensional array. The element value in the i-th row and j-th column of the array should be i _ j.*

Note: i=0,1.., X-1; j=0,1,¡­Y-1. Suppose the following inputs are given to the program: 3,5

Then, the output of the program should be:

[[0, 0, 0, 0, 0], [0, 1, 2, 3, 4], [0, 2, 4, 6, 8]]'''


val=(3,5)
print(val[0])
inp=-1
for a in range(val[0]):
    inp+=1
    for b in range(val[1]):
        print(b*inp,end="")
    print()












# x,y = map(int,input().split(','))
# lst = []
#
# for i in range(x):
#     tmp = []
#     for j in range(y):
#         tmp.append(i*j)
#     lst.append(tmp)
#
# print(lst)