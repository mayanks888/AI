import torch
print(torch.__version__)
from torch.autograd import variable

x=[2,3,4]
y=[6,8,9]

m=torch.tensor([3],requires_grad=True)
print ('variable data',m.data)

def forward(x):
    yval=x*m
    return yval

def loss(x,yreal):
    ypred=forward(x)
    return ((ypred-yreal)*(ypred-yreal))

print(forward(4).data[0])#.data is used to show the value of a tesnor
# print(torch.arange(1,5,6))
for weight in range(2):
    print('val of w is',weight)
    total=0
    for xdata,ydata in zip(x,y):
        # ypred = forward(xdata)
        myloss=loss(xdata,ydata)
        myloss.backward()#this signifies that i can hve backward propogation
        print('gradient val',m.grad.data)
        # print('newgrad',x.grad.data[0])#'list' object has no attribute 'grad'
        # print ("loss val is:",myloss)
        total=myloss+total
    print ('total loss is',total)