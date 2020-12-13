import numpy as np

w = 2

x = [2, 3, 4]
y = [6, 8, 9]


def forward(x, m):
    yval = x * m
    return yval


def loss(ypred, yreal):
    return ((ypred - yreal) * (ypred - yreal))


ypred = forward(x, w)
loss = loss(ypred, y)

print(loss)

for weight in np.arange(1, 5, 3):
    print('val of w is', weight)
    total = 0
    for xdata, ydata in zip(x, y):
        ypred = forward(xdata, weight)
        myloss = loss(ypred, ydata)
        # print ("loss val is:",myloss)
        total = myloss + total
    print('total loss is', total)
