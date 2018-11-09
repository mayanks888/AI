import torch
from torch.autograd import Variable


# ***************************************************************
# understanding the gradient  in pytorch
# f(x)= 4x+2
# df/dx= 4(differnetion of (4x+2 is 4))

def funtion_to_check_grad(x):
    return (4*x+2)

def funtion_to_check_grad2(x):
    return (3*x)

val=torch.tensor(3.0,requires_grad=True)#req grad mean in need grad of val value.

val2=funtion_to_check_grad(val)

val3=funtion_to_check_grad2(val2)

val3.backward()#here the actaul differnetion will happed


print('the gradient of val value is ',val.grad)#understand gradient of val will be calculated and not val2)

# ****************


x_data = [1.0, 2.0, 3.0]
y_data = [2.0, 4.0, 6.0]

w = torch.tensor(3.0,requires_grad=True)  # Any random value

# our model forward pass


def forward(x):
    return x * w

# Loss function


def loss(x, y):
    y_pred = forward(x)
    return (y_pred - y) * (y_pred - y)

# Before training
print("predict (before training)",  4, forward(4).data[0])

# Training loop
for epoch in range(10):
    for x_val, y_val in zip(x_data, y_data):
        l = loss(x_val, y_val)
        l.backward()
        print("\tgrad: ", x_val, y_val, w.grad.data[0])
        w.data = w.data - 0.01 * w.grad.data

        # zero the gradients after updating weights
        w.grad.data.zero_()

    print("progress:", epoch, l.data[0])

# After training
print("predict (after training)",  4, forward(4).data[0])
