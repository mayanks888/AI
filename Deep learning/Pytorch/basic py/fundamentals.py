import torch
from torch.autograd import variable#now its option function (torch.tensor does the jobs)

data=[1.0,2,26,28,4]

b=variable(data) #there is no concepts of variable now  so forget it
print('the variable is',b)
c=torch.tensor(data, requires_grad=True)#they both are same
print('c is torch tesnor',c)
print('cdata is',c.data)


'''# print("the data of tensor CC is ", c.data)
print("the data of tensor B is ", b.data[:])
e=c.data[3]
print(float(e))#in order to conver one elemet from tensor to scaler we can use float(val) ot int(val) :val=tensor
print('eitem is',e.item())#or simply use item () to for convering into scaler


f= c.tolist()#conver tesnor list to common list
print('list conversion',f)

g=c.numpy()#conver tensor to common array list
print("numpy conversion",g)'''

#I came to know the differnce between Torch.tensor and Variable

print(c.grad)#find the grad of the c(differentiation of c)


# ***************************************************************
# understanding the gradient  in pytorch
# f(x)= 4x+2
# df/dx= 4(differnetion of (4x+2 is 4))

def funtion_to_check_grad(x):
    return (4*x+2)


val=torch.tensor(3.0,requires_grad=True)#req grad mean in need grad of val value.

val2=funtion_to_check_grad(val)

val2.backward()#here the actaul differnetion will happed

print('the gradient of val value is ',val.grad)#understand gradient of val will be calculated and not val2)

# ***************************************************************