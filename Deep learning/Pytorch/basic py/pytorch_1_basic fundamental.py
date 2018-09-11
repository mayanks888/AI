import numpy as np
import torch

data=[[1,2],[2,3]]#this is a list
# print (data.type)
ndata=np.array(data)
# print(ndata.type)

data=np.ones(shape=(2,2))
print(data)

#now is tensor
tor=torch.ones(size=(2,2))
print(tor.data)

data=np.random.rand(2,2)
print(data)

tor=torch.rand(2,2)
print(tor.data)
####################
# for putting seeed
torch.manual_seed(0)
print(torch.rand(2,2))
####################
#numpy to torch bridge
data=np.ones(shape=(2,2))
print(type(data))#this is to show that its is type is numpy

#now changing to numpy to tensor
tensor_ten=torch.from_numpy(data)#this is most important remember this
print(tensor_ten.data)



#torch sensor to numpy array
tor=torch.rand(3,3)
print(tor.type)
ndata=tor.numpy()#this command convert torch tensor to numpy
print(type(ndata))


#this is to change the shape of tensor
tor=torch.ones(4,1)
print(tor.shape)
print(tor.data)
print(tor.view(1,4))#this is excellent function about the resize
new_dim_tor=tor.view(4)#this will convert the [4,1] to [4]ie 2d to one d
print(new_dim_tor.shape)

###################33
# inplace tensor

tor1=torch.ones(2,2)
tor2=torch.tensor([[2,2],[2,2]]).float()
tor1.add_(tor2)#this is inplaces tor1 to new value(ie replace the old value with new one)
print(tor1.data)

#torch mean

tor=torch.linspace(0,10,10)
data=np.random.randint(low=0,high=50,size=(5,5))
print(data)
tor=torch.from_numpy(data).float()#convert numpy to torch tensor
print(tor)
print(tor.mean(dim=0))#dim=0 meams  column wise means
print(tor.mean(dim=1))#dim=0 means row wise

