import torch


tor=torch.ones(2,2)
print(tor.data)
tor.cuda()#remember this convert cpu to cuda
print(tor.data)
tor.cpu#this will convert back to cpu
print(tor.data)