import torch

# x = torch.tensor([[1], [2], [3]])
x = torch.tensor([1,4,5,4])
print(x.size())
# torch.Size([3, 1])
dat=x.expand(3, 4)

print(dat.size())
out=(4,256)
x=torch.tensor([288.1084, 253.1771, 219.4308, 275.2784])
print(x.size())
# torch.Size([3, 1])
dat=x.expand(256,4)

print(dat.size())