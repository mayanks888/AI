import torch
input = torch.zeros((2))
print(input)

temp = torch.ones(1)
#temp = torch.ones((2, 1, 4, 1))

input = torch.cat((input,temp ), 0)
input = torch.cat((x[i][j],data ), 0)
print(input)