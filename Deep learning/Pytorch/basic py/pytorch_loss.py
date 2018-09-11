import torch
import torch.nn.functional as F
# input is of size N x C = 3 x 5
input = torch.arange(8).reshape(1, 2, 4)#nice way to iniatalse array
print(input.shape)


input1 = torch.randn(3, 5, requires_grad=True)
# each element in target has to have 0 <= value < C
target = torch.tensor([1, 0, 4])
cool=F.log_softmax(input1)
output = F.nll_loss(cool, target)

print ("the value",output.data[0])
output.backward()