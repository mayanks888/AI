import random
import torch
import torch.nn.functional as F
action= 4

myaction=random.randint(1, action)
print(myaction)
#
# expected=torch.tensor([1.1088, 1.1231, 1.3270, 1.5933, 1.0418, 1.2354, 0.9954, 1.0418, 1.4347,
#         0.9416, 1.0450, 1.4721, 1.1185, 1.1303, 1.1096],)
#        grad_fn=<ThAddBackward>)
#
# state_vale=torch.tensor([-0.1566, -0.0804,  0.1751,  0.1374, -0.0302, -0.0302, -0.0302, -0.0416,
#          0.5946, -0.0939,  0.1188,  0.5774,  0.1284,  0.1384,  0.0552],

expected=torch.tensor([1.1088, 1.1231, 1.3270, 1.5933, 1.0418, 1.2354, 0.9954, 1.0418, 1.4347,
        0.9416, 1.0450, 1.4721, 1.1185, 1.1303, 1.1096], requires_grad=True)

state_vale=torch.tensor([-0.1566, -0.0804,  0.1751,  0.1374, -0.0302, -0.0302, -0.0302, -0.0416,
         0.5946, -0.0939,  0.1188,  0.5774,  0.1284,  0.1384,  0.0552])

loss=F.smooth_l1_loss(expected,state_vale)
print(loss)

import torch
from torch.autograd import Variable
import torch.nn.functional as F

x = Variable(torch.randn(2, 2), requires_grad=True)
t = Variable(torch.randn(2, 2), requires_grad=False)
print(F.smooth_l1_loss(x, t, reduce=True))