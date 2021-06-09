
# from torchvision.models import resnet50
# from thop import profile
# model = resnet50()
# flops, params = profile(model,(1, 3, 224,224))


from torchvision.models import resnet50
from thop import profile
import torch
model = resnet50()
input = torch.randn(1, 3, 224, 224)
flops, params = profile(model, inputs=(input, ))
print(params)