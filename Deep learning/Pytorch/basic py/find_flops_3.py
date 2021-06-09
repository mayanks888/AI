from torchstat import stat
from torchvision.models import resnet50, resnet101, resnet152, resnext101_32x8d
import torchvision
import torch
import time
#
# model = resnet50()
# # stat(model, (3, 224, 224))
# stat(model, (3, 224, 224))
#
###################################################
# model = torch.hub.load('pytorch/vision:v0.9.0', 'mobilenet_v2', pretrained=False)
model = torchvision.models.mobilenet_v2(pretrained=False)
# model = torchvision.models.mnasnet1_3(pretrained=False)
# model = torchvision.models.wide_resnet101_2(pretrained=False)
# model = torchvision.models.resnet50(pretrained=False)
# model.eval()
stat(model, (3, 224, 224))
# input = torch.randn(1, 3, 224, 224)
# pred=model(input)
# # print(pred)


################################################################3
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# # device = torch.device("cpu")
# model = torchvision.models.wide_resnet101_2(pretrained=False).to(device)
# # model = torchvision.models.mobilenet_v2(pretrained=False).to(device)
# # model = torchvision.models.resnet50(pretrained=False).to(device)
# model.eval()
# t2 = time.time()
# input = torch.randn(1, 3, 224, 224).to(device)
# pred=model(input)
# print("total_time",(time.time() - t2)*1000)
# # print(pred)
#######################################################################