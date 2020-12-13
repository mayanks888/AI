import torchvision.models as models
import torch
model = models.vgg19(pretrained=False)

# model.state_dict(''/home/mayank_s/Desktop/template/model/vgg19-dcbb9e9d.pth'')
model.load_state_dict(torch.load('/home/mayank_s/Desktop/template/model/vgg19-dcbb9e9d.pth'))
print(1)