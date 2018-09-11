import torch
import torch.nn as nn
import torchvision

class Gen_model(nn.Module):
    def __init__(self):
        super(Gen_model, self).__init__()
        vggnet = torchvision.models.vgg16(pretrained=True)
        modules = list(vggnet.children())[:-1]  # delete the last fc layer.
        modules = list(modules[0])[:-1]  # delete the last pooling layer

        self.vggnet = nn.Sequential(*modules)

        for module in list(self.vggnet.children())[:10]:
            print("fix weight", module)
            for param in module.parameters():
                param.requires_grad = False

    def forward(self, images):
        """Extract the image feature vectors."""

        # return features in relu5_3
        features = self.vggnet(images)
        return features
#
# model = Gen_model()
#
# data=torch.ones((1,3,1000,600))
# output = model(data)
# print(output.shape)