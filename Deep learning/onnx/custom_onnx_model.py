import torch
from torch import nn
from torch.nn import functional as F
import numpy as np

class PFNLayer_mayank_seq_layer(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 use_norm=True,
                 last_layer=False, act=None):
        super().__init__()

        # self.last_vfe = last_layer
        self.use_norm = use_norm
        # if not self.last_vfe:
        #     out_channels = out_channels // 2

        if self.use_norm:
            self.conv_layer_1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0, bias=False)
            self.norm = nn.BatchNorm2d(out_channels, eps=1e-3, momentum=0.01)
        else:
            self.conv_layer_1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0, bias=False)
        # self.act = act()

        # if self.use_norm:
        #     self.conv_seq=nn.Sequential(
        #         nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0, bias=False),
        #         nn.BatchNorm2d(out_channels, eps=1e-3, momentum=0.01),
        #         nn.ReLU6(inplace=True)
        #     )
            # self.conv_seq = nn.Sequential(
            #     nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0, bias=False),
            #     nn.BatchNorm2d(out_channels),
            #     nn.ReLU6(inplace=True)
            # )

    # def forward(self, inputs):
    #     # x = self.conv_layer_1(inputs)
    #     # batch_norm = self.norm(x)
    #     # # x=nn.ReLU(x)
    #     # act_relu = F.relu(batch_norm)
    #     # x = self.act(x)
    #     x=self.conv_seq(inputs)
    #     # h, w = x.shape[2:]
    #     h,w=12000,60
    #     x_max = F.max_pool2d(x, (1, int(w)))
    #     return x_max

    def forward(self, inputs):
        x = self.conv_layer_1(inputs)
        x = self.norm(x)
        # x=nn.ReLU(x)
        x = F.relu(x)
        # x = self.act(x)
        h, w = x.shape[2:]
        x_max = F.max_pool2d(x, (1, int(w)))

        return x_max


if __name__ =antut = XXX # You need to provide the correct input here!
    # dummy_input = 608 # You need to provide the correct input here!
    # imgsz = 608
    dummy_input = torch.zeros((1, 9, 12000, 60))  # You need to provide the correct input here!

    # Check it's valid by running the PyTorch model
    # dummy_output = model(dummy_input)
    print("converting onnx")

    # If the input is valid, then exporting should work

    # torch.onnx.export(model, dummy_input, "pfe_custom_wo_weight.onnx" )
    torch.onnx.export(model, dummy_input, "pfe_custom_weight.onnx",export_params=True )