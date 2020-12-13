import torch
import torchvision

# No CUDA
# dummy_input = torch.randn(10, 3, 224, 224)
# model = torchvision.models.resnet18(pretrained=True)

# CUDA
dummy_input = torch.randn(10, 3, 224, 224, device='cuda')
model = torchvision.models.alexnet(pretrained=True).cuda()

input_names = ["actual_input_1"] + ["learned_%d" % i for i in range(16)]
output_names = ["output1"]

torch.onnx.export(model, dummy_input, "resnet18.onnx", verbose=True, input_names=input_names, output_names=output_names)
