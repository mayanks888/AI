import onnx
import torch

# import onnx_tensorrt.backend as backend

model = onnx.load(
    "/home/mayank_sati/pycharm_projects/pytorch/second .pytorch_traveller59_date_9_05_experiment_mode/second/pytorch/models/mayank_rpn.onnx")
print(onnx.checker.check_model(model))

# Print a human readable representation of the graph
print(onnx.helper.printable_graph(model.graph))
x = torch.randn(1, 64, 400, 400)
# cd = torch.randn(8500, 4, device='cuda')

cn = model(x=x)
print(cn)
#
# engine = backend.prepare(model, device='CUDA:1')
# # input_data = np.random.random(size=(32, 3, 224, 224)).astype(np.float32)
# input_data = torch.randn(1, 64,400,400)
# output_data = engine.run(input_data)[0]
# print(output_data)
# print(output_data.shape)
