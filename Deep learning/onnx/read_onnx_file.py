# # from onnx_tf.backend import prepare
# model = torch.onnx.load('mayank.onnx') # Load the ONNX file
# print(1)
# # tf_rep = prepare(model)


import onnx

# Load the ONNX model
model = onnx.load("mayank.onnx")

# Check that the IR is well formed
onnx.checker.check_model(model)

# Print a human readable representation of the graph
print(onnx.helper.printable_graph(model.graph))
print(1)
