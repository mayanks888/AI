import torch
import torch.onnx
# path="/home/mayank_s/codebase/others/yolo/yolov4/yolov3/mayank_script/yolo_v4_mayank.pt"
path="/home/mayank_s/codebase/others/yolo/yolov4/yolov3_4/weights/yolov3-spp-ultralytics.pt"
model = torch.load(path)

# dummy_input = XXX # You need to provide the correct input here!
# dummy_input = 608 # You need to provide the correct input here!
imgsz=608
dummy_input = torch.zeros((16, 3, imgsz, imgsz), device="cuda") # You need to provide the correct input here!

# Check it's valid by running the PyTorch model
# dummy_output = model(dummy_input)
print("Input is valid")

# If the input is valid, then exporting should work

torch.onnx.export(model, dummy_input, "pytorch_model.onnx")
# torch.onnx.export(model, dummy_input, path)