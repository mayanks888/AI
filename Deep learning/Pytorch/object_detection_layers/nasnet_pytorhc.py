import pretrainedmodels

print(pretrainedmodels.model_names)
print(pretrainedmodels.pretrained_settings['nasnetalarge'])

model_name = 'nasnetalarge'  # could be fbresnet152 or inceptionresnetv2
model = pretrainedmodels.__dict__[model_name](num_classes=1000, pretrained='imagenet')
model.eval()

import torch
import pretrainedmodels.utils as utils

load_img = utils.LoadImage()

# transformations depending on the model
# rescale, center crop, normalize, and others (ex: ToBGR, ToRange255)
tf_img = utils.TransformImage(model)

path_img = 'cat.jpg'

input_img = load_img(path_img)
input_tensor = tf_img(input_img)  # 3x400x225 -> 3x299x299 size may differ
input_tensor = input_tensor.unsqueeze(0)  # 3x299x299 -> 1x3x299x299
input = torch.autograd.Variable(input_tensor,
                                requires_grad=False)

output_logits = model(input)
output_logit = output_logits.squeeze()
values = torch.max(output_logit)
print(values)
# output_features = model.features(input) # 1x14x14x2048 size may differ
# output_logits = model.logits(output_features)
