
import torch
import torch.nn as nn
import torchvision
import create_model
import read_pascal_voc
import numpy as np
import region_propsal_Network
import anchor_target_layer
rpn=[]
feat_stride        = 16
anchor_scale       = [ 8, 16, 32 ]
Mode="train"
imageNameFile = "../../../Datasets/VOCdevkit/VOC2012/ImageSets/Main/aeroplane_train.txt"
vocPath       = "../../../Datasets/VOCdevkit/VOC2012"
import torch.optim as optim


# Image_data,boundingBX_labels,im_dims=read_pascal_voc.prepareBatch(0,3,imageNameFile,vocPath)

class RPN(nn.Module):
    def __init__(self):
        super(RPN, self).__init__()

        self.generate_model=create_model.Gen_model()
        self.rpn_net=region_propsal_Network.rpn(512,512)
        # self.image_data=read_pascal_voc.prepareBatch()#0,3,imageNameFile,vocPath)

    def forward(self,data_image):
        #data_image=read_pascal_voc.prepareBatch(0,3,imageNameFile,vocPath)
        features=self.generate_model(data_image)
        return features

    def run_rpn(self,features, image_size):
        # data=self.rpn_net()
        delta= self.rpn_net.forward(features, image_size)
        return delta

    def find_loss(self,delta, score, anchor, gt_bbox, image_size):
        loss_delta=self.rpn_net.loss(delta, score, anchor, gt_bbox, image_size)
        return loss_delta



model = RPN()
# optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.5)


data_image=read_pascal_voc.prepareBatch(0,1,imageNameFile,vocPath)
image=np.array(data_image[0])
image=np.swapaxes(image, 1, 3)
image_tensor=torch.from_numpy(image)
image_siz=data_image[2]
# gt_bbox=torch.from_numpy(data_image[1])
gt_bbox=data_image[1]
image_size=torch.from_numpy(image_siz[0])
data=torch.tensor((1000,600))


output = model.forward(image_tensor)
# optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.5)

print(output.shape)

pred_reg_delta, pred_score, anchor=model.run_rpn(output,data)
print(pred_score.shape)
print(pred_reg_delta.shape)
# print(out[0])
# Image_data,boundingBX_labels,im_dims=data_image(0,3,imageNameFile,vocPath)
# model.rpn_net.loss(delta, score, anchor, gt_bbox, image_size)
total_loss=model.find_loss(pred_reg_delta, pred_score, anchor, gt_bbox, image_size)
print(total_loss)

total_loss.backward()
# optimizer.zero_grad()
# # total_loss.backward()
# optimizer.step()

#rpn_labels,rpn_bbox_targets,rpn_bbox_inside_weights,rpn_bbox_outside_weights=anchor_target_layer.anchor_target_layer_python(output, gt_bbox, image_siz, feat_stride, anchor_scale)