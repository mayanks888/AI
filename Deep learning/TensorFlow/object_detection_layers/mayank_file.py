import anchor_target_layer
import data_handler
import region_proposal_network
import numpy as np
import rpn_softmax
#
# label_gt=np.array([[0,182,228,374,1],
#         [692,185,887,294,1],
#         [362,159,382,221,4],
#         [315,178,422,251,1],
#         [363,178,441,230,1],
#         [606,179,683,230,1],
#         [589,178,643,218,1],
#         [402,178,457,220,1],
#         [534,175,578,204,1]])
#
# label2=np.array([1,2,3,4])
#
# # print(label_gt)
#
# image_dim=[[ 375, 1242]]

# print(label_gt,image_dim)
rpn=[]
feat_stride        = 16
anchor_scale       = [ 8, 16, 32 ]
data_handler = data_handler.DataHandler()
data = "../../Datasets/kitti/train.txt"
label_file = "../../Datasets/kitti/label.txt"
data_handler.get_file_list(data, label_file)
data1, labels1, ims = data_handler.load_data(0, 1)
print(data1.shape, labels1.shape, ims.shape)
print(labels1, ims)

# rpn = region_proposal_network.RegionProposalNetwork(feature_vector=data1,ground_truth=labels1,im_dims=ims,anchor_scale=anchor_scale,Mode="train")
rpn_label,rpn_bbox_targets,rpn_bbox_inside_weights,rpn_bbox_outside_weights=anchor_target_layer.anchor_target_layer_python(rpn_cls_score=data1,gt_boxes=labels1,im_dims=ims,feat_strides=feat_stride,anchor_scales=anchor_scale)
(rpn_label)

#rpn_softmax.rpn_soft