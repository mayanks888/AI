import anchor_target_layer
import numpy as np
import read_pascal_voc
from batchup import data_source
rpn=[]
feat_stride        = 16
anchor_scale       = [ 8, 16, 32 ]
imageNameFile = "../../../Datasets/VOCdevkit/VOC2012/ImageSets/Main/aeroplane_train.txt"
vocPath       = "../../../Datasets/VOCdevkit/VOC2012"

Image_data,boundingBX_labels,im_dims=read_pascal_voc.prepareBatch(0,5,imageNameFile,vocPath)
print(Image_data,boundingBX_labels,im_dims)

epochs=1
for loop in range(epochs):

    # ______________________________________________________________________
    # my batch creater
    # Construct an array data source
    ds = data_source.ArrayDataSource([Image_data, boundingBX_labels,im_dims])
    # Iterate over samples, drawing batches of 64 elements in random order
    for (image_input, gt_box,image_dim) in ds.batch_iterator(batch_size=1, shuffle=True):#shuffle true will randomise every batch
        # accuoutput=sess.run([rpn], feed_dict={input_x: image_input, gt_bbox: gt_box, im_dimsal: image_dim})
        _label, rpn_bbox_targets, rpn_bbox_inside_weights, rpn_bbox_outside_weights = anchor_target_layer.anchor_target_layer_python(
            rpn_cls_score=image_input, gt_boxes=gt_box, im_dims=image_dim, feat_strides=feat_stride,
            anchor_scales=anchor_scale)

# (rpn_label)
