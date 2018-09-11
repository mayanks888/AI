# region_proposal_network.py
import tensorflow as tf
import numpy as np
import anchor_target_layer 

# class RegionProposalNetwork(object):
#     """
#     From convolutional feature map, generate bounding box, relative to anchor point, and give an objectness score to each candidate
# 
#     """
# 
#     def __init__(, feature_vector, ground_truth, im_dims, anchor_scale, Mode):
#         .feature_vector     = feature_vector
#         .ground_truth       = ground_truth
#         .im_dims            = im_dims
#         .anchor_scale       = anchor_scale
# 
#         .RPN_OUTPUT_CHANNEL = 512
#         .RPN_KERNEL_SIZE    = 3
#         .feat_stride        = 16
# 
#         .weights            = {
#         'w_rpn_conv1'     : tf.Variable(tf.random_normal([ .RPN_KERNEL_SIZE, .RPN_KERNEL_SIZE, 512, .RPN_OUTPUT_CHANNEL ], stddev = 0.01)),
#         'w_rpn_cls_score' : tf.Variable(tf.random_normal([ 1, 1, .RPN_OUTPUT_CHANNEL, 18  ], stddev = 0.01)),
#         'w_rpn_bbox_pred' : tf.Variable(tf.random_normal([ 1, 1, .RPN_OUTPUT_CHANNEL, 36  ], stddev = 0.01))
#         }
# 
#         .mode               = Mode # train or test
# 
#         .build_rpn()       
#   

# def build_rpn():
#
#     # rpn_conv1
#     # slide a network on the feature map, for each nxn (n = 3), use a conv kernel to produce another feature map.
#     # each pixel in this fature map in an anchor
#     ksize      = .RPN_KERNEL_SIZE
#     feat       = tf.nn.conv2d( .feature_vector, .weights['w_rpn_conv1'], strides = [1, 1, 1, 1], padding = 'SAME' )
#     feat       = tf.nn.relu( feat )
#     .feat  = feat
#
#     # for each anchor, propose k anchor boxes,
#     # for each box, regress: objectness score and coordinates
#
#     # box-classification layer ( objectness scor)
#     with tf.variable_scope('cls'):
#         .rpn_cls_score = tf.nn.conv2d(feat, .weights['w_rpn_cls_score'], strides = [ 1, 1, 1, 1], padding = 'SAME')
#
#     # bounding-box prediction
#     with tf.variable_scope('reg'):
#         .rpn_reg_pred  = tf.nn.conv2d(feat, .weights['w_rpn_bbox_pred'], strides = [1, 1, 1, 1], padding = 'SAME')
#
#     # Anchor Target Layer ( anchor and delta )
#     # rpn_bbox_inside_weights(Nbox) and rpn_bbox_outside_weights(iLcls) is normalisation factor for classification and regression boxes
#     # LL({pi}, {ti}) = Lcls + Lbox = 1 Ncls∑iLcls(pi, p∗i)+λNbox∑ip∗i⋅Lsmooth1(ti−t∗i)
#
#     with tf.variable_scope('anchor'):
#         if .mode == 'train':
#             .rpn_labels, .rpn_bbox_targets, .rpn_bbox_inside_weights, .rpn_bbox_outside_weights = \
#                 anchor_target_layer.anchor_target_layer( .rpn_cls_score, .ground_truth, .im_dims, .feat_stride, .anchor_scale )
#

# def get_rpn_input_feature():
#     return .feat
#
# def get_rpn_cls_score():
#     return .rpn_cls_score
#
# def get_rpn_labels():
#     return .rpn_labels
#
# def get_rpn_bbox_pred():
#     return .rpn_reg_pred
#
# def get_rpn_bbox_targets():
#     return .rpn_bbox_targets
#
# def get_rpn_bbox_inside_weights():
#     return .rpn_bbox_inside_weights
#
# def get_rpn_bbox_outside_weights():
#     return .rpn_bbox_outside_weights
#
# def get_rpn_bbox_loss():
#     rpn_bbox_pred            = .get_rpn_bbox_pred()
#     rpn_bbox_targets         = .get_rpn_bbox_targets()
#     rpn_bbox_inside_weights  = .get_rpn_bbox_inside_weights()
#     rpn_bbox_outside_weights = .get_rpn_bbox_outside_weights()
#     return .rpn_bbox_loss( rpn_bbox_pred, rpn_bbox_targets, rpn_bbox_inside_weights, rpn_bbox_outside_weights)

# def get_rpn_cls_loss():
#     rpn_cls_score            = get_rpn_cls_score()
#     rpn_labels               = get_rpn_labels()
#     return rpn_cls_loss(rpn_cls_score, rpn_labels)

def rpn_cls_loss(rpn_cls_score, rpn_labels):

    shape                     = tf.shape(rpn_cls_score)

    # Stack all classification scores into 2D matrix
    rpn_cls_score             = tf.transpose(rpn_cls_score,[0,3,1,2])
    rpn_cls_score             = tf.reshape(rpn_cls_score,[shape[0],2,shape[3]//2*shape[1],shape[2]])
    rpn_cls_score             = tf.transpose(rpn_cls_score,[0,2,3,1])
    rpn_cls_score             = tf.reshape(rpn_cls_score,[-1,2])

    # Stack labels
    rpn_labels                = tf.reshape(rpn_labels,[-1])

    # Ignore label=-1 (Neither object nor background: IoU between 0.3 and 0.7)
    rpn_cls_score             = tf.reshape(tf.gather(rpn_cls_score,tf.where(tf.not_equal(rpn_labels,-1))),[-1,2])
    rpn_labels                = tf.reshape(tf.gather(rpn_labels,tf.where(tf.not_equal(rpn_labels,-1))),[-1])
    # Cross entropy error
    rpn_cross_entropy         = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=rpn_cls_score, labels=rpn_labels))

    return rpn_cross_entropy


def rpn_bbox_loss(rpn_bbox_pred, rpn_bbox_targets, rpn_inside_weights, rpn_outside_weights):

    rpn_bbox_targets          = tf.transpose( rpn_bbox_targets,   [ 0, 2, 3, 1])
    rpn_inside_weights        = tf.transpose( rpn_inside_weights, [ 0, 2, 3, 1])
    rpn_outside_weights       = tf.transpose( rpn_outside_weights,[ 0, 2, 3, 1])

    sigma=3
    diff_sL1=(modified_smooth_l1(sigma, rpn_bbox_pred, rpn_bbox_targets, rpn_inside_weights, rpn_outside_weights))

    # diff                      = tf.multiply( rpn_inside_weights, rpn_bbox_pred - rpn_bbox_targets)
    # print(rpn_bbox_pred,'\n',rpn_bbox_targets)
    # print ("my diff value was",diff)
    # diff_sL1                  = smoothL1(diff,3.0)
    # # diff_sL1=1
    rpn_bbox_reg              = 10*tf.reduce_sum(tf.multiply(rpn_outside_weights, diff_sL1))
    return rpn_bbox_reg

def smoothL1(x, sigma):
    conditional               = tf.less(tf.abs(x), 1/sigma**2)
    close                     = 0.5* (sigma * 2 ) **2
    far                       = tf.abs(x) - 0.5/sigma ** 2

    return tf.where(conditional, close, far)

def modified_smooth_l1(sigma, bbox_pred, bbox_targets, bbox_inside_weights, bbox_outside_weights):
        """
            ResultLoss = outside_weights * SmoothL1(inside_weights * (bbox_pred - bbox_targets))
            SmoothL1(x) = 0.5 * (sigma * x)^2,    if |x| < 1 / sigma^2
                          |x| - 0.5 / sigma^2,    otherwise
        """
        sigma2 = sigma * sigma

        inside_mul = tf.multiply(bbox_inside_weights, tf.subtract(bbox_pred, bbox_targets))

        smooth_l1_sign = tf.cast(tf.less(tf.abs(inside_mul), 1.0 / sigma2), tf.float32)
        smooth_l1_option1 = tf.multiply(tf.multiply(inside_mul, inside_mul), 0.5 * sigma2)
        smooth_l1_option2 = tf.subtract(tf.abs(inside_mul), 0.5 / sigma2)
        smooth_l1_result = tf.add(tf.multiply(smooth_l1_option1, smooth_l1_sign),
                                  tf.multiply(smooth_l1_option2, tf.abs(tf.subtract(smooth_l1_sign, 1.0))))

        outside_mul = tf.multiply(bbox_outside_weights, smooth_l1_result)

        return outside_mul
