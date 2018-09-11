# import the necessary packages
from collections import namedtuple
import numpy as np
import cv2
import  tensorflow as tf
# define the `Detection` object
Detection = namedtuple("Detection", ["image_path", "gt", "pred"])

def bb_intersection_over_union(boxA, boxB):
    # determine the (x, y)-coordinates of the intersection rectangle
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    # compute the area of intersection rectangle
    # interArea = (xB - xA) * (yB - yA)
    interArea = max((xB - xA), 0) * max((yB - yA), 0)
    # compute the area of both the prediction and ground-truth
    # rectangles
    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])

    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = interArea / float(boxAArea + boxBArea - interArea)

    # return the intersection over union value
    return iou





# ground=[106,118,942,570]
# # ground=[316,59,578,226]
# predicted=[784,504,931,599]
#
#
# # predicted=[80,60,200,158]
# ground=np.array(ground)
# predicted=np.array(predicted)
# print(bb_intersection_over_union(predicted,ground))

# print(tf.metrics.mean_iou(labels=ground,predictions=predicted,num_classes=1))