import numpy as np
K=np
# from keras import backend as K

def box_overlap(anchorbb,groundbb):
    overlap=[]
    N=anchorbb.shape[0]
    K=groundbb.shape[0]
    for n in range(N):
        for k in range(K):
            overlap[n,k]=find_iou(groundbb(k),anchorbb(n))





def find_iou(groundbb, predicted_bb):
    data_ground = groundbb
    data_predicted = predicted_bb
    xminofmax = np.maximum(data_ground[0], data_predicted[0])
    yminofmax = np.maximum(data_ground[1], data_predicted[1])
    xmaxofmin = np.minimum(data_ground[2], data_predicted[2])
    ymaxofmin = np.minimum(data_ground[3], data_predicted[3])

    interction = ((xmaxofmin - xminofmax + 1) * (ymaxofmin - yminofmax + 1))
#   i have added 1 to all the equation save the equation from giving 0 iou value
#    AOG: area of ground box
    AOG = (np.abs(data_ground[0] - data_ground[2]) + 1) * (np.abs(data_ground[1] - data_ground[3]) + 1)
    #AOP:area of predicted box
    AOP = (np.abs(data_predicted[0] - data_predicted[2]) + 1) * (np.abs(data_predicted[1] - data_predicted[3]) + 1)
    union= (AOG + AOP) - interction
    iou = (interction /union)
    mean_iou = np.mean(iou)
    return (iou, mean_iou)




def iou_metric(y_true, y_pred):
    # iou as metric for bounding box regression
    # input must be as [x1, y1, x2, y2]

    # AOG = Area of Groundtruth box
    AoG = K.abs(K.transpose(y_true)[2] - K.transpose(y_true)[0]+1 ) * K.abs(
        K.transpose(y_true)[3] - K.transpose(y_true)[1] + 1)
    # print ("the valof AOG",AoG)
    # AOP = Area of Predicted box
    AoP = K.abs(K.transpose(y_pred)[2] - K.transpose(y_pred)[0] +1) * K.abs(
        K.transpose(y_pred)[3] - K.transpose(y_pred)[1] + 1)

    # overlaps are the co-ordinates of intersection box
    overlap_0 = K.maximum(K.transpose(y_true)[0], K.transpose(y_pred)[0])
    overlap_1 = K.maximum(K.transpose(y_true)[1], K.transpose(y_pred)[1])
    overlap_2 = K.minimum(K.transpose(y_true)[2], K.transpose(y_pred)[2])
    overlap_3 = K.minimum(K.transpose(y_true)[3], K.transpose(y_pred)[3])

    # intersection area
    intersection = (overlap_2 - overlap_0 + 1) * (overlap_3 - overlap_1 + 1)

    # area of union of both boxes
    union = AoG + AoP - intersection

    # iou calculation
    iou = intersection / union

    # bounding values of iou to (0,1)
    # iou = K.clip(iou, 0.0 + K.epsilon(), 1.0 - K.epsilon())
    return iou

def find_iou_for_frcnn(groundbb, predicted_bb):
    data_ground = groundbb
    data_predicted = predicted_bb
    xminofmax = np.maximum(data_ground[0], data_predicted[0])
    yminofmax = np.maximum(data_ground[1], data_predicted[1])
    xmaxofmin = np.minimum(data_ground[2], data_predicted[2])
    ymaxofmin = np.minimum(data_ground[3], data_predicted[3])

    interction = ((xmaxofmin - xminofmax + 1) * (ymaxofmin - yminofmax + 1))
#   i have added 1 to all the equation save the equation from giving 0 iou value
#    AOG: area of ground box
    AOG = (np.abs(data_ground[0] - data_ground[2]) + 1) * (np.abs(data_ground[1] - data_ground[3]) + 1)
    #AOP:area of predicted box
    AOP = (np.abs(data_predicted[0] - data_predicted[2]) + 1) * (np.abs(data_predicted[1] - data_predicted[3]) + 1)
    union= (AOG + AOP) - interction
    iou = (interction /union)
    mean_iou = np.mean(iou)
    return (iou, mean_iou)


# data1=np.random.randint(low=10,high=50,size=(4,3))
data_ground=np.array([[13,15,16],[12,14,15],[17,19,20],[20,30,35]])
data_predicted=np.array([[15,19,20],[14,16,14],[19,22,25],[21,34,32]])
data_ground=(np.array([13,15,16]))
data_predicted=(np.array([15,19,20]))
data_ground=(np.array([13,12,17,20]))
data_predicted=(np.array([13,12,17,20]))
#data_ground=np.array([13,15,16,22])
#data_predicted=np.array([15,19,17,25])ascontiguousarray

print(find_iou(data_ground,data_predicted))


data_ground=np.transpose(np.array([13,12,17,20]))
data_predicted=np.transpose(np.array([13,12,17,20]))
print(iou_metric(data_ground,data_predicted))
