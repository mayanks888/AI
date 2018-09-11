import tensorflow as tf
import numpy as np
from tensorflow.python import debug as tf_debug
def bbox_iou_corner_xy(bboxes1, bboxes2):
    """
    Args:
        bboxes1: shape (total_bboxes1, 4)
            with x1, y1, x2, y2 point order.
        bboxes2: shape (total_bboxes2, 4)
            with x1, y1, x2, y2 point order.

        p1 *-----
           |     |
           |_____* p2

    Returns:
        Tensor with shape (total_bboxes1, total_bboxes2)
        with the IoU (intersection over union) of bboxes1[i] and bboxes2[j]
        in [i, j].
    """

    x11, y11, x12, y12 = tf.split(bboxes1, 4, axis=1)
    x21, y21, x22, y22 = tf.split(bboxes2, 4, axis=1)

    xI1 = tf.maximum(x11, tf.transpose(x21))
    xI2 = tf.minimum(x12, tf.transpose(x22))

    yI1 = tf.minimum(y11, tf.transpose(y21))
    yI2 = tf.maximum(y12, tf.transpose(y22))

    inter_area = tf.maximum((xI2 - xI1), 0) * tf.maximum((yI1 - yI2), 0)

    bboxes1_area = (x12 - x11) * (y11 - y12)
    bboxes2_area = (x22 - x21) * (y21 - y22)

    union = (bboxes1_area + tf.transpose(bboxes2_area)) - inter_area

    # some invalid boxes should have iou of 0 instead of NaN
    # If inter_area is 0, then this result will be 0; if inter_area is
    # not 0, then union is not too, therefore adding a epsilon is OK.
    return inter_area / (union+0.0001)


def bbox_overlap_iou(bboxes1, bboxes2):
    """
    Args:
        bboxes1: shape (total_bboxes1, 4)
            with x1, y1, x2, y2 point order.
        bboxes2: shape (total_bboxes2, 4)
            with x1, y1, x2, y2 point order.

        p1 *-----
           |     |
           |_____* p2

    Returns:
        Tensor with shape (total_bboxes1, total_bboxes2)
        with the IoU (intersection over union) of bboxes1[i] and bboxes2[j]
        in [i, j].
    """

    x11, y11, x12, y12 = tf.split(bboxes1, 4, axis=1)
    x21, y21, x22, y22 = tf.split(bboxes2, 4, axis=1)

    xI1 = tf.maximum(x11, tf.transpose(x21))
    yI1 = tf.maximum(y11, tf.transpose(y21))

    xI2 = tf.minimum(x12, tf.transpose(x22))
    yI2 = tf.minimum(y12, tf.transpose(y22))

    inter_area = (xI2 - xI1 + 1) * (yI2 - yI1 + 1)

    bboxes1_area = (x12 - x11 + 1) * (y12 - y11 + 1)
    bboxes2_area = (x22 - x21 + 1) * (y22 - y21 + 1)

    return inter_area / ((bboxes1_area + tf.transpose(bboxes2_area)) - inter_area)

if __name__ == '__main__':
    bboxes1 = tf.placeholder(tf.float32)
    bboxes2 = tf.placeholder(tf.float32)
    overlap_op = bbox_iou_corner_xy(bboxes1, bboxes2)
    overlap_op1 = bbox_overlap_iou(bboxes1, bboxes2)

    # bboxes1_vals = [[39, 63, 203, 112], [0, 0, 10, 10],[5,6,77,8]]
    # bboxes2_vals = [[3, 4, 24, 32], [54, 66, 198, 114], [6, 7, 60, 44]]

    bboxes1_vals = [[39, 63, 203, 112],[39, 63, 203, 112]]
    bboxes2_vals = [[35, 60, 200, 110],[35, 60, 200, 110]]

    with tf.Session() as sess:
        overlap = sess.run(overlap_op, feed_dict={
            bboxes1: np.array(bboxes1_vals),
            bboxes2: np.array(bboxes2_vals),
        })
        bbox_overlap_iou
        print(overlap)

        overlap = sess.run(overlap_op1, feed_dict={
            bboxes1: np.array(bboxes1_vals),
            bboxes2: np.array(bboxes2_vals),
        })
        print(overlap)



#And here is a implementation for the YOLO style format

'''def bbox_iou_center_xy(bboxes1, bboxes2):
    """ same as `bbox_iou_corner_xy', except that we have
        center_x, center_y, w, h instead of x1, y1, x2, y2 """

    x11, y11, w11, h11 = tf.split(bboxes1, 4, axis=1)
    x21, y21, w21, h21 = tf.split(bboxes2, 4, axis=1)

    xi1 = tf.maximum(x11, tf.transpose(x21))
    xi2 = tf.minimum(x11, tf.transpose(x21))

    yi1 = tf.maximum(y11, tf.transpose(y21))
    yi2 = tf.minimum(y11, tf.transpose(y21))

    wi = w11/2.0 + tf.transpose(w21/2.0)
    hi = h11/2.0 + tf.transpose(h21/2.0)

    inter_area = tf.maximum(wi - (xi1 - xi2 + 1), 0) \
                  * tf.maximum(hi - (yi1 - yi2 + 1), 0)

    bboxes1_area = w11 * h11
    bboxes2_area = w21 * h21

    union = (bboxes1_area + tf.transpose(bboxes2_area)) - inter_area

    return inter_area / (union+0.0001)





def find_iou(data_ground, data_predicted):
    print("e")
    xminofmax = tf.maximum(data_ground[0], data_predicted[0])
    yminofmax = tf.maximum(data_ground[1], data_predicted[1])
    xmaxofmin = tf.minimum(data_ground[2], data_predicted[2])
    ymaxofmin = tf.minimum(data_ground[3], data_predicted[3])

    interction = ((xmaxofmin - xminofmax + 1) * (ymaxofmin - yminofmax + 1))
#   i have added 1 to all the equation save the equation from giving 0 iou value
#    AOG: area of ground box
    AOG = (tf.abs(data_ground[0] - data_ground[2]) + 1) * (tf.abs(data_ground[1] - data_ground[3]) + 1)
    #AOP:area of predicted box
    AOP = (tf.abs(data_predicted[0] - data_predicted[2]) + 1) * (tf.abs(data_predicted[1] - data_predicted[3]) + 1)
    union= (AOG + AOP) - interction
    iou = (interction /union)
    mean_iou = tf.reduce_mean(iou)
    return (mean_iou)




if __name__ == '__main__':
    groundbb = tf.placeholder(shape=(None,4),dtype=tf.float32)
    predicted_bb = tf.placeholder(shape=(None,4),dtype=tf.float32)
    #overlap_op=groundbb+predicted_bb

    overlap_op = find_iou(groundbb, predicted_bb)
    #overlap_op=groundbb+predicted_bb


    sess=tf.Session()

    # data1=np.random.randint(low=10,high=50,size=(4,3))
    data_ground=np.array([[13,15,16],[12,14,15],[17,19,20],[20,30,35]])
    data_predicted=np.array([[15,19,20],[14,16,14],[19,22,25],[21,34,32]])
    #data_ground=np.array([13.0,15.0,16.0,22.0])
    #data_predicted=np.array([15.0,19.0,17.0,25.0])
    data_ground1=np.resize(data_ground,(data_ground.shape[1],4))
    data_predicted1=np.resize(data_predicted,(data_predicted.shape[1],4))
    overlap = sess.run(overlap_op,feed_dict={groundbb: data_ground1, predicted_bb: data_predicted1})
    print(overlap)


# data_ground=np.transpose(np.array([[13,15,16],[12,14,15],[17,19,20],[20,30,35]]))
# data_predicted=np.transpose(np.array([[15,19,20],[14,16,14],[19,22,25],[21,34,32]]))
#
# print(iou_metric(data_ground,data_predicted))

# '_______________________________________________---' \
    
import tensorflow as tf
import numpy as np





def find_iou(data_ground,data_predicted):
    xminofmax = tf.maximum((data_ground[:,0]),(data_predicted[:,0]))
    yminofmax = tf.maximum(data_ground[:,1], data_predicted[:,1])
    xmaxofmin = tf.minimum(data_ground[:,2], data_predicted[:,2])
    ymaxofmin = tf.minimum(data_ground[:,3], data_predicted[:,3])
    
    #Sub=(xmaxofmin - xminofmax + 1)
    sub1=tf.add(tf.subtract(xmaxofmin,xminofmax),1)
    sub2=tf.add(tf.subtract(ymaxofmin,yminofmax),1)
    intercetion=tf.multiply(sub1,sub2)
    
    
    aog1=tf.add(tf.abs(tf.subtract(data_ground[:,0],data_ground[:,2])),1)
    
    aog2=tf.add(tf.abs(tf.subtract(data_ground[:,1],data_ground[:,3])),1)
    
    AOG=tf.multiply(aog1,aog2)
    
    aop1=tf.add(tf.abs(tf.subtract(data_predicted[:,0],data_predicted[:,2])),1)
    
    aop2=tf.add(tf.abs(tf.subtract(data_predicted[:,1],data_predicted[:,3])),1)
    
    AOP=tf.multiply(aog1,aog2)
    
    Union=tf.subtract(tf.add(AOG,AOP),intercetion)
    iou=tf.divide(intercetion,Union)
    mean_iou=tf.reduce_mean(iou)
#AOG = (tf.abs(data_ground[0] - data_ground[2]) + 1) * (tf.abs(data_ground[1] - data_ground[3]) 


#interction = ((xmaxofmin - xminofmax + 1) * (ymaxofmin - yminofmax + 1))
# i have added 1 to all the equation save the equation from giving 0 iou value
#    AOG: area of ground box
AOG = (tf.abs(data_ground[0] - data_ground[2]) + 1) * (tf.abs(data_ground[1] - data_ground[3]) + 1)
#AOP:area of predicted box
AOP = (tf.abs(data_predicted[0] - data_predicted[2]) + 1) * (tf.abs(data_predicted[1] - data_predicted[3]) + 1)
union= (AOG + AOP) - interction
iou = (interction /union)
mean_iou = tf.reduce_mean(iou)



data_g=np.array([[64,57,190,119],
                [77,23,160,79],
                [77,26,146,118],
                [75,12,148,108]])
    

data_ground = tf.placeholder(shape=(None,4),dtype=tf.float32)
data_predicted = tf.placeholder(shape=(None,4),dtype=tf.float32)#overlap_op=groundbb+predicted_bb
overlap=find_iou(data_ground,data_predicted)

# data1=np.random.randint(low=10,high=50,size=(4,3))
#data_g=np.array([[13,15,16],[12,14,15],[17,19,20],[20,30,35]])
#data_p=np.array([[15,19,20],[14,16,14],[19,22,25],[21,34,32]])
# data_g=np.array([13.0,15.0,16.0,22.0])
# data_p=np.array([15.0,19.0,17.0,25.0])
#
# #
# data_g=np.transpose(data_g)
# data_p=np.transpose(data_p)

# data_ground=[1,2,3,4]
# data_predicted=[2,5,3,6]

# data_ground1=np.resize(data_ground,(data_ground.shape[1],4))
# data_predicted1=np.resize(data_predicted,(data_predicted.shape[1],4))

data_g = np.array([[64, 57, 190, 119],
                   [77, 23, 160, 79],
                   [77, 26, 146, 118]])

# data_g=data_g.transpose()
data_p = data_g + 50
feed_dict1=({data_ground: data_g, data_predicted: data_p})
sess = tf.Session()
#sess = tf_debug.TensorBoardDebugWrapperSession(sess, "localhost:7000")
overlap= sess.run([overlap],feed_dict=feed_dict1)
print(overlap)

oz,mo,intc,uni = sess.run([xminofmax,yminofmax,interction,union],feed_dict=feed_dict1)
#print(overlap)
print(mo,'\n',oz,'\n',intc,'\n',uni)
# '__________________________________
oz,mo,AOG1 = sess.run([mean_iou,AOP,AOG],feed_dict=feed_dict1)
print(oz,'\n',mo,'\n',AOG1)'''
