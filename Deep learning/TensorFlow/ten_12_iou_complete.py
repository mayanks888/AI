import tensorflow as tf
import numpy as np

def find_my_iou(data_ground, data_predicted):
    xminofmax = tf.maximum((data_ground[:, 0]), (data_predicted[:, 0]))
    yminofmax = tf.maximum(data_ground[:, 1], data_predicted[:, 1])
    xmaxofmin = tf.minimum(data_ground[:, 2], data_predicted[:, 2])
    ymaxofmin = tf.minimum(data_ground[:, 3], data_predicted[:, 3])

    # Sub=(xmaxofmin - xminofmax + 1)
    sub1 = tf.add(tf.subtract(xmaxofmin, xminofmax), 1)
    sub2 = tf.add(tf.subtract(ymaxofmin, yminofmax), 1)
    intercetion = tf.multiply(sub1, sub2)

    aog1 = tf.add(tf.abs(tf.subtract(data_ground[:, 0], data_ground[:, 2])), 1)

    aog2 = tf.add(tf.abs(tf.subtract(data_ground[:, 1], data_ground[:, 3])), 1)

    AOG = tf.multiply(aog1, aog2)

    aop1 = tf.add(tf.abs(tf.subtract(data_predicted[:, 0], data_predicted[:, 2])), 1)

    aop2 = tf.add(tf.abs(tf.subtract(data_predicted[:, 1], data_predicted[:, 3])), 1)

    AOP = tf.multiply(aog1, aog2)

    Union = tf.subtract(tf.add(AOG, AOP), intercetion)
    iou = tf.divide(intercetion, Union)
    mean_iou = tf.reduce_mean(iou)
    return mean_iou

  

    #data_ground = tf.placeholder(shape=(None,4),dtype=tf.float32)
    #data_predicted = tf.placeholder(shape=(None,4),dtype=tf.float32)
if __name__ == '__main__':
    data_ground = tf.placeholder(shape=None,dtype=tf.float32)
    data_predicted = tf.placeholder(shape=None,dtype=tf.float32)
    meanIOU=find_my_iou(data_ground,data_predicted)

    data_g = np.array([[64, 57, 190, 119],
                       [77, 23, 160, 79],
                       [77, 26, 146, 118]])

    # data_g=data_g.transpose()
    data_p = data_g + .1
    feed_dict1=({data_ground: data_g, data_predicted: data_p})
    sess = tf.Session()
   # sess.run(cool,feed_dict=feed_dict1)
    #sess = tf_debug.TensorBoardDebugWrapperSession(sess, "localhost:7000")
    overlap= sess.run(meanIOU,feed_dict=feed_dict1)
    print(overlap)
