#!/usr/bin/env python
# ROS node libs
import time

import cv2
# General libs
import numpy as np
import rospy
import tensorflow as tf
from cv_bridge import CvBridge
# Detector libs
from object_detection.utils import label_map_util
from sensor_msgs.msg import Image
from std_msgs.msg import Float32MultiArray

# GPU settings: Select GPUs to use. Coment it to let the system decide
# os.environ["CUDA_VISIBLE_DEVICES"]="0"
count = 0


class ros_tensorflow_obj():
    def __init__(self):
        # ## Initial msg
        rospy.loginfo('  ## Starting ROS Tensorflow interface ##')
        # ## Load a (frozen) Tensorflow model into memory.
        detection_graph = tf.Graph()
        with detection_graph.as_default():
            od_graph_def = tf.GraphDef()
            with tf.gfile.GFile("./object_detection/saved_model_frcnn_4000_samples_00/frozen_inference_graph.pb",
                                'rb') as fid:
                serialized_graph = fid.read()
                od_graph_def.ParseFromString(serialized_graph)
                tf.import_graph_def(od_graph_def, name='')

        # ## Loading label map
        # Label maps map indices to category names, so that when our convolution network predicts `5`, we know that this corresponds to `airplane`.  Here we use internal utility functions, but anything 			that returns a dictionary mapping integers to appropriate string labels would be fine
        self.category_index = label_map_util.create_category_index_from_labelmap(
            "./object_detection/Baidu_models/Baidu_ssd_model_0/haval_label_map.pbtxt", use_display_name=True)
        # ## Get Tensors to run from model
        self.image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
        # Each box represents a part of the image where a particular object was detected.
        self.boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
        # Each score represent how level of confidence for each of the objects.
        # Score is shown on the result image, together with the class label.
        self.scores = detection_graph.get_tensor_by_name('detection_scores:0')
        self.classes = detection_graph.get_tensor_by_name('detection_classes:0')
        self.num_detections = detection_graph.get_tensor_by_name('num_detections:0')

        # # Tensorflow Session opening: Creates a session with log_device_placement set to True.
        # ## Session configuration
        config = tf.ConfigProto(log_device_placement=True)
        config.log_device_placement = True
        config.gpu_options.allow_growth = True

        # ## Session openning
        try:
            with detection_graph.as_default():
                self.sess = tf.Session(graph=detection_graph, config=config)
                rospy.loginfo('  ## Tensorflow session open: Starting inference... ##')
        except ValueError:
            rospy.logerr('   ## Error when openning session. Please restart the node ##')
            rospy.logerr(ValueError)

        image_np = cv2.imread("baid_0.jpg", 1)
        image_np_expanded = np.expand_dims(image_np, axis=0)
        (boxes_out, scores_out, classes_out, num_detections_out) = self.sess.run(
            [self.boxes, self.scores, self.classes, self.num_detections],
            feed_dict={self.image_tensor: image_np_expanded})

        print("ready to process----------------------------------------------------------")
        # # ROS environment setup
        # ##  Define subscribers
        self.subscribers_def()
        # ## Define publishers
        self.publishers_def()
        # ## Get cv_bridge: CvBridge is an object that converts between OpenCV Images and ROS Image messages
        self._cv_bridge = CvBridge()
        self.now = rospy.Time.now()

    # Define subscribers
    def subscribers_def(self):
        subs_topic = 'crop_tl_image'
        self._sub = rospy.Subscriber(subs_topic, Image, self.img_callback, queue_size=1, buff_size=2 ** 24)

    # Define publishers
    def publishers_def(self):
        tl_bbox_topic = '/tl_bbox_topic_megs'
        self._pub = rospy.Publisher('tl_bbox_topic', Float32MultiArray, queue_size=1)

    # Camera image callback
    def img_callback(self, image_msg):
        global count
        image_msg.encoding = "bgr8"
        first = time.time()
        image_np = self._cv_bridge.imgmsg_to_cv2(image_msg, "bgr8")
        cv2.imshow("im_show", image_np)
        cv2.waitKey(30)
        print("Received image from apollo")
        save_path = "/notebooks/catkin_ws/src/augmented_traffic_light/save_cropped_images/crop_img_" + str(
            count) + ".jpg"
        cv2.imwrite(save_path, image_np)
        rows, cols, channels = image_np.shape
        image_np_expanded = np.expand_dims(image_np, axis=0)
        second = time.time()
        time_cost = round((second - first) * 1000)
        third = time.time()
        (boxes_out, scores_out, classes_out, num_detections_out) = self.sess.run(
            [self.boxes, self.scores, self.classes, self.num_detections],
            feed_dict={self.image_tensor: image_np_expanded})
        numboxes = np.squeeze(num_detections_out)
        # numboxes = 0
        tl_bbox = Float32MultiArray()
        if int(numboxes) >= 1:
            tmp = -np.ones(5 * int(numboxes) + 1)

            for i in range(0, int(numboxes)):
                score = float(np.squeeze(scores_out)[i])
                bbox = [float(v) for v in np.squeeze(boxes_out)[i]]
                tmp[0] = numboxes
                if score > 0.3:
                    x_top_left = bbox[1] * cols
                    y_top_left = bbox[0] * rows
                    x_bottom_right = bbox[3] * cols
                    y_bottom_right = bbox[2] * rows
                    width = x_bottom_right - x_top_left
                    height = y_bottom_right - y_top_left
                    tmp[5 * i + 1] = x_top_left
                    tmp[5 * i + 2] = y_top_left
                    tmp[5 * i + 3] = width
                    tmp[5 * i + 4] = height
                    tmp[5 * i + 5] = score
        else:
            tmp = [-1.0, 10.0, 10.0, 10.0, 10.0]
        tl_bbox.data = tmp
        fourth = time.time()
        time_cost = round((fourth - third) * 1000)
        if time_cost > 100:
            print("image_size", rows, " ", cols)
            print("post processing Time", time_cost)
            print("Inference Time", time_cost)
            print("bbox ", tl_bbox.data)
        self._pub.publish(tl_bbox)
        count = count + 1


# Spin once
def spin(self):
    rospy.spin()


def main():
    rospy.init_node('tf_object_detector', anonymous=True)
    tf_ob = ros_tensorflow_obj()
    try:
        rospy.spin()
    except KeyboardInterrupt:
        print("Shutting down")


if __name__ == '__main__':
    main()
