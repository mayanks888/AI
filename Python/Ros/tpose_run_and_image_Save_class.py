#!/usr/bin/env python
import rospy
from modules.localization.proto.localization_pb2 import LocalizationEstimate
from sensor_msgs.msg import Image
# from bazel-apollo/py_proto/modules/localization/proto/localization_pb2.py
# from modules.localization.proto import localization_pb2
import getopt
import sys
import os
# OpenCV2 for saving an image
import cv2
# rospy for the subscriber
import rospy
# ROS Image message -> OpenCV2 image converter
from cv_bridge import CvBridge, CvBridgeError
# ROS Image message
from sensor_msgs.msg import Image


############################3333

class image_converter:

    def __init__(self):
        # self.image_pub = rospy.Publisher("image_topic_2",Image)

        self.bridge = CvBridge()
        # self.image_sub = rospy.Subscriber("/apollo/sensor/camera/traffic/image_short", Image, self.image_callback)
        self.image_sub = rospy.Subscriber("/apollo/sensor/camera/traffic/image_short", Image, self.image_callback,queue_size = 2, buff_size = 2 ** 15)
        # self.image_sub = rospy.Subscriber("/apollo/sensor/camera/traffic/image_left", Image, self.image_callback,queue_size = 10, buff_size = 2 ** 24)

        self.position=''

    def callback(self,data):
        # print("Hello")
        # rospy.loginfo(rospy.get_caller_id() + "I heard %s", data.data)
        # print(data.header.timestamp_sec)
        # print(data.pose.position)
        # return (data.pose.position.x)
        self.position=str(data.pose.position.x)+'_'+str(data.pose.position.y)

    def image_callback(self,msg):
        try:
            print ("Received an image")
            # Convert your ROS Image message to OpenCV2
            rospy.Subscriber("/apollo/localization/pose", LocalizationEstimate, self.callback)
            # print (pose_info)
            cv2_img = self.bridge.imgmsg_to_cv2(msg, "bgr8")
            # print(cv2_img)
            # msg.encoding = 'yuv422'
            # cv2_img = self.bridge.imgmsg_to_cv2(msg, "yuv422")
            #############################333
            # msg.encoding = 'yuv422'
            # # bridge = cv_bridge.CvBridge()
            # cv_img = self.bridge.imgmsg_to_cv2(msg, 'yuv422')
            ###############################
            # print(cv2_img)
        except CvBridgeError, e:
            print(e)
        else:
            # Save your OpenCV2 image as a jpeg
            time = msg.header.stamp
            ################33333
            # rospy.Subscriber("/apollo/localization/pose", LocalizationEstimate, callback)
            #######################
            # cv2_img=msg
            # base_path = "./xishui_1"
            base_path = "./new"
            # base_path = "./s-n-1"
            if not os.path.exists(base_path):
                print("base_path folder not present. Creating New folder...")
                os.makedirs(base_path)
                loop = 1
            # base_path = "./xishui_new"
            save_path=(base_path +"/" + str(time) +'_' +self.position+'.jpg')
            cv2.imwrite(save_path, cv2_img)
            # ros


    # def image_callback(self, data):
    #     print("Receive message")
    #     try:
    #         cv_image = self.bridge.imgmsg_to_cv2(data, "bgr8")
    #     except CvBridgeError as e:
    #         print(e)
    #
    #     # (rows,cols,channels) = cv_image.shape
    #     # if cols > 60 and rows > 60 :
    #     #  cv2.circle(cv_image, (50,50), 10, 255)
    #
    #     cv2.imshow("subscribe image", cv_image)
    #     cv2.waitKey(30)
    #
    # # try:
    # #   self.image_pub.publish(self.bridge.cv2_to_imgmsg(cv_image, "bgr8"))
    # # except CvBridgeError as e:
    # #   print(e)


def main(args):

    rospy.init_node('image_converter', anonymous=True)
    ic = image_converter()
    try:
        rospy.spin()
    except KeyboardInterrupt:
        print("Shutting down")
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main(sys.argv)

#################################
