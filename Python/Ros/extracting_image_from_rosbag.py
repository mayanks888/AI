#! /usr/bin/python
# Copyright (c) 2015, Rethink Robotics, Inc.

# Using this CvBridge Tutorial for converting
# ROS images to OpenCV2 images
# http://wiki.ros.org/cv_bridge/Tutorials/ConvertingBetweenROSImagesAndOpenCVImagesPython

# Using this OpenCV2 tutorial for saving Images:
# http://opencv-python-tutroals.readthedocs.org/en/latest/py_tutorials/py_gui/py_image_display/py_image_display.html

# For parsing command line
import getopt
import sys

# OpenCV2 for saving an image
import cv2
# rospy for the subscriber
import rospy
# ROS Image message -> OpenCV2 image converter
from cv_bridge import CvBridge, CvBridgeError
# ROS Image message
from sensor_msgs.msg import Image

# Instantiate CvBridgeoscore

bridge = CvBridge()


def image_callback(msg, prefix):
    file_prefix = ''
    if prefix == '':
        print("Received an image")
    else:
        print("Received an image for {}".format(prefix))
        file_prefix = prefix + '-'
    try:
        # Convert your ROS Image message to OpenCV2
        cv2_img = bridge.imgmsg_to_cv2(msg, "bgr8")
    except CvBridgeError, e:
        print(e)
    else:
        # Save your OpenCV2 image as a jpeg
        time = msg.header.stamp
        base_path = "/home/mayank_sati/Desktop/one_Shot_learning/farmington/traffic_shot"
        cv2.imwrite(base_path + '/' + file_prefix + str(time) + '.jpeg', cv2_img)
        rospy.sleep(1)


def usage():
    print
    'Usage: {} -t topic prefixes'.format(sys.argv[0])
    sys.exit(2)


def main():
    rospy.init_node('image_listener')

    try:
        opts, args = getopt.getopt(sys.argv[1:], 't:', ["topic="])
    except getopt.GetoptError:
        usage()

    # image_topic = "/camera/image_raw"
    image_topic = "/apollo/sensor/camera/traffic/image_short"
    for opt, arg in opts:
        print('Check option: {} {}'.format(opt, arg))
        if opt in ('-t', '--topic'):
            # Define your image topic
            image_topic = arg

    if len(args) == 0:
        # Set up your subscriber and define its callback
        print
        'Subscribing to: {}'.format(image_topic)
        rospy.Subscriber(image_topic, Image, image_callback, '')
    else:
        # Set up subscribers for each prefix
        for p in args:
            full_topic = '/' + p + image_topic
            print
            'Subscribing to: {}'.format(full_topic)
            rospy.Subscriber(full_topic, Image, image_callback, p)

    # Spin until ctrl + c
    rospy.spin()


if __name__ == '__main__':
    main()
