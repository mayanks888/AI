#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Copyright 2016 Massachusetts Institute of Technology

"""Publish a video as ROS messages.
"""

import cv2
import rospy
from cv_bridge import CvBridge
from sensor_msgs.msg import Image

bridge = CvBridge()


def main():
    """Publish a video as ROS messages.
    """
    # Set up node.
    rospy.init_node("img_publisher", anonymous=True)
    img_pub = rospy.Publisher("/camera/rgb/image", Image,
                              queue_size=10)

    # img_pub = rospy.Publisher("/camera/rgb/image", Image,
    #                          queue_size=10)

    # Loop through video frames.
    while not rospy.is_shutdown():
        resized_image = cv2.imread("./baid_0.jpg", 1)
        resized_image = cv2.resize(resized_image, (300, 300))
        rospy.Rate(10).sleep()
        print("publish img")
        # Publish image.
        img_msg = bridge.cv2_to_imgmsg(resized_image, "bgr8")
        img_msg.header.stamp = rospy.Time.now()
        img_pub.publish(img_msg)


if __name__ == '__main__':
    try:
        main()
    except rospy.ROSInterruptException:
        pass
