#!/usr/bin/env python

import time

import geometry_msgs.msg as geom_msg
import rospy
from geometry_msgs.msg import Quaternion, Pose, Point, Vector3
from std_msgs.msg import Header, ColorRGBA
from visualization_msgs.msg import Marker, MarkerArray
from visualization_msgs.msg import Marker


def wait_for_time():
    """Wait for simulated time to begin.
    """
    while rospy.Time().now().to_sec() == 0:
        pass


def main():
    rospy.init_node('my_node')


    def show_text_in_rviz(marker_publisher, text):
        marker = Marker(
            type=Marker.TEXT_VIEW_FACING,
            id=0,
            lifetime=rospy.Duration(5),
            pose=Pose(Point(0.5, 0.5, 1.45), Quaternion(0, 0, 0, 1)),
            scale=Vector3(0.06, 0.06, 0.06),
            header=Header(frame_id='base_link'),
            color=ColorRGBA(0.0, 1.0, 0.0, 0.8),
            text=text)
        marker_publisher.publish(marker)
        print("working")

    #
    # marker_publisher = rospy.Publisher('_mayank_visualization_marker', MarkerArray, queue_size=5)
    marker_pub = rospy.Publisher('line_visualization_marker', Marker, queue_size=5)
    # marker_pub_array = rospy.Publisher('line_visualization_marker_array', Marker, queue_size=5)

    for val in range(1000):
        rospy.sleep(1)
        show_text_in_rviz(marker_pub, 'Hello world!')
        # show_text_in_rviz_mullti_cube(marker_publisher, 'Hello world!')
        # show_text_in_rviz_mullti_line(marker_pub, 'Hello world!')
        # show_text_in_rviz_mullti_line_array(marker_pub_array, 'Hello world!')
        print(val)


if __name__ == '__main__':
    main()
