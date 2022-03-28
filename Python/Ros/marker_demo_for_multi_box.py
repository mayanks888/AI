#!/usr/bin/env python

import time

import geometry_msgs.msg as geom_msg
import rospy
from geometry_msgs.msg import Quaternion, Pose, Point, Vector3
from std_msgs.msg import Header, ColorRGBA
from visualization_msgs.msg import Marker, MarkerArray


def wait_for_time():
    """Wait for simulated time to begin.
    """
    while rospy.Time().now().to_sec() == 0:
        pass


def main():
    rospy.init_node('my_node')

    def show_text_in_rviz_mullti_line_array(marker_pub_array, text):
        markers = MarkerArray()
        # bbox_data=[[[1,2,3],[3,5,6]]
        bbox_data = [[1.56140001, 4.0275182, 0.4781957],
                     [6.08904442, 4.04472732, 0.4781957],
                     [6.08904442, 4.04472732, 1.45005068],
                     [1.56140001, 4.0275182, 1.45005068],
                     [1.56778923, 2.3465438, 0.4781957],
                     [6.09543363, 2.36375292, 0.4781957],
                     [6.09543363, 2.36375292, 1.45005068],
                     [1.56778923, 2.3465438, 1.45005068]]
        # bbox_data=bbox_data.T

        for i in range(len(bbox_data)):
            # marker = Marker(type=Marker.LINE_LIST, ns='velodyne', action=Marker.ADD)
            marker = Marker(type=Marker.LINE_STRIP, ns='velodyne', action=Marker.ADD)
            marker.header.frame_id = "velodyne" + str(i)
            marker.header.stamp = marker.header.stamp
            marker.action = Marker.ADD
            marker.id = 42 + i
            # if bbox_data[i][0][0] == frame:

            for n in range(2):
                # point = geom_msg.Point(bbox_data[i][n + 1][0], bbox_data[i][n + 1][1], bbox_data[i][n + 1][1])
                point = geom_msg.Point(bbox_data[i][n], bbox_data[i][n], bbox_data[i][n])
                marker.points.append(point)

            marker.scale.x = 2
            marker.lifetime = rospy.Duration(2)
            marker.color.a = 1.0
            marker.color.r = 0
            marker.color.g = 1
            marker.color.b = 0
            # marker.ns = "est_pose_" + str(i)

            markers.markers.append(marker)
            marker_pub_array.publish(marker)

        # self.bbox.publish(markers)
        # marker_pub_array.publish(markers)

    def show_text_in_rviz_mullti_line(marker_pub, text):

        line_color = ColorRGBA()  # a nice color for my line (royalblue)
        line_color.r = 0.254902
        line_color.g = 0.411765
        line_color.b = 0.882353
        line_color.a = 1.0
        start_point = Point()  # start point
        start_point.x = 0.2
        start_point.y = 0.0
        start_point.z = 0.2
        end_point = Point()  # end point
        end_point.x = 20
        end_point.y = 20
        end_point.z = 20

        marker3 = Marker()
        marker3.id = 3
        marker3.header.frame_id = 'base_link'
        marker3.type = Marker.LINE_STRIP
        marker3.ns = 'Testline'
        marker3.action = Marker.ADD
        marker3.scale.x = 0.8
        marker3.points.append(start_point)
        marker3.points.append(end_point)
        marker3.colors.append(line_color)
        marker3.colors.append(line_color)
        marker_pub.publish(marker3)

    def show_text_in_rviz_mullti_cube(marker_publisher, text):
        markers_my = MarkerArray()
        markers_my.markers = []
        for i in range(5):
            print(time.time())
            ###################################################333333
            marker = Marker(
                type=Marker.CUBE,
                lifetime=rospy.Duration(5),
                pose=Pose(Point(0.5 + (i * 20), 0.5 + (i * 20), 1.45), Quaternion(i, i, i, 1)),
                scale=Vector3(0.6, 0.6, 0.6),
                header=Header(frame_id='base_link'),
                color=ColorRGBA(0.0, 1, 0.0, .2))
            marker.action = Marker.ADD
            marker.ns = "est_pose_" + str(i)
            marker.id = 42 + i
            marker.header.stamp = marker.header.stamp
            marker.pose.orientation.w = 1.0
            marker.pose.position.x = (0.5 + (i * 2))
            marker.pose.position.y = (0.5 + (i * 2))
            marker.pose.position.z = 1.45

            print(i)
            markers_my.markers.append(marker)
            # rospy.sleep(1)
        marker_publisher.publish(markers_my)

    def show_text_in_rviz_mullti(marker_publisher, text):
        markers = MarkerArray()
        for i in range(3):
            marker = Marker(type=Marker.LINE_LIST, ns='velodyne', action=Marker.ADD)
            marker.header.frame_id = "velodyne"
            marker.header.stamp = rospy.Time.now()
            # if self.bbox_data[i][0][0] == frame:

            for n in range(8):
                point = geom_msg.Point(self.bbox_data[i][n + 1][0], self.bbox_data[i][n + 1][1],
                                       self.bbox_data[i][n + 1][1])
                marker.points.append(point)

            marker.scale.x = 0.02
            marker.lifetime = rospy.Duration.from_sec(0.1)
            marker.color.a = 1.0
            marker.color.r = 0.5
            marker.color.g = 0.5
            marker.color.b = 0.5
            rospy.sleep(1)
            markers.markers.append(marker)

        marker_publisher.publish(markers)

    def show_text_in_rviz_mayank(marker_publisher, text):
        marker = Marker(
            type=Marker.CUBE,
            id=0,
            lifetime=rospy.Duration(5),
            pose=Pose(Point(1, 6, 6), Quaternion(0, 0, 0, 1)),
            scale=Vector3(.6, .6, .6),
            header=Header(frame_id='base_link'),
            color=ColorRGBA(0.0, 1.0, 0.0, 0.8),
            text=text)
        marker_publisher.publish(marker)
        print("working")

    # wait_for_time()

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
        # show_text_in_rviz_mullti_cube(marker_publisher, 'Hello world!')
        show_text_in_rviz_mullti_line(marker_pub, 'Hello world!')
        # show_text_in_rviz_mullti_line_array(marker_pub_array, 'Hello world!')
        print(val)


if __name__ == '__main__':
    main()
