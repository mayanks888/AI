#!/usr/bin/env python
# ROS node libs
import time
import numpy as np
import rospy
import torch
# from geometry_msgs.msg import Quaternion, Pose, Point, Vector3
from pyquaternion import Quaternion
from sensor_msgs.msg import PointCloud2
from std_msgs.msg import Header, ColorRGBA
from visualization_msgs.msg import Marker, MarkerArray
from std_msgs.msg import Int16, Float32MultiArray
from jsk_recognition_msgs.msg import BoundingBox, BoundingBoxArray
# import ros_numpy
#######################################################3
import rospy
from std_msgs.msg import Header
from sensor_msgs.msg import PointCloud2, PointField
import sensor_msgs.point_cloud2 as pc2
import pickle
#################################################################

# GPU settings: Select GPUs to use. Coment it to let the system decide
# os.environ["CUDA_VISIBLE_DEVICES"]="0"

class Ros_lidar_det():
    def __init__(self):
        # ## Initial msg
        rospy.loginfo('  ## Starting ROS  interface ##')
        print("ready to process----------------------------------------------------------")
        ########################################################################
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = device
        ######################################################################

        self.datapath_file = '/home/mayank_sati/Documents/datasets/other/output_MinkU.pkl'
        ######################################################################

        self.publishers_def()
        self.pcl_msg = PointCloud2()
        self.now = rospy.Time.now()

    # # Define subscribers
    # def subscribers_def(self):
    #     # subs_topic = '/livox/lidar'
    #     # subs_topic = '/apollo/sensor/velodyne32C/compensator/PointCloud2'
    #     # subs_topic = '/velodyne_cloud_registered'
    #     subs_topic = '/livox_pcl2'
    #     # subs_topic = '/lidar_top'
    #     self._sub = rospy.Subscriber(subs_topic, PointCloud2, self.lidar_callback, queue_size=10, buff_size=2 ** 24)
    #     # mydata = rospy.Subscriber( subs_topic , PointCloud2, self.lidar_callback, queue_size=1, buff_size=2**24)
    #     # print(mydata)
    #     # self._sub = rospy.Subscriber( subs_topic , Image, self.lidar_callback, queue_size=1, buff_size=100)

    # Define publishers
    def publishers_def(self):
        self._pub = rospy.Publisher('pc_bbox_topic', Float32MultiArray, queue_size=1)
        self.pub_arr_bbox = rospy.Publisher("Detections", BoundingBoxArray, queue_size=1)
        self.pcl_publisher = rospy.Publisher('point_clouds', PointCloud2, queue_size=1)

    def convertCloudToRos(self,open3d_cloud, frame_id="odom"):
        FIELDS_XYZ = [
            PointField(name='x', offset=0, datatype=PointField.FLOAT32, count=1),
            PointField(name='y', offset=4, datatype=PointField.FLOAT32, count=1),
            PointField(name='z', offset=8, datatype=PointField.FLOAT32, count=1),
        ]

        FIELDS_XYZRGB = FIELDS_XYZ + \
                [PointField(name='rgb', offset=12, datatype=PointField.UINT32, count=1)]
        # Set "header"
        header = Header()
        header.stamp = rospy.Time.now()
        header.frame_id = frame_id

        # Set "fields" and "cloud_data"
        points = np.asarray(open3d_cloud)
        if 1 : # XYZ only
            fields = FIELDS_XYZ
            cloud_data = points
        else:  # XYZ + RGB
            fields = FIELDS_XYZRGB
            # -- Change rgb color from "three float" to "one 24-byte int"
            # 0x00FFFFFF is white, 0x00000000 is black.
            colors = np.floor(np.asarray(open3d_cloud.colors) * 255)  # nx3 matrix
            colors = colors[:, 0] * BIT_MOVE_16 + colors[:, 1] * BIT_MOVE_8 + colors[:, 2]
            cloud_data = np.c_[points, colors]

        # create ros_cloud
        return pc2.create_cloud(header, fields, cloud_data)

    def infer_pcd(self):
        # arr_bbox = BoundingBoxArray()
        pcd_data = pickle.load(open(self.datapath_file, "rb"))
        print(1)
        # rate = rospy.Rate(100)
        for iIndex, pcl_data in enumerate(pcd_data, start=0):
                pcl_data=pcl_data.T
            ######################################33333
                if 1:  # Use the cloud from file
                    rospy.loginfo("Converting cloud from Open3d to ROS PointCloud2 ...")
                    ros_cloud = self.convertCloudToRos(pcl_data)

                else:  # Use the cloud with 3 points generated below
                    rospy.loginfo("Converting a 3-point cloud into ROS PointCloud2 ...")
                    TEST_CLOUD_POINTS = [
                        [1.0, 0.0, 0.0, 0xff0000],
                        [0.0, 1.0, 0.0, 0x00ff00],
                        [0.0, 0.0, 1.0, 0x0000ff],
                    ]
                    ros_cloud = pc2.create_cloud(
                        Header(frame_id="odom"), FIELDS_XYZ, TEST_CLOUD_POINTS)

                # publish cloud
                self.pcl_publisher.publish(ros_cloud)
                # rospy.loginfo("Conversion and publish success ...\n")
                print(1)
                # rospy.sleep(1)
        ############################################3333


def spin(self):
    rospy.spin()


def main():
    rospy.init_node('LIDAR_NODE', anonymous=True)
    try:
        tf_ob = Ros_lidar_det()
        for _ in range(30000):
            tf_ob.infer_pcd()
    except rospy.ROSInterruptException:
        pass

if __name__ == '__main__':
    main()
