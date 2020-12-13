#!/usr/bin/env python
import rospy
# from modules.localization.proto.localization_pb2 import LocalizationEstimate
from sensor_msgs.msg import Image

# from bazel-apollo/py_proto/modules/localization/proto/localization_pb2.py
# from modules.localization.proto import localization_pb2
class Msg_reader(object):
    """Calculate mileage."""

    def __init__(self):
        """Init."""
        self.auto_mileage = 0.0
        self.manual_mileage = 0.0
        self.disengagements = 0


    def callback(data):
        # print("Hello")
        # rospy.loginfo(rospy.get_caller_id() + "I heard %s", data.data)
        # print(data.pose.position)
        time = data.header.stamp
        # print(data)
        print(time)


    def listener(self):
        # print("Lets go")
        rospy.init_node('listener', anonymous=True)
        # rospy.Subscriber("/apollo/localization/pose", LocalizationEstimate, callback)
        rospy.Subscriber("/apollo/sensor/camera/traffic/image_short", Image, callback)
        rospy.spin()


if __name__ == '__main__':
    mr=Msg_reader()
    mr.listener()
