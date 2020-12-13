#!/usr/bin/env python
import rospy
from modules.localization.proto.localization_pb2 import LocalizationEstimate
from sensor_msgs.msg import Image
# from bazel-apollo/py_proto/modules/localization/proto/localization_pb2.py
# from modules.localization.proto import localization_pb2
import getopt

class localisation_converter:

    def __init__(self):
        # # self.image_pub = rospy.Publisher("image_topic_2",Image)
        #
        # self.bridge = CvBridge()
        # self.image_sub = rospy.Subscriber("/camera/rgb/image", Image, self.callback)
        self.pose_info = rospy.Subscriber("/apollo/localization/pose", LocalizationEstimate, self.callback)

    def callback(self, data):
        print("Receive message")
        try:
            # print("Hello")
            # rospy.loginfo(rospy.get_caller_id() + "I heard %s", data.data)
            time = data.header.stamp
            # print(data.header.timestamp_sec)
            print(data.pose.position.x)
            return (data.pose.position.x)
            self.position = str(time + "_" + data.pose.position.x) + '_' + str(data.pose.position.y)
            return self.position
        except CvBridgeError as e:
            print(e)

        # (rows,cols,channels) = cv_image.shape
        # if cols > 60 and rows > 60 :
        #  cv2.circle(cv_image, (50,50), 10, 255)

        # cv2.imshow("subscribe image", cv_image)
        # cv2.waitKey(30)

    # try:
    #   self.image_pub.publish(self.bridge.cv2_to_imgmsg(cv_image, "bgr8"))
    # except CvBridgeError as e:
    #   print(e)


def main():
    rospy.init_node('listener', anonymous=True)
    ic = localisation_converter()
    try:
        rospy.spin()
    except KeyboardInterrupt:
        print("Shutting down")
    # cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
