#!/usr/bin/env python
# cmd = 'mkdir -p ~/catkin_ws/src'
# os.system(cmd)
#
# os.system('catkin_make')
import rospy
from std_msgs.msg import String


# os.system('roscore')
# os.system('source /opt/ros/kinetic/setup.bash')


def talker():
    pub = rospy.Publisher('chatter', String, queue_size=10)

    rospy.init_node('talker', anonymous=True)
    rate = rospy.Rate(10)  # 10hz
    while not rospy.is_shutdown():
        hello_str = "hello world %s" % rospy.get_time()
        rospy.loginfo(hello_str)
        pub.publish(hello_str)
        rate.sleep()


if __name__ == '__main__':
    try:
        talker()
    except rospy.ROSInterruptException:
        pass
