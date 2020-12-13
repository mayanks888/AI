# import pyrosbag as prb
import rosbag
import yaml
from rosbag.bag import Bag

# filename = "/home/mayanksati/PycharmProjects/models/AI-Hamid/bags_tracker/2018-11-24-09-25-36/2018-11-24-09-25-38_0.bag"
# filename = "/home/mayank_sati/Documents/ob_apollo_build/mayank_apollo/2019-01-15-13-41-55_0.bag"
# filename = "/home/mayank_sati/Documents/datsets/1017_PCD_COLLECTION/truck1/2018-10-17-16-48-57_0.bag"
filename = "/home/mayank_sati/Documents/datsets/1017_PCD_COLLECTION/ego_station_veh/2018-10-17-14-53-42_0.bag"

bag = rosbag.Bag(filename)
rosbag_info = rosbag.Bag(filename).read_messages()
# for topic, msg, t in rosbag.Bag(filename).read_messages():
# # for topic, msg, t in bag.read_messages(topics=['chatter', 'numbers']):
#     print msg
# bag.close()       # kotaro.send(inputs)

# ________________________________________________________________
# getting list of all the topic from rosbag
info_dict = yaml.load(Bag(filename, 'r')._get_yaml_info())
print(1)

# '/apollo/sensor/velodyne64/PointCloud2'
#
# bag = rosbag_info
#
# topics = bag.get_type_and_topic_info()[1].keys()
#
# types = []
#
# for i in range(0, len(bag.get_type_and_topic_info()[1].values())):
#
#     types.append(bag.get_type_and_topic_info()[1].values()[i][0])
# # ________________________________________________________________


num_msgs = 100

with rosbag.Bag(filename, 'w') as outbag:
    for topic, msg, t in rosbag.Bag(filename).read_messages():
        while num_msgs:
            outbag.write(topic, msg, t)
            num_msgs -= 1
