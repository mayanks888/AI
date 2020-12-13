from __future__ import print_function

import numpy as np

bag_file = '/home/mayanksati/Documents/point_clouds/kitt_rosbag/2011_09_30_drive_18.bag'

# info_dict = yaml.load(Bag(bag_file, 'r')._get_yaml_info())
#
# # Print the information contained in the info_dict
# info_dict.keys()
# for topic in info_dict["topics"]:
#     print("-"*50)
#     for k,v in topic.items():
#         print(k.ljust(20), v)

import rosbag
import mayavi.mlab
import sensor_msgs.point_cloud2 as pc2

bag = rosbag.Bag(bag_file)
for topic, msg, t in bag.read_messages(topics=['velodyne_points']):
    # print (msg)
    # msgString = str(msg)
    # msgList = string.split(msgString, '\n')
    lidar = np.fromstring(msg.data, dtype=np.float32)
    lidar = lidar.reshape(-1, 4)
    a = pc2.read_points(msg)

    lidar = pc2.read_points(msg)
    lidar = np.array(list(lidar))
    # import mayavi.mlab

    fig = mayavi.mlab.figure(bgcolor=(0, 0, 0), size=(800, 500))

    mayavi.mlab.points3d(lidar[:, 0], lidar[:, 1], lidar[:, 2],
                         lidar[:, 0] ** 2 + lidar[:, 1] ** 2,  # distance values for color
                         mode="point",
                         colormap='spectral',
                         figure=fig,
                         )
    mayavi.mlab.show()

    k = msg
bag.close()
#
