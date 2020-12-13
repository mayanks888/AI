import rosbag
import cv2
import cv_bridge
import yaml
flag = True
# mytopic = "/apollo/sensor/camera/traffic/image_short"
# mytopic = "/apollo/sensor/camera/traffic/image_side"
# bag_name_in='/home/mayank_sati/Documents/datsets/Rosbag_files/78_gb_us_demo_route.bag'
# bag_name_in = '/home/mayank_sati/Documents/datsets/Rosbag_files/traffic_light12.bag'
# bag_name_in='/home/mayank_sati/Documents/datsets/Rosbag_files/short_range_images/2018-11-24-09-31-33/2018-11-24-09-31-35_0.bag'
# bag_name_in='/home/mayank_sati/Documents/ob_apollo_build/mayank_apollo/lei_bag/2019-09-07-14-57-43/2019-09-07-14-57-45.bag.rct.tmp'
bag_name_in='/home/mayank_sati/Documents/ob_apollo_build/mayank_apollo/lei_bag/lei_to_mayank/2019-09-07-14-57-43/2019-09-07-14-57-45.bag'
# base_path = '/home/mayank_sati/Desktop/one_Shot_learning/farmington/traffic_shot_new'
yaml_list = yaml.load(rosbag.Bag(bag_name_in, 'r')._get_yaml_info())
with open('rostopic_2.yaml', 'a') as outfile:
    yaml.dump(yaml_list, outfile)