import rosbag
import cv2
import cv_bridge
import yaml
flag = True

# mytopic = "/apollo/localization/pose"
mytopic = "/apollo/sensor/camera/traffic/image_short"


# bag_name_in='/media/mayank_sati/DATA/datasets/one_shot_datasets/Farmington/bags/2019-09-27-14-36-06/2019-09-27-14-36-08.bag'
bag_name_in='/home/mayank_s/codebase/bagfiles/2_1.bag'
base_path = '/home/mayank_sati/Desktop/one_Shot_learning/farmington/traffic_shot_new'

yaml_list = yaml.load(rosbag.Bag(bag_name_in, 'r')._get_yaml_info())
# with open('rosbag_annotation.yaml', 'a') as outfile:
#     yaml.dump(yaml_list, outfile)

counter = 1
for index, (topic, msg, t) in enumerate(rosbag.Bag(bag_name_in).read_messages()):
    if topic == mytopic:
        # +++++++++++++++++++++++++++++++++++++++++++++++++++
        # image_path = image_file_path + str(counter) + '.jpg'
        if flag == True:
            print(t)
            msg.encoding = 'yuv422'
            bridge = cv_bridge.CvBridge()
            cv_img = bridge.imgmsg_to_cv2(msg, 'yuv422')
            # cv_img = bridge.imgmsg_to_cv2(msg, "bgr8")
            # cv_img = bridge.imgmsg_to_cv2(msg, 'passthrough')

            # cv_image = bridge.imgmsg_to_cv2(msg, desired_encoding="passthrough")
            cv_img = cv2.cvtColor(cv_img, cv2.COLOR_YUV2BGR_YUY2)
            path = base_path + '/' + str(counter) + '.jpg'
            print(path)
            cv2.imwrite(path, cv_img)
            counter += 1
