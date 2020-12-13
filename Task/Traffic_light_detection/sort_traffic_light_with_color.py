import os
import time

import cv2
import numpy as np
import pandas as pd

# from PIL import Image
ts = time.time()
# root="/home/mayank_sati/Documents/datsets/Rosbag_files/short_range_images/2018-11-24-09-44-06/Baidu_TL_dataset1"

# csv_path="/home/mayanksati/Documents/Rosbag_files/short_range_images/gwm_data_train.csv"
# csv_path="/home/mayank_sati/pycharm_projects/Data_Science_new/Deep learning/Load Datasets/new_traffic.csv"
# csv_path=('/home/mayanksati/Documents/csv/traffic_light_square.csv')
# csv_path=('/home/mayank_sati/Documents/datsets/Rosbag_files/short_range_images/2018-11-24-09-45-27/re3_5.csv')
# csv_path=('/home/mayank_sati/Documents/datsets/Rosbag_files/short_range_images/2018-11-24-09-31-33/re3_3.csv')
# csv_path=('/home/mayank_sati/Documents/datsets/Rosbag_files/short_range_images/2018-11-24-09-44-06/re3_4.csv')
# root='/home/mayanksati/PycharmProjects/models/tensorflow/re3-tensorflow-master/demo/GWM_dataset'

# dataset = pd.read_csv(csv_path)
# print(dataset.head())
# # dataset=dataset.iloc[np.random.permutation(len(dataset))]
# x = dataset.iloc[:, 0].values
# y = dataset.iloc[:, 4:8].values
val = 0
bblabel = []
counter = 0
# for loop,xml_aray in zip(x, y):
input_folder = root = "/home/mayank_sati/Desktop/one_Shot_learning/col"
# input_folder=root="/home/mayank_sati/Desktop/black"
# input_folder=root="/home/mayank_sati/Desktop/black and blue"
# input_folder=root="/home/mayank_sati/Desktop/yellow"
for root, _, filenames in os.walk(input_folder):
    if (len(filenames) == 0):
        print("Input folder is empty")
    #     return 1
    # time_start = time.time()
    for filename in filenames:

        counter += 1
        # if(counter>1135):
        #     break
        # print(loop)
        #####################

        image_path = os.path.join(root, filename)
        image_scale = cv2.imread(image_path, 1)
        ######################################################33333
        # breaking frames done
        # break
        hsv = cv2.cvtColor(image_scale, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv)

        # k=np.squeeze(h,axis=0)
        # k = h.view()
        h = h.flatten()
        s = s.flatten()
        v = v.flatten()
        # h=h*2
        red_range1 = cv2.inRange(h, 0, 15)
        # counter=np.unique(red_range,return_counts=True)
        total_red1 = np.count_nonzero(red_range1)

        red_range2 = cv2.inRange(h, 170, 180)
        # counter=np.unique(red_range,return_counts=True)
        total_red2 = np.count_nonzero(red_range2)

        total_red = total_red1 + total_red2

        # orange_range = cv2.inRange(h, 22, 44)
        # # counter = np.unique(orange_range, return_counts=True)
        # total_orange = np.count_nonzero(orange_range)

        yellow_range = cv2.inRange(h, 20, 40)
        # counter = np.unique(yellow_range, return_counts=True)
        total_yello = np.count_nonzero(yellow_range)

        # total_red=total_orange+total_red+total_orange

        green_range = cv2.inRange(h, 75, 100)
        # counter = np.unique(green_range, return_counts=True)
        total_green = np.count_nonzero(green_range)

        black_range = cv2.inRange(h, 50, 70)
        # counter = np.unique(cyan_range, return_counts=True)
        total_black = np.count_nonzero(black_range)

        black_sat = cv2.inRange(s, 0, 60)
        total_black_sat = np.count_nonzero(black_sat)

        ####################################################################33333
        colors = ['red_light', 'yellow_light', 'green_light', 'black_light']
        color_label = [total_red, total_yello, total_green, total_black]
        data = (max([(v, i) for i, v in enumerate(color_label)]))
        max_val = data[0]
        max_index = data[1]
        if max_index == 3:
            black_sat = s.shape[0]
            sat_ratio = total_black_sat / black_sat
            if sat_ratio > .4:
                print(sat_ratio)
                light_color = colors[3]
            else:
                del color_label[3]
                data = (max([(v, i) for i, v in enumerate(color_label)]))
                max_val = data[0]
                max_index = data[1]
                light_color = colors[max_index]
        else:
            light_color = colors[max_index]
        ######################################################################
        # saving light
        new_filePath = "/home/mayank_sati/Desktop/one_Shot_learning/col_2"
        new_filePath = new_filePath + str(light_color) + str(counter) + ".jpg"
        cv2.imwrite(new_filePath, image_scale)
        #########################################################################
        # data_label = [image_path, total_red, total_orange, total_yello, total_green, total_cyan, total_blue, total_purple,other]
        data_label = [filename, total_red, total_yello, total_green, total_black, total_black_sat]
        # data_label = [image_path, total_red,  total_yello,total_green, round((total_red/total_yello),3), round((total_orange/total_yello),3), round((total_green/total_yello),3), total_cyan, total_blue, total_purple,other]
        # data_label = [image_path, total_red,  total_yello,total_green, round((total_red/total_yello),3), round((total_orange/total_yello),3), round((total_green/total_yello),3), total_cyan, total_blue, total_purple,other]

        # if not ((xmin == xmax) and (ymin == ymax)):
        bblabel.append(data_label)
        # print()
        # if loop>=1018:
        #     break

        # columns = ['image_path', 'total_red', 'total_orange', 'total_yello', 'total_green', 'total_cyan', 'total_blue', 'total_purple','other']
        # columns = ['image_path', 'total_red', 'total_orange', 'total_yello', 'total_green', '(total_red/total_yello)', '(total_orange/total_yello)', '(total_green/total_yello','total_cyan', 'total_blue', 'total_purple','other']
        columns = ['filename', 'total_red', 'total_yello', 'total_green', 'total_black', 'total_black_sat']

        df = pd.DataFrame(bblabel, columns=columns)

print("into csv file")
df.to_csv('colormapsn.csv', index=False)

# lower_red = np.array([0, 120, 70])
# upper_red = np.array([10, 255, 255])
# mask1 = cv2.inRange(hsv, lower_red, upper_red)
# lower_red = np.array([170, 120, 70])
# upper_red = np.array([180, 255, 255])
# mask2 = cv2.inRange(hsv, lower_red, upper_red)
# mask1=mask1+mask2
# lower_red = np.array([110,50,50])
# upper_red = np.array([130,255,255])
# mask3 = cv2.inRange(hsv, lower_red, upper_red)
# print(1)
