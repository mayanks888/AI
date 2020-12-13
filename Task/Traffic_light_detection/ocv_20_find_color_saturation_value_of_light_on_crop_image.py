import os
import time

import cv2
import matplotlib.pyplot as plt
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
# input_folder=root="/home/mayank_sati/Desktop/crop_image"
# input_folder=root="/home/mayank_sati/Desktop/black"
# input_folder=root="/home/mayank_sati/Desktop/black and blue"
# input_folder=root="/home/mayank_sati/Desktop/yellow"
# input_folder=root="/home/mayank_sati/Desktop/yellow_black"
# input_folder = root = "//home/mayank_sati/Desktop/red_black"
input_folder = root = "/home/mayank_sati/Desktop/one_Shot_learning/col"
for root, _, filenames in os.walk(input_folder):
    if (len(filenames) == 0):
        print("Input folder is empty")
    #     return 1
    # time_start = time.time()
    for filename in filenames:

        counter += 1
        if (counter > 1135):
            break
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

        # tot=np.count_nonzero(red_range)
        # if loop=="gwm_20.jpg" or loop=="gwm_150.jpg" or loop=="gwm_650.jpg" or loop=="gwm_823.jpg" or loop=="gwm_912.jpg"or loop=="gwm_968.jpg" or loop=="gwm_1055.jpg" or loop=="gwm_1027.jpg":
        if val % 1 == 0:
            ##################################333
            cv2.imshow('streched_image', image_scale)
            # output_folder_path="/home/mayank-s/PycharmProjects/Datasets/aptive/object_detect/output/output_merccedes.png"
            # filepath=output_folder_path+my_image+".png"
            # cv2.imwrite(filepath,my_image)
            cv2.waitKey(2000)
            cv2.destroyAllWindows()
            ##########################################333
            fig = plt.figure()
            # plt.plot([1, 2, 3])
            plt.hist(h, bins=40)
            plt.ylabel('No of times')
            plt.ylabel('hue')
            plt.title(filename)
            plt.xlim(0, 180)
            # plt.axis([0, 180,0,200])
            plt.grid(True)
            # ax1 = fig.add_subplot(221)
            # plt.subplot(ax1)
            plt.show()
            #####################################3333
            plt.hist(s, bins=40)
            plt.ylabel('No of times')
            plt.ylabel('saturation')
            # plt.title(filename)
            plt.xlim(0, 180)
            # plt.axis([0, 180,0,200])
            plt.grid(True)
            # ax2 = fig.add_subplot(222)
            # plt.subplot(ax2)
            plt.show()
            ###############################33
            plt.hist(v, bins=40)
            plt.ylabel('No of times')
            plt.ylabel('value')
            # plt.title(filename)
            plt.xlim(0, 180)
            # plt.axis([0, 180,0,200])
            plt.grid(True)
            # ax3 = fig.add_subplot(221)
            # ax3 = fig.add_subplot(223)
            # # ax3 = plt.subplot(222, frameon=False)
            # plt.subplot(ax3)
            plt.show()
            ###########################
            ###############################33
            # plt.hist(v, bins=10)
            # plt.ylabel('No of times')
            # plt.ylabel('value')
            # # plt.title(filename)
            # plt.xlim(0, 180)
            # # plt.axis([0, 180,0,200])
            # plt.grid(True)
            # # ax3 = fig.add_subplot(221)
            # ax4 = fig.add_subplot(224)
            # # ax3 = plt.subplot(222, frameon=False)
            # plt.subplot(ax4)
            ###########################
            # plt.show()

        red_range = cv2.inRange(h, 0, 21)
        # counter=np.unique(red_range,return_counts=True)
        total_red = np.count_nonzero(red_range)

        orange_range = cv2.inRange(h, 22, 44)
        # counter = np.unique(orange_range, return_counts=True)
        total_orange = np.count_nonzero(orange_range)

        yellow_range = cv2.inRange(h, 45, 70)
        # counter = np.unique(yellow_range, return_counts=True)
        total_yello = np.count_nonzero(yellow_range)

        total_red = total_orange + total_red + total_orange

        green_range = cv2.inRange(h, 70, 155)
        # counter = np.unique(green_range, return_counts=True)
        total_green = np.count_nonzero(green_range)

        cyan_range = cv2.inRange(h, 155, 186)
        # counter = np.unique(cyan_range, return_counts=True)
        total_cyan = np.count_nonzero(cyan_range)

        blue_range = cv2.inRange(h, 186, 278)
        # counter = np.unique(blue_range, return_counts=True)
        total_blue = np.count_nonzero(blue_range)

        total_green = total_green + total_cyan + total_blue
        purple_range = cv2.inRange(h, 278, 330)
        # counter = np.unique(purple_range, return_counts=True)
        total_purple = np.count_nonzero(purple_range)

        last = cv2.inRange(h, 330, 360)
        # counter = np.unique(last, return_counts=True)
        other = np.count_nonzero(last)

        # data_label = [image_path, total_red, total_orange, total_yello, total_green, total_cyan, total_blue, total_purple,other]
        data_label = [image_path, total_red, total_orange, total_yello, total_green,
                      round((total_red / total_yello), 3), round((total_orange / total_yello), 3),
                      round((total_green / total_yello), 3), total_cyan, total_blue, total_purple, other]

        # if not ((xmin == xmax) and (ymin == ymax)):
        bblabel.append(data_label)
        # print()
        # if loop>=1018:
        #     break

        # columns = ['image_path', 'total_red', 'total_orange', 'total_yello', 'total_green', 'total_cyan', 'total_blue', 'total_purple','other']
        columns = ['image_path', 'total_red', 'total_orange', 'total_yello', 'total_green', '(total_red/total_yello)',
                   '(total_orange/total_yello)', '(total_green/total_yello', 'total_cyan', 'total_blue', 'total_purple',
                   'other']

        df = pd.DataFrame(bblabel, columns=columns)

print("into csv file")
df.to_csv('colormaps_n.csv', index=False)

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
