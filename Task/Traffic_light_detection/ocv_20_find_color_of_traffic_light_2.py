import os
import shutil
import time

import cv2
import numpy as np
import pandas as pd

# from PIL import Image
ts = time.time()
# root="/home/mayank_sati/Documents/datsets/Rosbag_files/short_range_images/2018-11-24-09-47-09/Baidu_TL_dataset1"
# root="/home/mayank_sati/Documents/datsets/Rosbag_files/short_range_images/2018-11-24-09-45-27/Baidu_TL_dataset1"
# root="/home/mayank_sati/Documents/datsets/Rosbag_files/short_range_images/2018-11-24-09-44-06/Baidu_TL_dataset1"
# root="/home/mayank_sati/Documents/datsets/Rosbag_files/short_range_images/2018-11-24-09-31-33/Baidu_TL_dataset1"
# root="/home/mayank_sati/Documents/datsets/Rosbag_files/short_range_images/2018-11-24-09-25-36_copy/GWM_dataset"

# csv_path="/home/mayanksati/Documents/Rosbag_files/short_range_images/gwm_data_train.csv"
# csv_path="/home/mayank_sati/pycharm_projects/Data_Science_new/Deep learning/Load Datasets/new_traffic.csv"
# csv_path=('/home/mayanksati/Documents/csv/traffic_light_square.csv')
csv_path = ('/home/mayank_sati/Documents/datsets/Rosbag_files/short_range_images/square_light_combined.csv')
# csv_path=('/home/mayank_sati/Documents/datsets/Rosbag_files/short_range_images/2018-11-24-09-31-33/re3_3.csv')
# csv_path=('/home/mayank_sati/Documents/datsets/Rosbag_files/short_range_images/2018-11-24-09-44-06/re3_4.csv')
# csv_path=('/home/mayank_sati/Documents/datsets/Rosbag_files/short_range_images/2018-11-24-09-47-09/re3_6.csv')
# csv_path=('/home/mayank_sati/Documents/datsets/Rosbag_files/short_range_images/2018-11-24-09-25-36_copy/traffic_light_yaml.csv')
# csv_path=('/home/mayank_sati/Documents/datsets/Rosbag_files/short_range_images/color_eval.csv')
# root='/home/mayank_sati/Documents/yaml/GWM_dataset'
root = ''

dataset = pd.read_csv(csv_path)
print(dataset.head())
# dataset=dataset.iloc[np.random.permutation(len(dataset))]
x = dataset.iloc[:, 0].values
y = dataset.iloc[:, 4:8].values
val = 0
bblabel = []
counter = 0
# for loop,xml_aray in zip(x, y):
# this will delete the previous folder and create a new folder
new_filePath = "/home/mayank_sati/Desktop/n"
if os.path.exists(new_filePath):
    print("Output folder present. deleting New folder...")
    shutil.rmtree(new_filePath)
    # os.makedirs(new_filePath)
for loop in x:
    counter += 1
    # if(counter>1135):
    #     break
    print(loop)
    #####################
    # loop=loop.split("/")[-1]
    # image_path = os.path.join(root, loop)
    # [Xmin,ymin,xmax,ymax]
    top = (y[val, 0], y[val, 3])
    bottom = (y[val, 2], y[val, 1])
    # image_path=os.path.join(root,loop+".jpg")

    # @@@@@@@@@@@@@@@@@@@@@@@
    image_path = loop
    try:
        image_scale = cv2.imread(image_path, 1)
        # print(my_image[0,0])
        # image_scale=cv2.resize(m_image,dsize=(800,600), interpolation=cv2.INTER_NEAREST)
        # cv2.rectangle(image_scale, pt1=(125,84),pt2=(159,38),color= (0,255,0), thickness=2)
        cv2.rectangle(image_scale, pt1=top, pt2=bottom, color=(0, 255, 0), thickness=2)
        val += 1
        print(val)

        xmin, ymin, xmax, ymax = y[val, 0], y[val, 1], y[val, 2], y[val, 3]

        # frame = image_scale[int(val[1]):int(val[3] -(val[1])), int(val[0]):int(val[2] -(val[0]))]
        frame = image_scale[int(ymin):int(ymax), int(xmin):int(xmax)]
        ###########################################333
        # ts = time.time()
        # st = datetime.datetime.fromtimestamp(ts).strftime('%d_%m_%Y_%H_%M_%S_%f')
        # cut_frame="/home/mayank_sati/Desktop/crop_image/"
        # cut_frame=cut_frame+"image_"+str(st)+".jpg"
        # cv2.imwrite(cut_frame, frame)
        #########################################################

        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
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

        yellow_range = cv2.inRange(h, 20, 30)
        # counter = np.unique(yellow_range, return_counts=True)
        total_yello = np.count_nonzero(yellow_range)
        # total_yello=0

        # total_red=total_orange+total_red+total_orange

        green_range = cv2.inRange(h, 75, 100)
        # counter = np.unique(green_range, return_counts=True)
        total_green = np.count_nonzero(green_range)

        black_range = cv2.inRange(h, 50, 70)
        # counter = np.unique(cyan_range, return_counts=True)
        total_black = np.count_nonzero(black_range)

        black_sat = cv2.inRange(s, 0, 40)
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
            black_hue_rat = total_black / h.shape[0]
            if sat_ratio > .4 or (black_hue_rat > .6):
                # if total_black_sat > 200:
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
        new_filePath = "/home/mayank_sati/Desktop/n/"
        if not os.path.exists(new_filePath):
            print("Output folder not present. Creating New folder...")
            os.makedirs(new_filePath)
        # # saving light
        # new_filePath = "/home/mayank_sati/Desktop/l/"
        new_filePath = new_filePath + str(light_color) + str(counter) + ".jpg"
        cv2.imwrite(new_filePath, frame)
        #########################################################################
        width = 500
        height = 500
        object_name = light_color
        data_label = [image_path, width, height, object_name, xmin, ymin, xmax, ymax]
        if not ((xmin == xmax) and (ymin == ymax)):
            bblabel.append(data_label)
    except:

        print(image_path)

        # print()
    # if loop>=1018:
    #     break

columns = ['filename', 'width', 'height', 'class', 'xmin', 'ymin', 'xmax', 'ymax']

df = pd.DataFrame(bblabel, columns=columns)

print("into csv file")
df.to_csv('re_square_color.csv', index=False)
