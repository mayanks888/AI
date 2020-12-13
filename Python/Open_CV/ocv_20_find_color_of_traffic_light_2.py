import os

import cv2
import numpy as np
import pandas as pd

root = ""

# csv_path="/home/mayanksati/Documents/Rosbag_files/short_range_images/gwm_data_train.csv"
csv_path = "/home/mayank_sati/pycharm_projects/Data_Science_new/Deep learning/Load Datasets/new_traffic.csv"
# csv_path=('/home/mayanksati/Documents/csv/traffic_light_square.csv')
# csv_path=('/home/mayanksati/PycharmProjects/Data_Science_new/Deep learning/Load Datasets/traffic_light.csv')
# root='/home/mayanksati/PycharmProjects/models/tensorflow/re3-tensorflow-master/demo/GWM_dataset'

dataset = pd.read_csv(csv_path)
print(dataset.head())
# dataset=dataset.iloc[np.random.permutation(len(dataset))]
x = dataset.iloc[:, 0].values
y = dataset.iloc[:, 4:8].values
val = 0
# for loop,xml_aray in zip(x, y):
for loop in x:
    print(loop)
    # [Xmin,ymin,xmax,ymax]
    top = (y[val, 0], y[val, 3])
    bottom = (y[val, 2], y[val, 1])
    # image_path=os.path.join(root,loop+".jpg")
    image_path = os.path.join(root, loop)
    image_scale = cv2.imread(image_path, 1)

    # print(my_image[0,0])
    # image_scale=cv2.resize(m_image,dsize=(800,600), interpolation=cv2.INTER_NEAREST)
    # cv2.rectangle(image_scale, pt1=(125,84),pt2=(159,38),color= (0,255,0), thickness=2)
    cv2.rectangle(image_scale, pt1=top, pt2=bottom, color=(0, 255, 0), thickness=2)
    cv2.circle(image_scale, center=(452, 274), radius=10, color=(0, 255, 0))
    cv2.imshow('streched_image', image_scale)
    # output_folder_path="/home/mayank-s/PycharmProjects/Datasets/aptive/object_detect/output/output_merccedes.png"
    # filepath=output_folder_path+my_image+".png"
    # cv2.imwrite(filepath,my_image)
    cv2.waitKey(100)
    cv2.destroyAllWindows()
    val += 1
    print(val)

    xmin, ymin, xmax, ymax = y[val, 0], y[val, 1], y[val, 2], y[val, 3]

    # frame = image_scale[int(val[1]):int(val[3] -(val[1])), int(val[0]):int(val[2] -(val[0]))]
    frame = image_scale[int(ymin):int(ymax), int(xmin):int(xmax)]
    frame1 = image_scale[100:150, 150:220]


    # cv2.imshow('Image', frame)
    # cv2.waitKey(100)
    # cv2.destroyAllWindows()
    # crop_img = image_scale[0:y + h, x:x + w]
    def find_postive(ind):
        num_zeros = (maskg == 0).sum()
        num_ones = (maskg == 255).sum()
        return num_ones


    ######################################################33333
    # breaking frames done
    # break
    # hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2Lab)
    lower_green = np.array([0, 0, 0])  # scaling upto original pixel value
    upper_green = np.array([255, 112, 150])  # max cr at peak of hist, max cb at 150 to filter out yellow
    maskg = cv2.inRange(hsv, lower_green, upper_green)
    print(find_postive(maskg))
    num_zeros = (maskg == 0).sum()
    num_ones = (maskg == 255).sum()
    # cpy=mask2.copy()
    # print(int(sg[i,0]))
    lower_red = np.array([0, 150, 112])  # scaling upto original pixel value :int(peak2s*10.24)
    upper_red = np.array([255, 255, 255])
    maskr = cv2.inRange(hsv, lower_red, upper_red)
    print(find_postive(maskr))

    lower_yel = np.array([0, 112, 150])  # scaling upto original pixel value:
    upper_yel = np.array([255, 150, 255])
    masky = cv2.inRange(hsv, lower_yel, upper_yel)
    print(find_postive(masky))

    print(np.count_nonzero(y))
    print(1)
