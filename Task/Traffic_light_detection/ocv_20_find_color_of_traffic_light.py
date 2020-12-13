import os

import cv2
import numpy as np
import pandas as pd

root = "/home/mayank_sati/Documents/datsets/Rosbag_files/short_range_images/2018-11-24-09-45-27/Baidu_TL_dataset1"

# csv_path="/home/mayanksati/Documents/Rosbag_files/short_range_images/gwm_data_train.csv"
# csv_path="/home/mayank_sati/pycharm_projects/Data_Science_new/Deep learning/Load Datasets/new_traffic.csv"
# csv_path=('/home/mayanksati/Documents/csv/traffic_light_square.csv')
csv_path = ('/home/mayank_sati/Documents/datsets/Rosbag_files/short_range_images/2018-11-24-09-45-27/re3_5.csv')
# root='/home/mayanksati/PycharmProjects/models/tensorflow/re3-tensorflow-master/demo/GWM_dataset'

dataset = pd.read_csv(csv_path)
print(dataset.head())
# dataset=dataset.iloc[np.random.permutation(len(dataset))]
x = dataset.iloc[:, 0].values
y = dataset.iloc[:, 4:8].values
val = 0
bblabel = []
counter = 0
# for loop,xml_aray in zip(x, y):
for loop in x:
    counter += 1
    if (counter > 1665):
        break
    print(loop)
    #####################
    loop = loop.split("/")[-1]
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
    # cv2.imshow('streched_image',image_scale)
    # output_folder_path="/home/mayank-s/PycharmProjects/Datasets/aptive/object_detect/output/output_merccedes.png"
    # filepath=output_folder_path+my_image+".png"
    # cv2.imwrite(filepath,my_image)
    # cv2.waitKey(100)
    # cv2.destroyAllWindows()
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

    ######################################################33333
    # breaking frames done
    # break
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    # h=h*2
    red_range = cv2.inRange(h, 0, 21)
    # counter=np.unique(red_range,return_counts=True)
    total_red = np.count_nonzero(red_range)
    # tot=np.count_nonzero(red_range)

    orange_range = cv2.inRange(h, 22, 44)
    # counter = np.unique(orange_range, return_counts=True)
    total_orange = np.count_nonzero(orange_range)

    yellow_range = cv2.inRange(h, 45, 70)
    # counter = np.unique(yellow_range, return_counts=True)
    total_yello = np.count_nonzero(yellow_range)

    green_range = cv2.inRange(h, 70, 155)
    # counter = np.unique(green_range, return_counts=True)
    total_green = np.count_nonzero(green_range)

    cyan_range = cv2.inRange(h, 155, 186)
    # counter = np.unique(cyan_range, return_counts=True)
    total_cyan = np.count_nonzero(cyan_range)

    blue_range = cv2.inRange(h, 186, 278)
    # counter = np.unique(blue_range, return_counts=True)
    total_blue = np.count_nonzero(blue_range)

    purple_range = cv2.inRange(h, 278, 330)
    # counter = np.unique(purple_range, return_counts=True)
    total_purple = np.count_nonzero(purple_range)

    last = cv2.inRange(h, 330, 360)
    # counter = np.unique(last, return_counts=True)
    other = np.count_nonzero(last)

    # data_label = [image_path, total_red, total_orange, total_yello, total_green, total_cyan, total_blue, total_purple,other]
    data_label = [image_path, total_red, total_orange, total_yello, total_green, round((total_red / total_yello), 3),
                  round((total_orange / total_yello), 3), round((total_green / total_yello), 3), total_cyan, total_blue,
                  total_purple, other]

    if not ((xmin == xmax) and (ymin == ymax)):
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
df.to_csv('colormaps27.csv', index=False)

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
