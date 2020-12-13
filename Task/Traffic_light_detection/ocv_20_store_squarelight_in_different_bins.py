import datetime
import os
import time

import cv2
import pandas as pd

# from PIL import Image
ts = time.time()
# root="/home/mayank_sati/Documents/datsets/Rosbag_files/short_range_images/2018-11-24-09-44-06/Baidu_TL_dataset1"

# csv_path="/home/mayanksati/Documents/Rosbag_files/short_range_images/gwm_data_train.csv"
# csv_path="/home/mayank_sati/pycharm_projects/Data_Science_new/Deep learning/Load Datasets/new_traffic.csv"
# csv_path=('/home/mayanksati/Documents/csv/traffic_light_square.csv')
# csv_path=('/home/mayank_sati/Documents/datsets/Rosbag_files/short_range_images/2018-11-24-09-45-27/re3_5.csv')
# csv_path=('/home/mayank_sati/Documents/datsets/Rosbag_files/short_range_images/2018-11-24-09-31-33/re3_3.csv')
csv_path = ('/home/mayank_sati/pycharm_projects/Data_Science_new/Task/Traffic_light_detection/re_square_color.csv')
# root='/home/mayanksati/PycharmProjects/models/tensorflow/re3-tensorflow-master/demo/GWM_dataset'
root = ''
dataset = pd.read_csv(csv_path)
print(dataset.head())
# dataset=dataset.iloc[np.random.permutation(len(dataset))]
x = dataset.iloc[:, 0].values
y = dataset.iloc[:, 4:8].values
clas = dataset.iloc[:, 3].values
val = 0
bblabel = []
counter = 0
# for loop,xml_aray in zip(x, y):
for loop in x:
    counter += 1
    # if(counter>1135):
    #     break
    # print(loop)
    print(val)
    #####################
    image_name = loop.split("/")[-1]
    # [Xmin,ymin,xmax,ymax]
    #     top = (y[val,0], y[val,3])
    #     bottom = (y[val,2], y[val,1])
    # image_path=os.path.join(root,loop+".jpg")
    image_path = os.path.join(root, loop)
    image_scale = cv2.imread(image_path, 1)

    # print(my_image[0,0])
    # image_scale=cv2.resize(m_image,dsize=(800,600), interpolation=cv2.INTER_NEAREST)
    # cv2.rectangle(image_scale, pt1=(125,84),pt2=(159,38),color= (0,255,0), thickness=2)
    # cv2.rectangle(image_scale, pt1=top, pt2=bottom, color=(0, 255, 0), thickness=2)
    # cv2.circle(image_scale,center=(452,274),radius=10,color=(0,255,0))
    # cv2.imshow('streched_image',image_scale)
    # output_folder_path="/home/mayank-s/PycharmProjects/Datasets/aptive/object_detect/output/output_merccedes.png"
    # filepath=output_folder_path+my_image+".png"
    # cv2.imwrite(filepath,my_image)
    # cv2.waitKey(100)
    # cv2.destroyAllWindows()
    # val+=1
    # print (val)

    xmin, ymin, xmax, ymax = y[val, 0], y[val, 1], y[val, 2], y[val, 3]

    # frame = image_scale[int(val[1]):int(val[3] -(val[1])), int(val[0]):int(val[2] -(val[0]))]
    frame = image_scale[int(ymin):int(ymax), int(xmin):int(xmax)]

    if clas[val] == 'red_light':
        image_save_path = "/home/mayank_sati/Desktop/sorting_light/final_sort/red/"
    elif clas[val] == 'green_light':
        image_save_path = "/home/mayank_sati/Desktop/sorting_light/final_sort/green/"
    elif clas[val] == 'yellow_light':
        image_save_path = "/home/mayank_sati/Desktop/sorting_light/final_sort/yellow/"
    else:
        image_save_path = "/home/mayank_sati/Desktop/sorting_light/final_sort/black/"

    ts = time.time()
    st = datetime.datetime.fromtimestamp(ts).strftime('%d_%m_%Y_%H_%M_%S_%f')
    # cut_frame="/home/mayank_sati/Desktop/crop_image/"
    # cut_frame=cut_frame+"image_"+str(st)+".jpg"
    # cv2.imwrite(cut_frame, frame)

    color_path = image_save_path + str(st) + image_name
    print(color_path)
    cv2.imwrite(color_path, frame)

    val += 1
