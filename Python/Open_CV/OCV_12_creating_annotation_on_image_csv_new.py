import os

import cv2
import pandas as pd

# dataset = pd.read_csv("SortedXmlresult.csv")
# /home/mayank-s/PycharmProjects/Datasets/Berkely DeepDrive/bdd100k/images/100k/train
# /home/mayank-s/PycharmProjects/Datasets/Berkely DeepDrive/berkely_train.csv
# csv_path="/home/mayanksati/PycharmProjects/Data_Science_new/lidar/point_csv/pointpil.csv"
# csv_path="/home/mayanksati/PycharmProjects/Data_Science_new/Python/python_code/combined.csv"
# root="/home/mayanksati/Documents/Rosbag_files/short_range_images/gwm_data_train.csv"
# root="/home/mayanksati/Documents/datasets/BDD/ubuntu_aws_instance/Mayank_datastore/rosbag_images/2018-11-24-09-48-37/Baidu_TL_dataset1"
root = ""
# csv_path="/home/mayanksati/Documents/Rosbag_files/short_range_images/gwm_data_train.csv"
csv_path = "/home/mayank_sati/Documents/datsets/Rosbag_files/short_range_images/square_light_combined.csv"
# csv_path="/home/mayank_sati/Documents/datsets/Rosbag_files/square data/square_data0-1/square01.csv"
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

z = dataset.iloc[:, 3].values
for loop in x:
    print(loop)
    # [Xmin,ymin,xmax,ymax]
    top = (y[val, 0], y[val, 3])
    bottom = (y[val, 2], y[val, 1])
    # image_path=os.path.join(root,loop+".jpg")
    image_path = os.path.join(root, loop)
    try:
        image_scale = cv2.imread(image_path, 1)
        # print(my_image[0,0])
        # image_scale=cv2.resize(m_image,dsize=(800,600), interpolation=cv2.INTER_NEAREST)
        # cv2.rectangle(image_scale, pt1=(125,84),pt2=(159,38),color= (0,255,0), thickness=2)
        cv2.rectangle(image_scale, pt1=top, pt2=bottom, color=(0, 255, 0), thickness=2)
        cv2.circle(image_scale, center=(452, 274), radius=10, color=(0, 255, 0))
        cv2.putText(image_scale, z[val], (20, 20), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), lineType=cv2.LINE_AA)
        # output_folder_path="/home/mayank-s/PycharmProjects/Datasets/aptive/object_detect/output/output_merccedes.png"
        # filepath=output_folder_path+my_image+".png"
        # cv2.imwrite(filepath,my_image)
        # font = cv2.FONT_HERSHEY_SIMPLEX
        # cv2.putText(image_scale, z[val], (100, 200), font, 4, (0, 255, 0), 2, cv2.LINE_AA)
        if val > 7377:
            # if val>1:
            print(200)
            # cv2.waitKey(1000)
            cv2.imshow('streched_image', image_scale)

            ch = cv2.waitKey(200)  # refresh after 1 milisecong
            if ch & 0XFF == ord('q'):
                cv2.destroyAllWindows()
            cv2.destroyAllWindows()
        val += 1
        print(val)
    # break
    except:
        print(image_path)
        val += 1
