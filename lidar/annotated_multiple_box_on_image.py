import os

import cv2
import pandas as pd

saving_path = "/home/mayanksati/Documents/point_clouds/KITTI_DATASET_ROOT/pycharm_work/annotated/"
data = pd.read_csv("/home/mayanksati/PycharmProjects/Data_Science_new/lidar/ppillar2.csv")
mydata = data.groupby('filename')
print(data.groupby('class').count())
len_group = mydata.ngroups
mygroup = mydata.groups
new = data.groupby(['filename'])['class'].count()
###############################################3333

x = data.iloc[:, 0].values
y = data.iloc[:, 4:8].values
##################################################
loop = 1
for da in mygroup.keys():
    index = mydata.groups[da].values
    # image_path = os.path.join(root, loop)
    # image_scale = cv2.imread(image_path, 1)
    image_scale = cv2.imread(da, 1)
    for read_index in index:
        print(index)
        top = (y[read_index][0], y[read_index][3])
        bottom = (y[read_index][2], y[read_index][1])
        cv2.rectangle(image_scale, pt1=top, pt2=bottom, color=(0, 255, 0), thickness=2)
        y[read_index][0]
        print(da)
        # break
    cv2.imshow('streched_image', image_scale)
    # output_folder_path="/home/mayank-s/PycharmProjects/Datasets/aptive/object_detect/output/output_merccedes.png"
    # filepath=output_folder_path+my_image+".png"
    # cv2.imwrite(filepath,my_image)
    cv2.waitKey(100)
    output_path = (os.path.join(saving_path, str(loop) + ".jpg"))
    loop += 1
    cv2.imwrite(output_path, image_scale)
    cv2.destroyAllWindows()
    # val += 1
    # print(val)
