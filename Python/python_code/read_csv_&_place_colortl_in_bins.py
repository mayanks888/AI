import os
import cv2
import pandas as pd
# import tkMessageBox
# import tkSimpleDialog
import time
from datetime import datetime

bblabel = []
root = "/home/mayank_s/datasets/bdd/bdd100k_images/bdd100k/images/100k/val"
# csv_path = '/home/mayank_s/datasets/bdd/training_set/only_csv/front_light_bdd.csv'
csv_path = '/home/mayank_s/codebase/AI/Machine_learning/Deep learning/Load Datasets/data_format_yolo/BBD_daytime_val.csv'
data = pd.read_csv(csv_path)
mydata = data.groupby(['filename'], sort=True)
output_path_crop="/home/mayank_s/datasets/color_tl_datasets/BDD/part_All"
###############################################3333
all_data=data.iloc[:,:].values
x = data.iloc[:, 0].values
y = data.iloc[:, 4:8].values
##################################################
flag=False
loop=0
for dat in all_data:
    loop+=1
    print(loop)
    file_name=dat[0]
    print(loop)
    # if file_name =='0b351bc5-0fd3d464.jpg':
    flag=True
    if flag:
        top=(dat[4],dat[7])
        bottom=(dat[6],dat[5])
        image_path = os.path.join(root, file_name)
        image_np = cv2.imread(image_path, 1)
        # cv2.rectangle(image_scale, pt1=top, pt2=bottom, color=(0, 255, 0), thickness=2)
        # cv2.circle(image_scale, center=(452, 274), radius=10, color=(0, 255, 0))
        # cv2.putText(image_scale, dat[12], (500, 500), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), lineType=cv2.LINE_AA)

        # cv2.imshow('streched_image', image_np)
        # ch = cv2.waitKey(100)
        # # mychoice = id_option_more()
        # # print(mychoice)
        # if ch & 0XFF == ord('q'):
        #     cv2.destroyAllWindows()
        # cv2.destroyAllWindows()
        (xmin,ymin,xmax,ymax)=(dat[4],dat[5],dat[6],dat[7])
        width=xmax-xmin
        height=ymax-ymin
        lim=25
        print (width,height)
        if width<lim or height < lim:
            continue
        image_np_crop = image_np[ymin:ymax, xmin:xmax]
        cv2.imshow('streched_image', image_np_crop)
        ch = cv2.waitKey(10)
        # output_path_crop1 = (os.path.join(self.output_path_crop, filename))
        ts = time.time()
        st = datetime.fromtimestamp(ts).strftime('%d_%m_%Y_%H_%M_%S_%f')
        new_file_name = str(st) + ".jpg"


        fd=dat[12]
        if fd=="green":
            color_path="green"
        elif fd =="red":
            color_path='red'
        elif fd == "yellow":
            color_path="yellow"
        elif fd == "black":
            color_path="black"
        else:
            color_path = "unknown"
        output_path_crop1 = (os.path.join(output_path_crop,color_path, new_file_name))
        cv2.imwrite(output_path_crop1, image_np_crop)
