import pandas as pd
import yaml
import cv2
import os
import time
import time
import datetime
from datetime import datetime
col_sav_path='/home/mayank_s/datasets/color_tl_datasets/bosch'
filename = []
width = 0
height = 0
Class = []
xmin = []
ymin = []
xmax = []
ymax = []
light_color = []
bblabel = []
loop = 1
image_path = "/home/mayank_s/datasets/bdd_bosch/data/images/val"
object_name = "traffic_light"
# yaml_path = "/home/mayank_s/codebase/others/yolo/bstld/label_files/test.yaml"
# yaml_path = "/home/mayank_s/datasets/bosch/main/test.yaml"
yaml_path = "/home/mayank_s/datasets/bosch/test.yaml"
with open(yaml_path, 'r') as ymlfile:
    # with open("/home/mayanksati/PycharmProjects/models/tensorflow/re3-tensorflow-master/demo/mayank_first.yaml", 'r') as ymlfile:
    cfg = yaml.load(ymlfile)

for i ,section in enumerate(cfg):
    print(i)
    # if section['boxes']==None:
    if section['boxes'].__len__()==0:
        continue
    #     print(section)
    for boxes in section['boxes']:
        try:
            # print(boxes)
            loop += 1
            xmax = int(boxes['x_max'])
            xmin =  int(boxes['x_min'])
            ymax =  int(boxes['y_max'])
            ymin =  int(boxes['y_min'])
            label=boxes['label']
            # height = section['boxes'][0]['boxes']['height']
            # width = section['boxes'][0]['boxes']['width']
            # ___________________________________________
            # playing with file
            file_path = section['path']
            img_name = file_path.split("/")[-1]
            ########################################
            # img_name=img_name.split(".png")[0]
            # first = time.time()
            # # image_np = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)
            # ts = time.time()
            # st = datetime.fromtimestamp(ts).strftime('%d_%m_%Y_%H_%M_%S_%f')
            # # color_path = image_save_path + filename.split("/")[-3] + filename.split("/")[-1]
            # img_name = img_name + str(st) + ".jpg"
            #########################################
            file_name = file_path.split("/")[-1]
            new_path = image_path + "/" + file_name
            file_path = new_path
            my_image = cv2.imread(new_path, 1)

            width = my_image.shape[1]

            height = my_image.shape[0]

            data_label = [img_name, width, height, object_name, xmin, ymin, xmax, ymax]
            if not ((xmin == xmax) and (ymin == ymax)):
                bblabel.append(data_label)
                # print()
            # if loop>=1018:
            #     break
        except:
            print("does not exist",file_path)

columns = ['filename', 'width', 'height', 'class', 'xmin', 'ymin', 'xmax', 'ymax']

df = pd.DataFrame(bblabel, columns=columns)
print("into csv file")
df.to_csv('bosch_val.csv', index=False)
