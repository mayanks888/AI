import json
import os

import cv2
import pandas as pd

filename = []
width = []
height = []
Class = []
xmin = []
ymin = []
xmax = []
ymax = []
light_color = []
a = []
file_number = 0
# classes=['bus','light','traffic_sign','person','bike','truck','motor','car','train','Rider']
# classes=['bus','light','traffic light','person','bike','truck','motor','car','train','Rider',"traffic sign"]
classes = ['traffic light']
bblabel = []
loop = 0
jason_path="/home/mayank_s/datasets/bdd/bdd100k_labels_release/bdd100k/labels/bdd100k_labels_images_val.json"
# jason_path = "/home/mayank_s/datasets/bdd/bdd100k_labels_release/bdd100k/labels/bdd100k_labels_images_train.json"
image_path = "/home/mayank_s/datasets/bdd/bdd100k_images/bdd100k/images/100k/val"


if 1:
    data = open(jason_path, 'r')
    data1 = data.read()
    data.close()
    Json = json.loads(data1)
    # filename.append(Json['name'])
    # for obj in root.iter('object'):
    # for ki in Json['frames'][0]['objects']:
    for ki in Json:
        loop += 1
        # print("count run", loop)
        # object_name=ki['category']
        file_name = ki['name']
        # ________________________________________________________________________
        Image_com_path = os.path.join(image_path, file_name)
        exists = os.path.isfile(Image_com_path)
        if exists:
            # if image_name.split('.')==k.split('.'):
            my_image = cv2.imread(Image_com_path, 1)
            # _______________________________________________________________________

            for dat_obj in ki['labels']:
                object_name = dat_obj['category']

                if ki['attributes']['timeofday'] == "daytime":
                    if object_name in classes:
                        sub_class = ki['labels'][0]['attributes']['trafficLightColor']
                        xmin = int(dat_obj['box2d']['x1'])
                        ymin = int(dat_obj['box2d']['y1'])
                        xmax = int(dat_obj['box2d']['x2'])
                        ymax = int(dat_obj['box2d']['y2'])
                        image_width = my_image.shape[1]
                        image_height = my_image.shape[0]

                        ###################################################3333
                        # image_width = 1280
                        # image_height = 720

                        # x-y => Center of the box2d
                        bbox_x = (xmin + xmax) / 2
                        bbox_y = (ymin + ymax) / 2

                        bbox_x_normalized = bbox_x / image_width
                        bbox_y_normalized = bbox_y / image_height

                        bbox_width = abs(xmin - xmax)
                        bbox_height = abs(ymin - ymax)

                        bbox_width_normalized = bbox_width / image_width
                        bbox_height_normalized = bbox_height / image_height

                        print("Center X-Y & Width-Height")
                        print(bbox_x, bbox_y, bbox_width, bbox_height)
                        print("NORMALIZED Center X-Y & Width-Height")
                        print(bbox_x_normalized, bbox_y_normalized, bbox_width_normalized, bbox_height_normalized)
                        #########################################################
                        # coordinate = [xmin, ymin, xmax, ymax, class_num]
                        # object_name=object_name+"_"+light_color
                        object_name = "traffic_light"
                        data_label = [file_name, image_width, image_height, object_name, xmin, ymin, xmax, ymax,bbox_x_normalized, bbox_y_normalized, bbox_width_normalized, bbox_height_normalized,sub_class]
                        # data_label = [file_name, width, height, object_name, xmin, ymin, xmax, ymax]
                        if not ((xmin == xmax) and (ymin == ymax)):
                            bblabel.append(data_label)
                            print("file_name")
                            # print()
                        else:
                            print("found probem with ",file_name)


        else:
            print("file_name")


columns = ['filename', 'width', 'height', 'class', 'xmin', 'ymin', 'xmax', 'ymax','bbox_x_normalized', 'bbox_y_normalized', 'bbox_width_normalized', 'bbox_height_normalized','subclass']


df = pd.DataFrame(bblabel, columns=columns)
df.to_csv('BBD_daytime_train.csv',index=False)
