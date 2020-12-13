#remember the output of json to csv will genreate the bbox in (xmin,ymin,xmax,yman format)
# and from json format it came like ( xmin, ymin , width, height)
import os
import cv2
import pandas as pd
import tqdm
import json
import numpy as np
# csv_path='yolo1.csv'
show=False
csv_path='//home/mayank_s/datasets/nuscene/manipulating_nuscene_from_our_tl_datasets/bosch_bdd_ddtl_farm.csv'
# csv_path='/home/mayank_s/datasets/bdd/training_set/only_csv/front_light_bdd.csv'
root='/home/mayank_s/datasets/bdd_bosch/data/images/train_val'
saving_path = "/home/mayank_s/datasets/nuscene/manipulating_nuscene_from_our_tl_datasets/bosch_bdd_ddtl_farm_our"
data = pd.read_csv(csv_path)
# mydata = data.groupby('img_name')
mydata = data.groupby(['filename'], sort=True)
# print(data.groupby('class').count())
len_group = mydata.ngroups
mygroup = mydata.groups
###############################################3333
x = data.iloc[:, 0].values
y = data.iloc[:, 4:8].values
z = data.iloc[:, -1].values
##################################################
loop = 0
bblabel=[]
images = list()
annotations = list()
counter=0
counter2=-1
attr_dict = dict()
attr_dict["categories"] = [
    {"supercategory": "none", "id": 1, "name": "traffic_light"}]
# "person","rider","car","bus","truck","bike","motor", "traffic light","traffic sign","train"
attr_id_dict = {i['name']: i['id'] for i in attr_dict['categories']}
for ind, da1 in enumerate(sorted(mygroup.keys())):
    counter2=0
    loop += 1
    # print(da1)
    index = mydata.groups[da1].values
    ###########33
    counter += 1
    # if counter > 10:
    #     # break
    # 1
    print(counter)


    ##############3
    da = os.path.join(root, da1)
    # image_scale = cv2.imread(image_path, 1)
    image_scale = cv2.imread(da, 1)
    try:
        height = image_scale.shape[0]
        width = image_scale.shape[1]
        # print(height, width)

        #######################333
        image = dict()
        image['file_name'] = da1.split("/")[-1]
        # print(image['file_name'])
        # image['file_name'] = da1
        image['height'] = height
        image['width'] = width
        image['id'] = counter
        empty_image = True
        ##############################
        for read_index in index:
            counter2 += 1

            #######################################################
            xmin=y[read_index][0]
            ymin=y[read_index][1]
            xmax =y[read_index][2]
            ymax=y[read_index][3]
            image_np = image_scale[ymin:ymax, xmin:xmax]
            tl_width=int(y[read_index][2]-y[read_index][0])
            tl_height=int(y[read_index][3]-y[read_index][1])
            #########################################################
            new_img_name=image['file_name'].split(".")[0]
            new_img_name=new_img_name + "_" + str(counter2) + '.' + image['file_name'].split(".")[-1]
            base_file_path = (os.path.join(saving_path, new_img_name))
            cv2.imwrite(base_file_path, image_np)
            ############################################################################
            data_label = [new_img_name, tl_width, tl_height]
            # if xmin < 0 or ymin < 0:
            #     # if xmin < 0 or ymin < 0 or xmax>500 or ymax>500:
            #     print(filename)
            #     continue
            bblabel.append(data_label)
        #

            ######################################################################
            # print(index)
            top = (int(y[read_index][0]), int(y[read_index][3]))
            bottom = (int(y[read_index][2]), int(y[read_index][1]))
            cv2.rectangle(image_scale, pt1=top, pt2=bottom, color=(0, 255, 0), thickness=2)
            cv2.putText(image_scale, str(z[read_index]),
                        (int((y[read_index][0] + y[read_index][2]) / 2), int(y[read_index][1] - 10)),
                        cv2.FONT_HERSHEY_SIMPLEX, .5, (255, 255, 0), lineType=cv2.LINE_AA)

            ###############################################333
            if show:

                # cv2.imshow('streched_image', image_scale)
                cv2.imshow('streched_image', image_np)
                ch = cv2.waitKey(1000)
                if ch & 0XFF == ord('q'):
                    cv2.destroyAllWindows()
    except:
        print(da1)

columns = ['filename', 'tl_width', 'tl_height']

df = pd.DataFrame(bblabel, columns=columns)
df.to_csv('crop_tl.csv', index=False)


##########################3333
