#remember the output of json to csv will genreate the bbox in (xmin,ymin,xmax,yman format)
# and from json format it came like ( xmin, ymin , width, height)
import os
import cv2
import pandas as pd
import tqdm
import json
import numpy as np
# csv_path='yolo1.csv'
import random
show=False
bblabel=[]
counter_falied=0
doomed_called=0
already_got=[]
################################################3333
#csv 1
csv_path='crop_tl.csv'
# csv_path='/home/mayank_s/datasets/bdd/training_set/only_csv/front_light_bdd.csv'
# root=         '/home/mayank_s/datasets/nuscene/manipulating_nuscene_from_our_tl_datasets/crop_tl_for_replacement/bosch_bdd_ddtl_farm_our'
# saving_path = "/home/mayank_s/datasets/nuscene/manipulating_nuscene_from_our_tl_datasets/all_front/nuscene_replace"
root=         '/home/mayank_s/datasets/nuscene/manipulating_nuscene_from_our_tl_datasets/crop_tl_for_replacement/bosch_bdd_ddtl_farm_our'
saving_path = "/home/mayank_s/datasets/nuscene/manipulating_nuscene_from_our_tl_datasets/mini/replace_images"
data = pd.read_csv(csv_path)

# # mydata = data.groupby('img_name')
# mydata = data.groupby(['filename'], sort=True)
# # print(data.groupby('class').count())
# len_group = mydata.ngroups
# mygroup = mydata.groups
###############################################3333
#my modification
# df = pd.DataFrame({'column1':['a','s','k','a','a'],'column2':
# [54.2,78.5,89.62,77.2,65.56]})
# k=(df.groupby('column1')['column2'].apply(list))
#
# j=data.groupby('tl_width')['tl_height'].apply(list)
# l=data.groupby('tl_width')['tl_height']['filename'].apply(list)
# df.groupby(["tl_width", "tl_height"])["filename"].count()
# width=14
# height=36
m=data.groupby(["tl_width", "tl_height"])["filename"].apply(list)
#################################################3333
#2nd csv
csv_path='/home/mayank_s/datasets/nuscene/manipulating_nuscene_from_our_tl_datasets/all_front/yolov4_nuscene_all_prediction.csv'
data_2 = pd.read_csv(csv_path)
new_val = data_2.iloc[:].values
for i,values in enumerate(new_val):
    doomed_called = 0
    # if i >2000:
    #     break
    filename, width, height,obj_name ,xmin, ymin, xmax, ymax, score, crop_width, crop_height=values
    print(filename)
#######################################################
    found_flag=False
    while(found_flag==False):
        # crop_width,crop_height=15,30
        for da,ad in zip(m.index,m):
            # found_flag=False
            # print(da,ad)
            if (da[0]==crop_width and da[1]==crop_height):
                print(ad)
                for dummy in range(len(ad)):
                    random_file = random.choice(ad)
                    if not random_file in already_got:
                        print(random_file)
                        already_got.append(random_file)
                        found_flag=True
                        break
            if found_flag==True:
                break
        if found_flag==False:
            counter_falied+=1
            doomed_called +=1
            # if counter_falied>14:
            #     print(1)
            print("doomed",counter_falied)
            # change_par=random.choice([0,1])
            if doomed_called %2==0:
            # if change_par==1:
                crop_width+=1
            else:
                crop_height+=1
            print(crop_width,crop_height)
            if doomed_called>20:
                found_flag = True
        else:
            data_label = [filename, width, height,obj_name ,xmin, ymin, xmax, ymax, score, crop_width, crop_height, random_file]
            bblabel.append(data_label)
        ######################################################

columns = ['filename', 'width', 'height','class' ,'xmin', 'ymin', 'xmax', 'ymax', 'score', 'crop_width', 'crop_height','replace_image']

df = pd.DataFrame(bblabel, columns=columns)
df.to_csv('nuscene_manipulate.csv', index=False)

'''
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
            # cv2.imwrite(base_file_path, image_np)
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

# columns = ['filename', 'tl_width', 'tl_height']
# df = pd.DataFrame(bblabel, columns=columns)
# df.to_csv('crop_tl.csv', index=False)
'''