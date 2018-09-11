import os
import xml.etree.ElementTree as ET
import numpy as np
import cv2
import pandas as pd


labels = ['Background', 'Car', 'Van', 'Truck', 'Pedestrian', 'Person_sitting', 'Cyclist', 'Tram', 'Misc',
               'DontCare']

def kitti_to_csv(kitti_xml_path,kitti_image_path):
    anotion_info=[]
    for root, _, filenames in os.walk(kitti_xml_path):
        if (len(filenames) == 0):
            print("Input folder is empty")
            #return 1
        for filename in filenames:
            image_name = filename.split(".")[0] + ".png"
            imageNameFile = os.path.join(kitti_image_path, image_name)
            img = cv2.imread(imageNameFile, 1)
            xml_file=os.path.join(kitti_xml_path, filename)
            data_from_text=load_data(image_name,img,xml_file)
            anotion_info.append(data_from_text)
            


def load_data(image_name,img,xml_file):
        bblabel=[]
        
        labels = ['Background', 'Car', 'Van', 'Truck', 'Pedestrian', 'Person_sitting', 'Cyclist', 'Tram', 'Misc',
               'DontCare']
        height = img.shape[0]
        width = img.shape[1]
        channel = img.shape[2]
        #image_name=img#+'.png'
        data_label = []
        for line in open(xml_file):
            data_text=line.strip().split(' ')

            object_name=data_text[0]
            coordinate = data_text[4:8]
            #coordinate.append(class_id)
            data_label = [image_name, width, height, object_name, coordinate[0], coordinate[1], coordinate[2], coordinate[3]]
            # result = [float(c) for c in coordinate]
            # result = np.array(result, dtype=np.int32)
            
            # result = for c in data_label
            # result = np.array(result, dtype=np.int32)
            bblabel.append(data_label)

        return bblabel

def kiiti_to_csv_onw_func(kitti_xml_path,kitti_image_path):
    loop=0
    bblabel=[]
    for root, _, filenames in os.walk(kitti_xml_path):
        if (len(filenames) == 0):
            print("Input folder is empty")
            #return 1
        for filename in filenames:
            image_name = filename.split(".")[0] + ".png"
            imageNameFile = os.path.join(kitti_image_path, image_name)
            img = cv2.imread(imageNameFile, 1)
            height = img.shape[0]
            width = img.shape[1]
            channel = img.shape[2]
            xml_file=os.path.join(kitti_xml_path, filename)
            print(image_name)
            loop+=1
            if loop >7200:
                break
            print (loop)

            for line in open(xml_file):
                data_text=line.strip().split(' ')

                object_name=data_text[0]
                coordinate = data_text[4:8]
                #coordinate.append(class_id)
                data_label = [image_name, width, height, object_name, coordinate[0], coordinate[1], coordinate[2], coordinate[3]]
                bblabel.append(data_label)
    return bblabel
            
            

kitti_xml_path='/home/mayank-s/PycharmProjects/Datasets/kitti_complete/data_object_label_2/training/label_2'
kitti_image_path='/home/mayank-s/PycharmProjects/Datasets/kitti_complete/data_object_image_2/training/image_2'
data2=kiiti_to_csv_onw_func(kitti_xml_path,kitti_image_path)

df=pd.DataFrame(data2)
#df.to_csv('kitti_datasets.csv')
#Image_data,boundingBX_labels,im_dims=prepareBatch(0,2,imageNameFile,vocPath)
# print(Image_data,boundingBX_labels,im_dims)

