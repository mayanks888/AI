import csv
import glob
import os

import cv2

NewImagePath = "/home/ubuntu/Chidanand/Traffic_Light/Ground_truth/"
ImagePath = '/home/ubuntu/Downloads/lisa-traffic-light-dataset/'
traffic_light_list = []
image_height = 960
image_width = 1280
iCounter = 0

Annotationfilenames = glob.glob(os.path.join('/home/ubuntu/Downloads/lisa-traffic-light-dataset/Annotations', '*csv'))

for Annotationfilename in Annotationfilenames:
    with open(Annotationfilename, 'r') as fin:
        reader = csv.reader(fin)
        data = list(reader)
    print(Annotationfilename)

    for iIndex in range(len(data) - 1):
        string_parser = data[iIndex + 1]
        string2 = string_parser[0].split(";")
        fileName = ImagePath + string2[0]
        xmin = int(string2[2])
        ymin = int(string2[3])
        xmax = int(string2[4])
        ymax = int(string2[5])
        box_height = ymax - ymin
        box_width = xmax - xmin
        if (box_height < 30 or box_width < 30 or xmin < 200 or xmin > 1000):
            continue
        print(fileName)
        xroimin_new = max(xmin - 5 * box_width, 0)
        yroimin_new = max(ymin - 3 * box_height, 0)
        xroimax_new = min(xmax + 5 * box_width, image_width)
        yroimax_new = min(ymax + 3 * box_height, image_height)
        InpImg = cv2.imread(fileName)
        InpImg_New = InpImg[yroimin_new:yroimax_new, xroimin_new:xroimax_new]
        # cv2.imshow("Input",InpImg)
        # cv2.imshow("ROI", InpImg_New)
        # cv2.waitKey(1)
        iCounter = iCounter + 1
        NewFileName = NewImagePath + str(iCounter) + ".jpg"
        cv2.imwrite(NewFileName, InpImg_New)
        xmin_new = xmin - xroimin_new
        ymin_new = ymin - yroimin_new
        xmax_new = xmax - xroimin_new
        ymax_new = ymax - yroimin_new
        new_imageHeight = ymax_new - ymin_new
        new_imageWidth = xmax_new - xmin_new
        features = {
            'filename': NewFileName,
            'format': 'JPG',
            'width': new_imageWidth,
            'height': new_imageHeight,
            'xmins': xmin_new,
            'xmaxs': xmax_new,
            'ymins': ymin_new,
            'ymaxs': ymax_new,
        }
        traffic_light_list.append(features)

# train_samples = traffic_light_list
# train_record_fname = os.path.join(FLAGS.output_dir, 'train.record')
# create_tf_record(train_record_fname, train_samples)
