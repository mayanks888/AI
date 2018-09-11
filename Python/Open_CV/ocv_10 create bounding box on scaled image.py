import cv2
import os
import xml.etree.ElementTree as ET
import numpy as np
# bounding box
labels=['cat','dog']
xmlPath='/home/mayank-s/PycharmProjects/Datasets/single_object_detection/american_bulldog_135.xml'
xmlPath='american_bulldog_135.xml'

def parseXML(xmlPath, labels,pixel_size):
    """
    Args:
      xmlPath: The path of the xml file of this image
      labels: label names of pascal voc dataset
      side: an image is divided into side*side grid
    """
    tree = ET.parse(xmlPath)
    root = tree.getroot()

    width = int(root.find('size').find('width').text)
    height = int(root.find('size').find('height').text)

    for obj in root.iter('object'):
        class_num = labels.index(obj.find('name').text)#finding the index of label mention so as to store data into correct label
        bndbox = obj.find('bndbox')
        xmin = int(bndbox.find('xmin').text)
        ymin = int(bndbox.find('ymin').text)
        xmax = int(bndbox.find('xmax').text)
        ymax = int(bndbox.find('ymax').text)
        h = ymax - ymin
        w = xmax - xmin
        # which cell this obj falls into
        centerx = (xmax + xmin) / 2.0
        centery = (ymax + ymin) / 2.0
        #448 is size of the input image
        newx = (pixel_size / width) * centerx
        newy = (pixel_size / height) * centery

        h_new = h * (pixel_size / height)
        w_new = w * (pixel_size / width)

        new_xmin=int(newx-(w_new/2))
        new_xmax=int(newx+(w_new/2))
        new_ymin=int(newy-(h_new/2))
        new_ymax=int(newy+(h_new/2))



        # print "row,col:",row,col,centerx,centery

        # cell_left = col * cell_size
        # cell_top = row * cell_size
        # cord_x = (newx - cell_left) / cell_size
        # cord_y = (newy - cell_top) / cell_size

        #objif = objInfo(cord_x, cord_y, np.sqrt(h_new / 448.0), np.sqrt(w_new / 448.0), class_num)
        #self.boxes[row][col].has_obj = True
        #self.boxes[row][co

'''<xmin>284</xmin>
<ymin>58</ymin>
<xmax>363</xmax>
<ymax>128</ymax>'''
top=(284,128)
bottom=(363,58)


# bottom=(400,20)
# this is how to draw bounding box
'''x1,y1 ------
    |          |
    |          |
    |          |
    --------x2,y2'''

my_image=cv2.imread('american_bulldog_135.jpg',1)
image_scale=cv2.resize(my_image,dsize=(224,224),interpolation=cv2.INTER_NEAREST)

# this is for stretch image
cv2.rectangle(my_image, pt1=top,pt2=bottom,color= (0,255,0), thickness=2)
# cv2.imshow('bounding_box image',my_image)


parseXML(xmlPath,labels,220)

cv2.rectangle(image_scale, pt1=(125,84),pt2=(159,38),color= (0,255,0), thickness=2)
cv2.imshow('streched image',image_scale)

# '_______________________________________'


cv2.waitKey(0)
cv2.destroyAllWindows()
