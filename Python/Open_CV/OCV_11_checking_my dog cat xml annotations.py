import cv2
import os
import xml.etree.ElementTree as ET
import  pandas as pd


# dataset = pd.read_csv("SortedXmlresult.csv")

dataset = pd.read_csv("SortedXmlresult_linux.csv")
x = dataset.iloc[:, 1].values
y = dataset.iloc[:, 3:8].values
val=0
#for loop,xml_aray in zip(x, y):
for loop in x:
# [Xmin,ymin,xmax,ymax]
    top = (y[val,0], y[val,3])
    bottom = (y[val,2], y[val,1])

    my_image=cv2.imread(loop,1)
    image_scale=cv2.resize(my_image,dsize=(224,224),interpolation=cv2.INTER_NEAREST)
    # cv2.rectangle(image_scale, pt1=(125,84),pt2=(159,38),color= (0,255,0), thickness=2)
    cv2.rectangle(image_scale, pt1=top, pt2=bottom, color=(0, 255, 0), thickness=2)
    cv2.imshow('streched image',image_scale)
    cv2.waitKey(1000)
    cv2.destroyAllWindows()
    val+=1