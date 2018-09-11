import cv2
import numpy as np
import pandas as pd
import OCV_8_mean_IOu
# bounding box
# data=pd.read_csv('../../../Datasets/anchor_inside_image.csv')

data=pd.read_csv('../../../Datasets/anchor_inside_image.csv')

features=data.iloc[:,:].values

xmin1=316
ymin1=59
xmax1=578
ymax1=220
top1=(xmin1,ymax1)
bottom1=(xmax1,ymin1)

xmin2=106
ymin2=118
xmax2=942
ymax2=570
top2=(xmin2,ymax2)
bottom2=(xmax2,ymin2)




'''<xmin>284</xmin>
<ymin>58</ymin>
<xmax>363</xmax>
<ymax>128</ymax>'''

# this is how to draw bounding box
'''x1,y1 ------
    |          |
    |          |
    |          |
    --------x2,y2'''
# my_image=cv2.imread('american_bulldog_135.jpg',1)
my_image=cv2.imread('2008_000008.jpg',1)
#cv2.imshow('MyImage',my_image)
image_scale=cv2.resize(my_image,dsize=(1000,600),interpolation=cv2.INTER_NEAREST)
print(my_image.shape)
my_image=image_scale


total=0
for loop in range(1500,len(features)):
# for loop in range(2):
    # img = my_image.c
    img=my_image.copy()
    # img=copy(my_image)
    xmin=features[loop,0]
    ymin=features[loop,1]
    xmax=features[loop,2]
    ymax=features[loop,3]
    top=(xmin,ymax)
    bottom=(xmax,ymin)

    cv2.rectangle(img, pt1=top, pt2=bottom, color=(0, 255, 0), thickness=2)
    cv2.rectangle(img, pt1=top1, pt2=bottom1, color=(255, 0, 0), thickness=2)
    cv2.rectangle(img, pt1=top2, pt2=bottom2, color=(0, 0, 255), thickness=2)
    # cv2.namedWindow('image', cv2.WINDOW_NORMAL)
    # cv2.namedWindow("image", cv2.WND_PROP_FULLSCREEN)
    # cv2.setWindowProperty("image",cv2.WND_PROP_FULLSCREEN,cv2.WINDOW_FULLSCREEN)
    first_iou=OCV_8_mean_IOu.bb_intersection_over_union([xmin,ymin,xmax,ymax],[xmin1,ymin1,xmax1,ymax1])#blue Small box
    second_iou=OCV_8_mean_IOu.bb_intersection_over_union([xmin,ymin,xmax,ymax],[xmin2,ymin2,xmax2,ymax2])#red bigger box
    if (first_iou >0 or second_iou>0):
        print ('small box iou anchor no ',loop+1,first_iou)
        print('bigger box iou anchor no', loop + 1, second_iou)
        total=total+1
        cv2.imshow('image'+str(loop), img)

        # '_______________________________________'
        cv2.waitKey(500)
        cv2.destroyAllWindows()
        cv2.waitKey(10)
    img = None
print (total)
# ffffff