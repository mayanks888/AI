import cv2
import numpy as np
import pandas as pd
'''one=np.ones([500,500,3],dtype='uint8')#create all 2 channel image of 1
white_image=one*255#converting complete image into 255(white)

top=(-10,40)
bottom=(20,10)

# cv2.imshow("canvas", white_image)
cv2.rectangle(white_image, pt1=top,pt2=bottom,color= (0,255,0), thickness=2)
cv2.imshow("canvas", white_image)

cv2.waitKey(0)
cv2.destroyAllWindows()'''
#       -83   -39   100    56
#      -175   -87   192   104
#      -359  -183   376   200
#       -55   -55    72    72
#      -119  -119   136   136
#      -247  -247   264   264
#       -35   -79    52    96
#       -79  -167    96   184
#      -167  -343   184   360


# data=pd.read_csv('../../../Datasets/anchorbox.csv')
data=pd.read_csv('/home/mayank-s/Desktop/pytorch_retinanet_my_creation/anchor_34.csv')
features=data.iloc[:,:].values
print(features.shape[0])



# ___________________________________________

'''>xmin>284>/xmin>
>ymin>58>/ymin>
>xmax>363>/xmax>
>ymax>128>/ymax>'''
# top=(284,128)
# bottom=(363,58)
# top=(490,220)
# bottom=(545,178)
# top1=(490+50,220)
# bottom1=(545+50,178)
# bottom=(400,20)
# this is how to draw bounding box
'''x1,y1 ------
    |          |
    |          |
    |          |
    --------x2,y2'''
# my_image=cv2.imread('american_bulldog_135.jpg',1)
one=np.ones([608,1056,3],dtype='uint8')#create all 2 channel image of 1
white_image=one*255#converting complete image into 255(white)
#cv2.imshow('MyImage',my_image)
print(white_image.shape)
a=0
img1=white_image.copy()
counter=0
width_max=1
height_max=1
for loop in range(len(features)):
    img = img1.copy()
    xmin=int(features[loop,0])
    ymin=int(features[loop,1])
    xmax=int(features[loop,2])
    ymax=int(features[loop,3])
    width= abs(xmax-xmin)
    height= abs(ymax-ymin)
    top=(xmin+a,ymax+a)
    bottom=(xmax+a,ymin+a)

    # if (xmin > 0 and  ymin > 0 and xmax > 0 and ymax > 0 and height > 400 and ymax<=608 and xmax<=1056):
    if (xmin > 0 and  ymin > 0 and xmax > 0 and ymax > 0 and height > 400 and ymax<=608 and xmax<=1056):
        cv2.rectangle(img, pt1=top,pt2=bottom,color= (0,255,0), thickness=2)
        print('heigth is {ht} and width is {wt}'.format(ht=height,wt=width))
        counter+=1
        print(counter)
        max=1
        if width_max<=width:
            width_max=width
        if height_max <= height:
            height_max = height
        # cv2.imshow('bounding_box image',white_image)
        cv2.imshow('image' + str(loop), img)
        cv2.waitKey(300)
        cv2.destroyAllWindows()
        cv2.waitKey(10)
        img = None
    # cv2.destroyAllWindows()

print('counter : {ct} :max heigth is {ht} :   max width is {wt}'.format(ct=counter,ht=height_max,wt=width_max))
# _____________________________________________________________

def keep_inside(self, all_anchors, im_info):
    # only keep anchors inside the image
    assert len(im_info) == 1
    im_info = im_info[0]
    inds_inside = np.where(
        (all_anchors[:, 0] >= -self.allowed_border) &
        (all_anchors[:, 1] >= -self.allowed_border) &
        (all_anchors[:, 2] < im_info[1] + self.allowed_border) &  # width
        (all_anchors[:, 3] < im_info[0] + self.allowed_border)  # height
    )[0]

    # keep only inside anchors
    anchors = all_anchors[inds_inside, :]

    return inds_inside, anchors