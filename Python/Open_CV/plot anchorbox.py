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


data=pd.read_csv('../../../Datasets/anchorbox.csv')
features=data.iloc[:,:].values



# ___________________________________________

'''<xmin>284</xmin>
<ymin>58</ymin>
<xmax>363</xmax>
<ymax>128</ymax>'''
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
one=np.ones([1000,1000,3],dtype='uint8')#create all 2 channel image of 1
white_image=one*255#converting complete image into 255(white)
#cv2.imshow('MyImage',my_image)
print(white_image.shape)
a=380
for loop in range(len(features)):
    xmin=features[loop,0]
    ymin=features[loop,1]
    xmax=features[loop,2]
    ymax=features[loop,3]
    top=(xmin+a,ymax+a)
    bottom=(xmax+a,ymin+a)
    cv2.rectangle(white_image, pt1=top,pt2=bottom,color= (0,255,0), thickness=2)

cv2.imshow('bounding_box image',white_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
# _____________________________________________________________