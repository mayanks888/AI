import cv2
import numpy as np
# bounding box
'''<xmin>284</xmin>
<ymin>58</ymin>
<xmax>363</xmax>
<ymax>128</ymax>'''
# top=(284,128)
# bottom=(363,58)
# 396	264	579	359
#
# 304	216	671	407

xmin4=256
ymin4 =40
xmax4  =623
ymax4    =231
top4=(xmin4,ymax4)
bottom4=(xmax4,ymin4)

xmin=304
ymin=216
xmax=671
ymax=407
top=(xmin,ymax)
bottom=(xmax,ymin)



xmin2=106
ymin2=118
xmax2=942
ymax2=570
top2=(xmin2,ymax2)
bottom2=(xmax2,ymin2)

xmin3=316
ymin3=59
xmax3=578
ymax3=220
top3=(xmin3,ymax3)
bottom3=(xmax3,ymin3)


# top=(53,420)
# bottom=(471,87)
# top1=(106,540)
# bottom1=(942,118)
# bottom=(400,20)
# this is how to draw bounding box
'''x1,y1 ------
    |          |
    |          |
    |          |
    --------x2,y2'''
# my_image=cv2.imread('american_bulldog_135.jpg',1)
my_image=cv2.imread('2008_000008.jpg',1)
# cv2.rectangle(my_image, pt1=top,pt2=bottom,color= (0,255,0), thickness=2)
# cv2.imshow('MyImage',my_image)

image_scale=cv2.resize(my_image,dsize=(1000,600),interpolation=cv2.INTER_NEAREST)
cv2.rectangle(image_scale, pt1=top,pt2=bottom,color= (0,255,0), thickness=2)
cv2.rectangle(image_scale, pt1=top2,pt2=bottom2,color= (0,255,0), thickness=2)
cv2.rectangle(image_scale, pt1=top3,pt2=bottom3,color= (0,255,0), thickness=2)
cv2.rectangle(image_scale, pt1=top4,pt2=bottom4,color= (0,255,0), thickness=2)  
cv2.imshow('MyImage',image_scale)




# cv2.imshow('bounding_box image',my_image)

# '_______________________________________'
cv2.waitKey(0)
cv2.destroyAllWindows()