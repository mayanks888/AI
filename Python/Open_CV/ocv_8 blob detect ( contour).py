import cv2
import numpy as np

image=cv2.imread("blob_detect.png",1)
# image=cv2.imread("blob_detect2.jpg",1)
gray_scale=cv2.cvtColor(image,cv2.COLOR_RGB2GRAY)
cv2.imshow('gray',gray_scale)
thresh=cv2.adaptiveThreshold(gray_scale,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,115,1)
cv2.imshow('segment',thresh)


_,contours,hierarchy=cv2.findContours(image=thresh,mode=cv2.RETR_TREE,method=cv2.CHAIN_APPROX_SIMPLE)

image2=image.copy()
index=-1
line_thickness=1
cont_color=(0,255,0)
cv2.drawContours(image2,contours,color=cont_color,thickness=line_thickness,contourIdx=index)
cv2.imshow('contours',image2)

# _____________________________________________________
# this is to find the area and centre of contour
my_object=np.zeros(shape=(image.shape[0],image.shape[1],3),dtype='uint8')
for c in contours:
    # cv2.drawContours(my_object,[c],-1,color=cont_color,-1)
    cv2.drawContours(my_object, contours, color=cont_color, thickness=line_thickness, contourIdx=index)
    area=cv2.contourArea(c)#get the area and parameter of all shape in the image
    perimeter=cv2.arcLength(c,True)
    print('area,perimeter',area,perimeter)

cv2.waitKey(0)
cv2.destroyAllWindows()
