# segmentation is converting  edge into black and white
import numpy as np
import cv2
b_w_image=cv2.imread('mercedes.jpg',0)
height,width=b_w_image.shape[0:2]
b_w_image = cv2.resize(b_w_image, dsize=(0, 0), fx=.25, fy=.25)  # resize frame


# '____________________________________________________________________________________'
cv2.imshow('black',b_w_image)
binary_val=np.zeros([height,width,1],dtype='uint8')
threshold_val=86
ret,thres=cv2.threshold(b_w_image,threshold_val,255,cv2.THRESH_BINARY)#converted into segmented value
cv2.imshow('Segmented image',thres)


# This is like a direct method of converting image into segmentation without using any function
# use it for better understanding but need to work in this as this was not working correctly
'''for row in range(0,height):
    for col in range(0,width):
       if (binary_val[row][col]>threshold_val):
           binary_val[row][col]=255

# b_w_image = cv2.resize(binary_val, dsize=(0, 0), fx=.25, fy=.25)  # resize frame
# cv2.imshow('Segmented image',binary_val)'''
# ')_________________________________________________________________________________/'
# adaptive threshold for segmentation
thresh_adapt=cv2.adaptiveThreshold(b_w_image,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,115,1)
cv2.imshow('adaptive segmentation image',thresh_adapt)
cv2.waitKey(0)
cv2.destroyAllWindows()
