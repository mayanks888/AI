import cv2
import numpy as np
#loading inmage
img=cv2.imread('dog1.jpg')
# creating working windows
'''cv2.namedWindow("cool",cv2.WINDOW_NORMAL)
# @show your image
cv2.imshow("image",img)
# waitKEy is showing window in milisecond
cv2.waitKey(1000)
#saving open cv image
cv2.imwrite('output.jpg',img)'''

# ______________________________________________
# geting pixel info of image
new_img=cv2.imread('mercedes.jpg',1)#1 in arguments indicated loading the default image check for more options
print(new_img)#change to pixel of each image
print('width of image',len(new_img))#pixel hieight
print('height of image',len(new_img[0]))#image pixel width
print('no of channel',len(new_img[0][0]))#channel in image
print("shape of image",new_img.shape)#shape of image
print(new_img[3,4])#value of pixel in 3 row and 5 column
print(new_img[3,4,2])#value of pixel in 3 row and 5 column and 2 nd channel
print(new_img[:,:,0])#all value in channel 0
print('type of image',type(new_img))
print ('total no of pixel in image',new_img.size)