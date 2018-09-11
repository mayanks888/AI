import cv2
import numpy as np

black=np.zeros([150,200,1],dtype='uint8')#creating 3d array array of 150 widht ,200 hieght , and 1 channel
print(black)
print(black.shape)
#cv2.imshow('Black',black)

one=np.ones([150,200,1],dtype='uint8')

#creating image out of black array
# cv2.imshow('black2',one)#nothing will happen with ones

white=one*255#chanhing my array to val to 255 which is equalent to 255
#cv2.imshow('white',white)
# creating color array of image
create_color=np.random.random_integers(low=1,high=244,size=[4608,2048,3])
creare_color=np.array(create_color)
print(create_color)
print ('shape of color image is',create_color.shape)
print('pixel value',create_color[1,2,0])
#cv2.imshow('color image',create_color)
# cv2.imwrite('output1.jpg',create_color)
cv2.waitKey(5000)
cv2.destroyAllWindows()
