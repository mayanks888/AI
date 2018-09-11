import cv2
import numpy as np
color_image=cv2.imread('mercedes.jpg')
# '________________________________________________'
# cv2.namedWindow("cool",cv2.WINDOW_NORMAL)
cv2.imshow('car',color_image)

height,widht,channel=color_image.shape#parsing shape of image
b,g,r=cv2.split(color_image)#spliting imageinto rgb format

#create empty numpy array
rgb_color=np.empty([height,widht*3,channel],dtype='uint8')
# print(rgb_color,rgb_color.shape)
how_merging_works=cv2.merge([b,b,b])#this create one channel blue to 3 dimnsiion blue 
rgb_color[:,0:widht,:]=cv2.merge([b,b,b])#assigning first half of image to blue(check in spyder)
print('my rgb rgb_color[:,0:widht,:]',rgb_color[:,0:widht])
rgb_color[:,widht:widht*2,:]=cv2.merge([g,g,g])
rgb_color[:,widht*2:widht*3,:]=cv2.merge([r,r,r])
# print(b)
cv2.imshow('rgb',rgb_color)
# ____________________________________________________
# change into grey scale
gray_scale=cv2.cvtColor(color_image,cv2.COLOR_RGB2GRAY)
cv2.imshow('GRAy',gray_scale)

#transparency layer
r=color_image[:,:,0]#equivalent to split options
b=color_image[:,:,1]
g=color_image[:,:,2]

rbga_image=cv2.merge([b,g,r,g])
cv2.imwrite('output_merccedes.png',rbga_image)
cv2.imshow('rbga',rbga_image)

# '_______________________________________'
# scaling image
image_stretch=cv2.resize(color_image,dsize=(400,400))
cv2.imshow('stretch',image_stretch)
# scale with interpolation
image_stretchwrt_interpol=cv2.resize(color_image,dsize=(400,400),interpolation=cv2.INTER_NEAREST)
cv2.imshow('inter',image_stretchwrt_interpol)
# '_______________________________________'
#rotation
m=cv2.getRotationMatrix2D(center=(0,0),angle=35,scale=1)#sccale is scalablity
rotated=cv2.warpAffine(color_image,m,dsize=(color_image.shape[1],color_image.shape[1]))#color_image.shape[1] is way to find the width,height of image (try youself)

cv2.imshow('rotated image',rotated)
# '_______________________________________'
cv2.waitKey(0)
cv2.destroyAllWindows()