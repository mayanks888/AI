'''import scipy.imread
img_array = scipy.misc.imread ( "myfile.png" , flatten = True )

import base64

with open("myfile.png", "rb") as imageFile:
    str = base64.b64encode(imageFile.read())
    print(str)

ac=[5,6,8]
# print (ac[]-2)
from PIL import Image
im = Image.open("myfile.png",'r')
pix_val = list(im.getdata(0))
pix_val = (im.getdata())
# print (pix_val.shape)
# pix_val1=pix_val.tolist()
print (len(pix_val))
# print (pix_val.shape)
# print( pix_val-255)
mydata=[]
for data in pix_val:
    print (data)
    # mydata.append(data-255)
print (mydata)'''
import tensorflow as tf
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
im = Image.open('myfile.png')
# pixels = list(im.getdata())
pixels = list(im.getdata(0))
print (pixels)
print (len(pixels))
# width, height = im.size
# pixels = [pixels[i * width:(i + 1) * width] for i in range(height)]
# print (pixels)
mydata=[]
for data in pixels:
    print (data)
    mydata.append(255-data)
print(mydata)
mydata = np.array(mydata, ndmin=2).T
image_array = np.asfarray(mydata.reshape(28, 28))
plt.imshow(image_array, cmap='Greys', interpolation='None')
plt.show()

# @edited to check version


tf.metrics.mean_iou