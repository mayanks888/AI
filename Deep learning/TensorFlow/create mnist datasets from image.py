
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
im = Image.open('3_3_pic.png')
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
    # data=str(data)
    mydata.append(255-data)
    # mydata(data).replace(' ', '')
print(mydata)
mydata = np.array(mydata, ndmin=2).T
image_array = np.asfarray(mydata.reshape(28, 28))
plt.imshow(image_array, cmap='Greys', interpolation='None')
plt.show()