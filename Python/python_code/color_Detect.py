# import cv2
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

image = Image.open('tl.jpg')

# box = 233, 261, 301, 288
# top, left, bottom, right = box
# box = 254, 591, 312, 618
# box=156,99,180,160
box = 99, 156, 160, 180
box = 129, 236, 245, 276

top, left, bottom, right = box
box_colour_detect = left, top, right, bottom
roi = image.crop(box_colour_detect)
R, G, B = roi.split()

R_mean = np.mean(R)
G_mean = np.mean(G)
B_mean = np.mean(B)

print(R_mean, G_mean, B_mean)

ax = plt.subplot(2, 3, 1)
ax.imshow(image)
ax = plt.subplot(2, 3, 2)
ax.imshow(roi)
ax = plt.subplot(2, 3, 4)
ax.set_title('R')
ax.imshow(R, cmap='binary')
ax = plt.subplot(2, 3, 5)
ax.set_title('G')
ax.imshow(G, cmap='binary')
ax = plt.subplot(2, 3, 6)
ax.set_title('B')
ax.imshow(B, cmap='binary')
plt.show()
