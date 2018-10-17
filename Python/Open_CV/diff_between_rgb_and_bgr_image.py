import cv2
# import
import matplotlib.pyplot as plt
import os

'''img=cv2.imread('2008_000008.jpg')
print(1)
# img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)


# b,r,g = cv2.split(img) :Confusion in this format
r,g,b = cv2.split(img)

img1 = cv2.merge((b,g,r))
img1[:, :, [1,2]] = 0


img2 = cv2.merge((r,b,g))
img2[:, :, [0, 2]] = 0

img3= cv2.merge((g,b,r))
img3[:, :, [0,1]] = 0

# cv2.imshow('Image_bgr_', img1)
# cv2.imshow('Image2_rbg', img2)
# cv2.imshow('Image_rbg_orignial', img)

cv2.imshow('red', img1)
cv2.imshow('green', img2)
cv2.imshow('blue', img)
cv2.waitKey(0)
cv2.destroyAllWindows()'''

# *************************************************************
# 2nd method
img=cv2.imread('2008_000008.jpg')
print(1)
# img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)


# b,r,g = cv2.split(img) :Confusion in this format
r,g,b = cv2.split(img)

img1 = cv2.merge((r,g,b))

img2=img3=img1

img1[:, :, [1,2]] = 0



img2[:, :, [0, 2]] = 0


img3[:, :, [0,1]] = 0


cv2.imshow('red', img1)
cv2.imshow('green', img2)
cv2.imshow('blue', img3)
cv2.waitKey(0)
cv2.destroyAllWindows()

# *********************************************************

# Corrected version
# Show RGB channel separately in color

fig, axes = plt.subplots(1, 3)
image=cv2.imread('2008_000008.jpg')
# image[:, :, 0] is R channel, replace the rest by 0.
imageR = image.copy()
imageR[:, :, 1:3] = 0
axes[0].set_title('R channel')
axes[0].imshow(imageR)

# image[:, :, 1] is G channel, replace the rest by 0.
imageG = image.copy()
imageG[:, :, [0, 2]] = 0
axes[1].set_title('G channel')
axes[1].imshow(imageG)

# image[:, :, 2] is B channel, replace the rest by 0.
imageB = image.copy()
imageB[:, :, 0:2] = 0
axes[2].set_title('B channel')
axes[2].imshow(imageB)
# plt.savefig(os.path.join(basedir, 'RGB_color.jpg'))
plt.savefig( 'RGB_color.jpg')



# *********************************************************


# *********************************************************
# modified version

 # Show RGB channel separately in color

fig, axes = plt.subplots(1, 3)
# image=cv2.imread('2008_000008.jpg')
image=img3
# image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
# image[:, :, 0] is R channel, replace the rest by 0.
imageR = image.copy()
imageR[:, :, [1,2]] = 0
axes[0].set_title('R channel')
axes[0].imshow(imageR)

# image[:, :, 1] is G channel, replace the rest by 0.
imageG = image.copy()
imageG[:, :, [0, 2]] = 0
axes[1].set_title('G channel')
axes[1].imshow(imageG)

# image[:, :, 2] is B channel, replace the rest by 0.
imageB = image.copy()
imageB[:, :, [0,1]] = 0
axes[2].set_title('B channel')
axes[2].imshow(imageB)
# plt.savefig(os.path.join(basedir, 'RGB_color.jpg'))
plt.savefig( 'RGB_color_2.jpg')
