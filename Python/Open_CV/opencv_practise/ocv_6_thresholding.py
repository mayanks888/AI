import cv2
import matplotlib.pyplot as plt

# img = cv2.imread("gradient.jpg", 1)
img = cv2.imread("img_2.png", 1)
# a very simple concenpt if val of pixel is greater that threshold make is 255 or else make 0
_, img2 = cv2.threshold(img, 25, 255, type=cv2.THRESH_BINARY)
_, img3 = cv2.threshold(img, 127, 255, type=cv2.THRESH_BINARY_INV)
_, img4 = cv2.threshold(img, 127, 255,
                        type=cv2.THRESH_TRUNC)  # if val of pixel is > threshold make 127 and for less < threshold make keepit same.
_, img5 = cv2.threshold(img, 127, 255,
                        type=cv2.THRESH_TOZERO)  # if val of pixel is > threshold keep it same and for less < threshold make 0
# img6 = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
title = ["image", "thresbinary", "inverserBinary", "trunc", "to_zero","otsu"]
imgs = [img, img2, img3, img4, img5]

for i in range(5):
    plt.subplot(2, 3, i + 1), plt.imshow(imgs[i])
    plt.title(title[i])

plt.show()

cv2.waitKey(0)
cv2.destroyAllWindows()

# cv2.imshow("image",img)
# cv2.imshow("img2",img2)
# cv2.imshow("img3",img3)
# cv2.imshow("truc",img4)
# cv2.imshow("tozero",img5)
# cv2.waitKey()
# cv2.destroyAllWindows()

#
