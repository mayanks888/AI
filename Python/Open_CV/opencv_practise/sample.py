import cv2
import matplotlib.pyplot as plt

img = cv2.imread("gradient.jpg", 1)


title = ["image", "thresbinary", "inverserBinary", "trunc", "to_zero"]
imgs = [img, img2, img3, img4, img5]

for i in range(5):
    plt.subplot(2, 3, i + 1), plt.imshow(imgs[i])
    plt.title(title[i])

plt.show()


#
