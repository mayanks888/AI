import cv2
import numpy as np
img=cv2.imread("img_2.png",0)
#a very simple concenpt if val of pixel is greater that threshold make is 255 or else make 0
_,img2=cv2.threshold(img,127,255, type=cv2.THRESH_BINARY)
# _,img3=cv2.threshold(img,127,255, type=cv2.THRESH_BINARY_INV)
# _,img4=cv2.threshold(img,127,255, type=cv2.THRESH_TRUNC)#if val of pixel is > threshold make 127 and for less < threshold make keepit same.
# _,img5=cv2.threshold(img,127,255, type=cv2.THRESH_TOZERO)#if val of pixel is > threshold keep it same and for less < threshold make 0
img_6=cv2.adaptiveThreshold(img,255,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY,11,2)
img_7=cv2.adaptiveThreshold(img,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,11,2)
# th3 = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2);


cv2.imshow("image",img)
cv2.imshow("img2",img2)
# cv2.imshow("img3",img3)
# cv2.imshow("truc",img4)
# cv2.imshow("tozero",img5)
cv2.imshow("adapt",img_6)
cv2.imshow("adapt_g",img_7)
cv2.waitKey()
cv2.destroyAllWindows()

#