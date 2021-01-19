import cv2
import numpy as np


def mycallbak(x):
   pass


cv2.namedWindow("mybar")
cv2.createTrackbar("LH", "mybar", 0, 255, mycallbak)
cv2.createTrackbar("LS", "mybar", 0, 255, mycallbak)
cv2.createTrackbar("LV", "mybar", 0, 255, mycallbak)
cv2.createTrackbar("UH", "mybar", 255, 255, mycallbak)
cv2.createTrackbar("US", "mybar", 255, 255, mycallbak)
cv2.createTrackbar("UV", "mybar", 255, 255, mycallbak)

while (True):
    img = cv2.imread("color_balls.png", 1)
    img2 = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    l_h = cv2.getTrackbarPos("LH", "mybar")
    l_s = cv2.getTrackbarPos("LS", "mybar")
    l_v = cv2.getTrackbarPos("LV", "mybar")
    u_h = cv2.getTrackbarPos("UH", "mybar")
    u_s = cv2.getTrackbarPos("US", "mybar")
    u_v = cv2.getTrackbarPos("UV", "mybar")

    l_b = np.array([l_h, l_s, l_v])
    u_b = np.array([u_h, u_s, u_v])

    mask = cv2.inRange(img2, l_b, u_b)#this is basically filtering the values in in the whole hsv range ie it wil
    # make all the non range value to 0
    res = cv2.bitwise_and(img, img, mask=mask)
    # res = cv2.bitwise_and(img, mask)

    cv2.imshow("mask", mask)
    cv2.imshow("image", img)
    cv2.imshow("res", res)
    key = cv2.waitKey(1)
    if key == 27:
        break
cv2.destroyAllWindows()
