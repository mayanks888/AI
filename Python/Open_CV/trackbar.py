import cv2

import numpy as np


# def nothing(x):
#     pass
# cap = cv2.VideoCapture(0)
# cv2.namedWindow("frame")
# cv2.createTrackbar("test", "frame", 50, 500, nothing)
# cv2.createTrackbar("color/gray", "frame", 0, 1, nothing)
# while True:
#     _, frame = cap.read()
#     test = cv2.getTrackbarPos("test", "frame")
#     font = cv2.FONT_HERSHEY_COMPLEX
#     cv2.putText(frame, str(test), (50, 150), font, 4, (0, 0, 255))
#     s = cv2.getTrackbarPos("color/gray", "frame")
#     if s == 0:
#         pass
#     else:
#         frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#     cv2.imshow("frame", frame)
#     key = cv2.waitKey(1)
#     if key == 27:
#         break
# cap.release()
# cv2.destroyAllWindows()

def nothing(x):
    pass


# Create a black image, a window
img = np.zeros((300, 512, 3), np.uint8)
cv2.namedWindow('image')

# create trackbars for color change
cv2.createTrackbar('R', 'image', 0, 255, nothing)
cv2.createTrackbar('G', 'image', 0, 255, nothing)
cv2.createTrackbar('B', 'image', 0, 255, nothing)

# create switch for ON/OFF functionality
switch = '0 : OFF \n1 : ON'
cv2.createTrackbar(switch, 'image', 0, 1, nothing)

while (1):
    cv2.imshow('image', img)
    k = cv2.waitKey(1) & 0xFF
    if k == 27:
        break

    # get current positions of four trackbars
    r = cv2.getTrackbarPos('R', 'image')
    g = cv2.getTrackbarPos('G', 'image')
    b = cv2.getTrackbarPos('B', 'image')
    s = cv2.getTrackbarPos(switch, 'image')

    if s == 0:
        img[:] = 0
    else:
        img[:] = [b, g, r]

cv2.destroyAllWindows()
