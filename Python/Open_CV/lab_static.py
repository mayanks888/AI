import sys
import time

import cv2
import numpy as np

font = cv2.FONT_HERSHEY_SIMPLEX
bottomLeftCornerOfText = (10, 500)
fontScale = 2
fontColor = (0, 0, 255)
lineType = 2

start_time = time.time()
img = cv2.imread(sys.argv[1])
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2Lab)
# hist = cv2.calcHist( [hsv], [1], None, [25], [0, 256] )
# hist1 = cv2.calcHist([hsv], [2], None, [25], [0, 256] ) 


lower_green = np.array([0, 0, 0])  # scaling upto original pixel value
upper_green = np.array([255, 112, 150])  # max cr at peak of hist, max cb at 150 to filter out yellow
mask2 = cv2.inRange(hsv, lower_green, upper_green)
# cpy=mask2.copy()
# print(int(sg[i,0]))
lower_red = np.array([0, 150, 112])  # scaling upto original pixel value :int(peak2s*10.24)
upper_red = np.array([255, 255, 255])
maskr = cv2.inRange(hsv, lower_red, upper_red)

lower_yel = np.array([0, 112, 150])  # scaling upto original pixel value:
upper_yel = np.array([255, 150, 255])
masky = cv2.inRange(hsv, lower_yel, upper_yel)

cpy = np.zeros(img.shape, np.uint8)
contoursg, hierarchy = cv2.findContours(mask2, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
areasg = [cv2.contourArea(c) for c in contoursg]
i = 0
area_img = np.shape(img)[0] * np.shape(img)[1]
for c in contoursg:
    # print("area:"+str(areasg[i]))
    if ((areasg[i] / area_img) > 0.00006 and (areasg[i] / area_img) < 0.008):
        x, y, w, h = cv2.boundingRect(c)
        cv2.drawContours(cpy, contoursg, i, (0, 255, 0), -1)
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)

    i = i + 1
contoursr, hierarchy = cv2.findContours(maskr, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
areasr = [cv2.contourArea(c) for c in contoursr]
i = 0
for c in contoursr:
    # print("area:"+str(areasr[i]))
    if ((areasr[i] / area_img) > 0.00006 and (areasr[i] / area_img) < 0.008):
        x, y, w, h = cv2.boundingRect(c)
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 2)
    i = i + 1

contoursy, hierarchy = cv2.findContours(masky, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
areasy = [cv2.contourArea(c) for c in contoursy]
i = 0
for c in contoursy:
    # print("area:"+str(areasr[i]))
    if ((areasy[i] / area_img) > 0.00006 and (areasy[i] / area_img) < 0.008):
        x, y, w, h = cv2.boundingRect(c)
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 255), 2)
    i = i + 1

cpy1 = np.zeros(img.shape, np.uint8)
cpy2 = np.zeros(img.shape, np.uint8)
# print(contours)
# np.array(contours)
cv2.drawContours(cpy, contoursg, -1, (0, 255, 0), -1)
cv2.drawContours(cpy, contoursr, -1, (0, 0, 255), -1)
cv2.drawContours(cpy, contoursy, -1, (0, 255, 255), -1)
cv2.imshow("close", cpy)
# cv2.imshow("near",cpy1)
# cv2.imshow("far",cpy2)
cv2.imshow("orig", img)
print("--- %s seconds ---" % (time.time() - start_time))
cv2.waitKey()

# s_size=np.shape(s)
# print("size of s: "+ str(s_size))


val_prev = 0
# print(s)
# for i in range(0,25):

#     if abs(s[i][0] - val_prev) < 4:
#         print(i)
#         continue

#     elif s[i][0] < 100 :
#         msg="peak no."+str(i)+" green val: "+ str(s[i][0])
#         lower_red = np.array([0,0,0])
#         upper_red = np.array([255,int(s[i][0]*10.24),255])
#     elif s[i][0] > 130 :
#         msg="peak no."+str(i)+" red val: "+ str(s[i][0])
#         lower_red = np.array([0,int(s[i][0]*10.24),0])
#         upper_red = np.array([255,255,255])
#     else:
#         msg="peak no."+str(i)+" other noise val: "+ str(s[i][0])
#         lower_red = np.array([0,100,0])
#         upper_red = np.array([255,int(s[i][0]*10.24),255])
#     mask2 = cv2.inRange(hsv, lower_red, upper_red)    

#     print(msg)
#     #cv2.putText(img,msg,bottomLeftCornerOfText,font,fontScale,fontColor,lineType)
#     cv2.imshow("result",mask2)
#     cv2.imshow("orig",img)
#     val_prev=s[i][0]
#     cv2.waitKey()
