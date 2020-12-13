import cv2
import numpy as np

cap = cv2.VideoCapture('/home/mayank_s/Videos/new_bag_snowball_Demo.avi')

fourcc = cv2.VideoWriter_fourcc(*'XVID')
# out = cv2.VideoWriter('output1.avi', fourcc, 5, (1280, 720))
out = cv2.VideoWriter('output1.avi', fourcc, 80, (1280, 720))
count=0
while True:
    ret, frame = cap.read()
    if ret == True:
        b = cv2.resize(frame, (1280, 720), fx=0, fy=0, interpolation=cv2.INTER_CUBIC)
        out.write(b)
        count+=1
        # if count>2500:
        #     break
    else:
        break

cap.release()
out.release()
cv2.destroyAllWindows()