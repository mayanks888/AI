import cv2
import numpy as np

#parsing webcam
cap=cv2.VideoCapture(0)
while(True):
    ret,frame=cap.read()
    frame=cv2.resize(frame,dsize=(0,0),fx=1,fy=1)#resize frame
    cv2.imshow("frame",frame)
    #press "q" to come out of the loop
    ch=cv2.waitKey(1)#refresh after 1 milisecong
    if ch & 0XFF==ord('q'):
        break
cap.release()
cv2.destroyAllWindows()


l].objs.append(objif)