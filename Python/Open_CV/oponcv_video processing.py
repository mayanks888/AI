import cv2
import numpy as np

#parsing webcam
cap=cv2.VideoCapture(0)

circle_color=(0,200,0)#color is green
line_width=4
radius=200
point=(234,341)#base center for the  printed circle
#Function for mouse click
def click(event,x,y,flags,param):
    global point,pressed
    if event==cv2.EVENT_LBUTTONDOWN:
        print('pressed',x,y)
        point=(x,y)
        
cv2.namedWindow('Frame')#this is to create a windows  with size like autosize or fixed size
cv2.setMouseCallback('Frame',click)

while(True):
    ret,frame=cap.read()
    frame=cv2.resize(frame,dsize=(0,0),fx=1,fy=1)#resize frame
    cv2.circle(img=frame,center=point,radius=radius,color=circle_color,thickness=line_width)
    cv2.imshow("Frame",frame)#remember name is ververy important to work in the same windows
    #press "q" to come out of the loop
    ch=cv2.waitKey(1)#refresh after 1 milisecong
    if ch & 0XFF==ord('q'):
        break
cv2.waitKey(0)
cap.release()
cv2.destroyAllWindows()