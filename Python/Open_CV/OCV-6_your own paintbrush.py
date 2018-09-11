import cv2
import numpy as np

one=np.ones([500,500,3],dtype='uint8')#create all 2 channel image of 1
white_image=one*255#converting complete image into 255(white)
# cv2.imshow("canvas",white_image)


circle_color=(0,200,0)#color is green
line_width=4
radius=200
point=(234,341)#base center for the  printed circle
pressed=False

#Function for mouse click
def click(event,x,y,flags,param):
    global canvas,pressed
    if event==cv2.EVENT_LBUTTONDOWN:
        cv2.circle(white_image,center=(x,y),radius=3,color=circle_color,thickness=line_width)
        pressed=True
        print('mouse left button')
    elif event==cv2.EVENT_LBUTTONUP:
        pressed=False
        print('right button')
    elif (event==cv2.EVENT_MOUSEMOVE and pressed==True):
        cv2.circle(white_image, center=(x, y), radius=3, color=circle_color, thickness=line_width)
    #     print('mosuse wheel move')


# cv2.namedWindow('canvas')  # this is to create a windows  with size like autosize or fixed size
# cv2.setMouseCallback('canvas', click)

while(True):

    cv2.namedWindow('canvas')  # this is to create a windows  with size like autosize or fixed size
    cv2.setMouseCallback('canvas', click)

    cv2.imshow("canvas", white_image)
    ch=cv2.waitKey(1)#refresh after 1 milisecong
    if ch & 0XFF==ord('q'):
        break
    elif ch & 0XFF==ord('b'):
        circle_color=(255,0,0)#this is to change the color (if key 'b' is pressed then change circle color to blue)
    elif ch & 0XFF==ord('r'):
        circle_color=(0,0,255)#this is to change the color (if key 'r' is pressed then change circle color to red)

cv2.waitKey(0)
cv2.destroyAllWindows()
