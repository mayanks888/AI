import cv2
import numpy as np
img=cv2.imread("car.jpg",1)
img=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
# draw line on opencv
# cv2.line(img,(2,2),(50,50),(255,0,255
#creat custom images
# img=np.zeros([512,512])
# img = np.random.randint(0,255, size=(512,512,3),dtype=np.uint8)
img = np.random.randint(0,255, size=(512,512),dtype=np.uint8)
cv2.rectangle(img,(20,40),(50,10),(255,5,0),thickness=-1)#(pt1=(xmin,ymax),pt2=(smin,ymax)

cv2.circle(img,(80,80),radius=50,color=(255,255,0),thickness=2)
cv2.imshow("image",img)
cv2.waitKey()
cv2.destroyAllWindows()

#