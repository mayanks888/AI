import cv2

image_scale=cv2.imread("car.jpg", 1)
print(image_scale.shape)
# cv2.rectangle(image_scale, pt1=(0,20),pt2=(20,0),color=(0,255,0),thickness=3)
# cv2.rectangle(image_scale, pt1=top, pt2=bottom, color=(0, 255, 0), thickness=2)
image_scale = cv2.Canny(image_scale, 50, 240)
# cv2.rectangle()
cv2.imshow("img_2", image_scale)
cv2.waitKey(1000)
cv2.destroyAllWindows()
# cv2.destroyWindow()