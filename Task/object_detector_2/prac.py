import cv2

image_scale = cv2.imread("temp.jpg", 1)

cv2.putText(image_scale, "hello", (100, 200), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), lineType=cv2.LINE_AA)
