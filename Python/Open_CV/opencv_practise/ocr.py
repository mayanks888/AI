# import cv2
# import pytesseract
#
# img = cv2.imread('img_3.png')
# h, w, c = img.shape
# boxes = pytesseract.image_to_boxes(img)
# for b in boxes.splitlines():
#     b = b.split(' ')
#     img = cv2.rectangle(img, (int(b[1]), h - int(b[2])), (int(b[3]), h - int(b[4])), (0, 255, 0), 2)
#
# cv2.imshow('img', img)
# cv2.waitKey(0)


import pytesseract
from PIL import Image
print(pytesseract.image_to_string(Image.open('img_3.png'), lang="ara"))