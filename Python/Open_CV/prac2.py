import cv2


image = cv2.imread("ab.jpg", 1)
# image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# cv2.imshow('bounding_box image', image)
# cv2.waitKey(1000)
# cv2.destroyAllWindows()
# _____________________________________________________________
# cv2.namedWindow('color detector : Recogniser',cv2.WINDOW_NORMAL)
# cv2.namedWindow('color detector : Recogniser',cv2.WINDOW_AUTOSIZE)
cv2.namedWindow('color detector : Recogniser',cv2.WINDOW_FREERATIO)
cv2.resizeWindow('color detector : Recogniser', 600,600)
cv2.moveWindow('color detector : Recogniser', 1,1)
# cv2.imshow('color detector : Recogniser', image)
# cv2.waitKey(1000)
cv2.destroyAllWindows()
# ______________________

import numpy as np
create_color = np.random.random_integers(low=1, high=244, size=[224, 224, 3]).astype('float32')
1