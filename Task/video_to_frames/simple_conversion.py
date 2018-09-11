import cv2
import numpy as np
path='/home/mayank-s/PycharmProjects/Datasets/Video_to_frame/input/sjj.mp4'
vidcap = cv2.VideoCapture(path)
success,image = vidcap.read()
count = 0
success = True
while success:
  success,image = vidcap.read()
  frame_expanded = np.expand_dims(image, axis=0)
  cv2.imwrite("frame%d.jpg" % count, image)     # save frame as JPEG file
  if cv2.waitKey(10) == 27:                     # exit if Escape is hit
      break
  count += 1