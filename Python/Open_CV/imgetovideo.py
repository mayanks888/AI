# For more info: http://docs.opencv.org/3.0-beta/doc/py_tutorials/py_gui/py_video_display/py_video_display.html
import cv2 as cv2
import os
import os

import cv2 as cv2

FILE_OUTPUT = 'output.avi'

# Checks and deletes the output file
# You cant have a existing file or it will through an error
if os.path.isfile(FILE_OUTPUT):
    os.remove(FILE_OUTPUT)

# Playing video from file:
cap = cv2.VideoCapture('/home/mayank_sati/codebase/git/AI/Machine_learning/Python/Open_CV/image_video.avi')
# Capturing video from webcam:
# cap = cv2.VideoCapture(0)

currentFrame = 0

width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)  # float
height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)  # float

# width = cap.set(cv2.CAP_PROP_FRAME_WIDTH, 160);
# height  = cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 120);


# Get current width of frame
# width = cap.get(cv2.CV_CAP_PROP_FRAME_WIDTH)   # float
# Get current height of frame
# height = cap.get(cv2.CV_CAP_PROP_FRAME_HEIGHT) # float


# Define the codec and create VideoWriter object
# fourcc = cv2.CV_FOURCC(*'X264')

fourcc = cv2.VideoWriter_fourcc(*'DIVX')
# fourcc = cv2.VideoWriter_fourcc(*'X264')
# fourcc = cv2.VideoWriter_fourcc('x', '2','6','4')
# x', '2','6','4'
out = cv2.VideoWriter(FILE_OUTPUT, fourcc, 25.0, (int(width), int(height)))

# while(True):
while (cap.isOpened()):
    # Capture frame-by-frame
    ret, frame = cap.read()

    if ret == True:
        # Handles the mirroring of the current frame
        # frame = cv2.flip(frame,1)

        # Saves for video
        out.write(frame)

        # Display the resulting frame
        cv2.imshow('frame', frame)
    else:
        break
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    # To stop duplicate images
    currentFrame += 1

# When everything done, release the capture
cap.release()
out.release()
