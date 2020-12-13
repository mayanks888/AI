# import numpy as np
# import cv2
# import os
#
# # this two lines are for loading the videos.
# # in this case the video are named as: cut1.mp4, cut2.mp4, ..., cut15.mp4
# # videofiles = [n for n in os.listdir('.') if n[0]=='c' and n[-4:]=='.mp4']
# videofiles=['aa.avi','a.avi']
# # videofiles = sorted(videofiles, key=lambda item: int( item.partition('.')[0][3:]))
#
# video_index = 0
# cap = cv2.VideoCapture(videofiles[0])
#
# # video resolution: 1624x1234 px
# # out = cv2.VideoWriter("video.avi", cv2.cv.CV_FOURCC('F','M','P', '4'), 15, (1624, 1234), 1)
# #############3333
# height=1555
# width=2055
# fourcc = cv2.VideoWriter_fourcc(*'XVID')
# out = cv2.VideoWriter("video.avi", fourcc, 15.0, (width, height))
# #######################
# while(cap.isOpened()):
#     ret, frame = cap.read()
#     if frame is None:
#         print ("end of video " + str(video_index) + " .. next one now")
#         video_index += 1
#         if video_index >= len(videofiles):
#             break
#         cap = cv2.VideoCapture(videofiles[ video_index ])
#         ret, frame = cap.read()
#     cv2.imshow('frame',frame)
#     out.write(frame)
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break
#
# cap.release()
# out.release()
# cv2.destroyAllWindows()
#
# print ("end.")
#

###########3
import cv2
import numpy as np

cap = cv2.VideoCapture('a.avi', 0)
cap1 = cv2.VideoCapture('aa.avi', 0)
height = 1032
width = 772
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter("video.avi", fourcc, 8.0, (width, height))
while (cap.isOpened()):

    ret, frame = cap.read()
    ret1, frame1 = cap1.read()
    if ret == True:

        both = np.concatenate((frame, frame1), axis=1)

        cv2.imshow('Frame', both)
        out.write(frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break


    else:
        break

cap.release()
out.release()

cv2.waitKey(0)
cv2.destroyAllWindows()
