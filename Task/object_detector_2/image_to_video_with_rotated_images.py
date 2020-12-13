import os

import cv2
import natsort

# image_folder = '/home/mayank_sati/Desktop/pp/save_img'
# video_name = 'video.avi'


dir_path = '/home/mayank_sati/Desktop/pp/save_img_sweep'
# dir_path = '/home/mayank_sati/Desktop/pp/save_lid_annotation'
# dir_path = '/home/mayanksati/PycharmProjects/Data_Science_new/Task/object_detector_2/training/traffic_mulitple_video'
# ext = 'jpg'
video_name = '/home/mayank_sati/Desktop/pp/video_bird_eye_sweep.mp4'
# video_name = '/home/mayank_sati/Desktop/pp/video_pcl.mp4'

images = []

###################################33333
for root, _, filenames in os.walk(dir_path):
    if (len(filenames) == 0):
        print("Input folder is empty")
        # return 1
    filenames = natsort.natsorted(filenames, reverse=False)
    # time_start = time.time()
    for filename in filenames:
        ##########################################
        # for f in os.listdir(dir_path):
        # if f.endswith(ext):
        images.append(filename)

# images = [img for img in os.listdir(image_folder) if img.endswith(".jpg")]
frame = cv2.imread(os.path.join(dir_path, images[0]))
height, width, layers = frame.shape

# video = cv2.VideoWriter(video_name, 0, 1, (width,height))

######################################3333
# fourcc = cv2.VideoWriter_fourcc(*'XVID')
fourcc = cv2.VideoWriter_fourcc(*'XVID')
video = cv2.VideoWriter(video_name, fourcc, 15.0, (width, height))
# video = cv2.VideoWriter(video_name,0, 10.0, (width,height))
##############################################3

for image in images:
    img = cv2.imread(os.path.join(dir_path, image))
    # get image height, width
    (h, w) = img.shape[:2]
    # calculate the center of the image
    center = (w / 2, h / 2)

    angle90 = 90
    angle180 = 180
    angle270 = 270

    scale = 1.0

    # Perform the counter clockwise rotation holding at the center
    # 90 degrees
    # M = cv2.getRotationMatrix2D(center, angle90, scale)
    # rotated90 = cv2.warpAffine(img, M, (h, w))

    # 180 degrees
    M = cv2.getRotationMatrix2D(center, angle180, scale)
    rotated180 = cv2.warpAffine(img, M, (w, h))
    video.write(rotated180)

cv2.destroyAllWindows()
video.release()
