import os

import cv2
import natsort

# image_folder = '/home/mayank_sati/Desktop/pp/save_img'
# video_name = 'video.avi'


# dir_path = '/home/mayank_sati/Desktop/pp/save_img'
# dir_path = '//home/mayank_sati/Desktop/pp/save_lid_annotation_without_score'
# dir_path = '/home/mayank_sati/Desktop/pp/save_lid_annotation_without_score_sweep'
dir_path = '/home/mayank_sati/Documents/new_state/1030'
# dir_path = '/home/mayanksati/PycharmProjects/Data_Science_new/Task/object_detector_2/training/traffic_mulitple_video'
# ext = 'jpg'
# video_name = '/home/mayank_sati/Desktop/pp/video_bird_eye.mp4'
video_name = '/media/mayank_sati/DATA/datasets/one_shot_datasets/Farmington/images/new_farm_imageS_with_correct_box_n_qual/template/temp1.avi'
# video_name = '/home/mayank_sati/Documents/new_state/temp_n_detect_2.avi'

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
fourcc = cv2.VideoWriter_fourcc(*'XVID')
video = cv2.VideoWriter(video_name, fourcc, 15.0, (width, height))
##############################################3

for image in images:
    video.write(cv2.imread(os.path.join(dir_path, image)))

cv2.destroyAllWindows()
video.release()
