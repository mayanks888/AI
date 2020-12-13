import os
import time

import cv2

# from PIL import Image
ts = time.time()
ts = time.time()

# cut_frame="/home/mayank_sati/Desktop/crop_image/"
# cut_frame=cut_frame+"image_"+str(st)+".jpg"
# cv2.imwrite(cut_frame, frame)

saved_path = "/home/mayank_sati/Desktop/sorting_light/complete_image_with_diff_name/black/"
input_folder = root = "/home/mayank_sati/Desktop/sorting_light/complte_data/black"
for root, _, filenames in os.walk(input_folder):
    if (len(filenames) == 0):
        print("Input folder is empty")
    #     return 1
    # time_start = time.time()
    for filename in filenames:
        image_path = os.path.join(root, filename)
        image_scale = cv2.imread(image_path, 1)
        ts = time.time()
        st = datetime.datetime.fromtimestamp(ts).strftime('%d_%m_%Y_%H_%M_%S_%f')
        cut_frame = saved_path + "image_" + str(st) + ".jpg"
        cv2.imwrite(cut_frame, image_scale)
