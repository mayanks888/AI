import os
import cv2
import numpy as np

output_folder='/media/mayank_sati/DATA/datasets/one_shot_datasets/Farmington/images/valid_farm'
copy_path='/media/mayank_sati/DATA/datasets/one_shot_datasets/Farmington/images/2019-09-27-14-39-41_test'
input_folder='/home/mayank_sati/Desktop/farm_test'
for root, _, filenames in os.walk(input_folder):
    if (len(filenames) == 0):
        print("Input folder is empty")
        # return 1
    # time_start = time.time()
    for filename in filenames:
            print("file : {fn}".format(fn=filename), '\n')
            file_path = (os.path.join(copy_path, filename))
            imageToPredict = cv2.imread(file_path, 3)
            # print(imageToPredict.shape)
            # y_ = imageToPredict.shape[0]
            # x_ = imageToPredict.shape[1]
            # # targetsize
            # targetSize_x = int(x_ / 2)
            # targetSize_y = int(y_ / 2)
            # img = cv2.resize(imageToPredict, (targetSize_x, targetSize_y));

            output_path = (os.path.join(output_folder, filename))
            cv2.imwrite(output_path, imageToPredict)
