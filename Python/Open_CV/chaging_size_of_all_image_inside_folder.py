import os

import cv2

input_folder = '/home/mayank_s/Desktop/template/farm_2/farm_2_images2'
output_folder = '/home/mayank_s/Desktop/template/farm_2/farm_2_images2_scaled'

for root, _, filenames in os.walk(input_folder):
    if (len(filenames) == 0):
        print("Input folder is empty")
        # return 1
    # time_start = time.time()
    for filename in filenames:
            print("file : {fn}".format(fn=filename), '\n')
            file_path = (os.path.join(root, filename))
            imageToPredict = cv2.imread(file_path, 3)
            # print(imageToPredict.shape)
            y_ = imageToPredict.shape[0]
            x_ = imageToPredict.shape[1]
            # targetsize
            targetSize_x = int(x_ / 1.5)
            targetSize_y = int(y_ / 1.5)
            img = cv2.resize(imageToPredict, (targetSize_x, targetSize_y));

            output_path = (os.path.join(output_folder, filename))
            cv2.imwrite(output_path, img, [cv2.IMWRITE_JPEG_QUALITY, 100])
