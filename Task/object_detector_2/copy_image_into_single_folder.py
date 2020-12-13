import os
from random import randint

import cv2

loop = 1
file_path = '/home/mayanksati/PycharmProjects/Data_Science_new/Task/object_detector_2/training/save_cropped_images_300'
path = "/home/mayanksati/Desktop/crop_r_313/"
for root, dirs, files in os.walk(file_path):
    for filename in files:
        print(filename)

        # for filename in filenames:
        # loop = loop + 1
        # output_path = path + str(loop) + ".jpg"

        # try:
        print("Creating object detection for file : {fn}".format(fn=filename), '\n')

        file_path = (os.path.join(root, filename))
        image = cv2.imread(file_path, 1)
        # image=cv2.resize(image, (800, 800), fx=0.5, fy=0.5)
        widht = (randint(300, 600))
        lenght = (randint(300, 600))
        image = cv2.resize(image, dsize=(widht, lenght), interpolation=cv2.INTER_NEAREST)
        output_path = path + filename
        print(output_path)
        cv2.imwrite(output_path, image)
