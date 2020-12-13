###########3
import cv2
import numpy as np

all_img = []
cap = cv2.VideoCapture(
    '/media/mayank_sati/DATA/datasets/one_shot_datasets/Farmington/images/new_farm_imageS_with_correct_box_n_qual/template/temp1.avi',
    0)
cap1 = cv2.VideoCapture(
    '/media/mayank_sati/DATA/datasets/one_shot_datasets/Farmington/images/new_farm_imageS_with_correct_box_n_qual/template/temp_2.avi',
    0)
video_name = '/media/mayank_sati/DATA/datasets/one_shot_datasets/Farmington/images/new_farm_imageS_with_correct_box_n_qual/template/temp_combined.avi'
while (cap.isOpened()):

    ret, frame = cap.read()
    ret1, frame1 = cap1.read()
    if ret1 == True:

        both = np.concatenate((frame, frame1), axis=1)
        print(both.shape)
        all_img.append(both)

        # cv2.imshow('Frame', both)
        # out.write(frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    else:
        break

######################################3333
frame = all_img[0]
height, width, layers = frame.shape
fourcc = cv2.VideoWriter_fourcc(*'XVID')
video = cv2.VideoWriter(video_name, fourcc, 15.0, (width, height))
##############################################3

for image in all_img:
    # out.write(cv2.imread(os.path.join(dir_path, image)))
    video.write(image)

cv2.destroyAllWindows()

cap.release()
video.release()

cv2.waitKey(0)
cv2.destroyAllWindows()
