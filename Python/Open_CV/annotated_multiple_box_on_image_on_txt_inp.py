
import os
import cv2
import numpy as np

Pred_path = '/home/mayank_sati/Desktop/prediction_apollo'
Ground_path = '/home/mayank_sati/Desktop/Groundthruth_front_facing'
image_path='/media/mayank_sati/DATA/datasets/one_shot_datasets/Farmington/images/new_farm_imageS_with_correct_box_n_qual/farm_eval'
output_folder = '/home/mayank_sati/Desktop/apollo_all'

if not os.path.exists(output_folder):
    print("Output folder not present. Creating New folder...")
    os.makedirs(output_folder)
    # loop = 1
for root, _, filenames in os.walk(Ground_path):
    if (len(filenames) == 0):
        print("Input folder is empty")
        # return 1
    # time_start = time.time()
    for index,filename in enumerate(filenames):
        print(index, filename)
        # print("file : {fn}".format(fn=filename), '\n')
        file_path = (os.path.join(root, filename))

        ###############################3333
        labelname = (os.path.join(root, filename))
        Pred_label_filename = (os.path.join(Pred_path, filename))
        ######################################################33
        # txt_file_name=filename.split('.')[0]
        txt_file_name = filename.split('.txt')[0]
        # txt_file_name=filename.split('/')[-1]
        txt_file_name = txt_file_name + ".jpg"
        jpg_file_path = (os.path.join(image_path, txt_file_name))
        image_scale = cv2.imread(jpg_file_path, 1)
        bbox_cnt = 0
        if os.path.exists(labelname):
            # with open(output_labelfilename, 'w') as k:
                with open(labelname) as f:
                    for (i, line) in enumerate(f):
                        yolo_data = line.strip().split()
                        top = (int(yolo_data[0+1]), int(yolo_data[3+1]))
                        bottom = (int(yolo_data[2+1]), int(yolo_data[1+1]))
                        cv2.rectangle(image_scale, pt1=top, pt2=bottom, color=(0, 255, 0), thickness=2)
                        # cv2.putText(image_scale, str(round(y[read_index][4],2)),(int((y[read_index][0] + y[read_index][2]) / 2), int(y[read_index][1] - 10)), cv2.FONT_HERSHEY_SIMPLEX, .5, (0, 255, 0), lineType=cv2.LINE_AA)


        if os.path.exists(Pred_label_filename):
            # with open(output_labelfilename, 'w') as k:
                with open(Pred_label_filename) as f:
                    for (i, line) in enumerate(f):
                        yolo_data = line.strip().split()
                        top = (int(yolo_data[0+2]), int(yolo_data[3+2]))
                        bottom = (int(yolo_data[2+2]), int(yolo_data[1+2]))
                        cv2.rectangle(image_scale, pt1=top, pt2=bottom, color=(0, 0, 255), thickness=2)
                        # cv2.putText(image_scale, str(round(float(yolo_data[1]),2)),(int((yolo_data[0+2] + int(yolo_data[2+2])) / 2), int(yolo_data[1+2] - 10)), cv2.FONT_HERSHEY_SIMPLEX, .5, (0, 255, 0), lineType=cv2.LINE_AA)
                        cv2.putText(image_scale, str(round(float(yolo_data[1]),2)),(bottom), cv2.FONT_HERSHEY_SIMPLEX, .5,(0, 0, 255), lineType=cv2.LINE_AA)

        show=False
        if show:
            cv2.imshow('streched_image', image_scale)
            ch = cv2.waitKey(1000)
            if ch & 0XFF == ord('q'):
                cv2.destroyAllWindows()
            cv2.destroyAllWindows()


        output_file_path = (os.path.join(output_folder, txt_file_name))
        cv2.imwrite(output_file_path, image_scale)
#     # cv2.destroyAllWindows()
