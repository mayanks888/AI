import os

import pandas as pd
import shutil

csv_path = "/home/mayank_s/codebase/others/yolo/yolov4_custom/yolov4_bdd_prediction_val.csv"
# csv_path="/home/mayanksati/PycharmProjects/Data_Science_new/Deep learning/Load Datasets/BBD_daytime_train.csv"
base_image_path = '/home/mayank_s/Desktop/custom_farm_front_tl/farm_parking_1/yolo_output'
datasets = pd.read_csv(csv_path)
data = datasets.iloc[:].values
source='/home/mayank_s/datasets/farminton/demo_route_complete_images/parking_lot_1'
desitnation='/home/mayank_s/Desktop/custom_farm_front_tl/farm_parking_1/farm_parking_1_final'
id_list = [1001, 1002, 1003]
bblabel = []
new_file_path = ''
for values in data:
    # filename, image_width, image_height, object_name, xmin, ymin, xmax, ymax = values
    # filename, time_stamp, width, height, obj_class, xmin, ymin, xmax, ymax, obj_id, x_pos, y_pos = values
    image_name, width, height, object_name, xmin, ymin, xmax, ymax, score=values
    filename=image_name
    # filename="/home/ubuntu/Documents/traffic_combined/gwmdata/gwm_sq_tf/"+filename
    # filename="/home/mayanksati/Desktop/square_tf/traffic_light_combined_to_train/"+filename
    ################################################333
    # image_name1 = filename.split("/")[-1]
    # image_name2 = filename.split("/")[-2]
    #
    # folder_path = filename.split("/")[-1]
    # new_file_path = (dir_path + str(image_name2) + '/' + str(image_name1))
    # new
    # for root, dirs, files in os.walk(dir_path):
    #     for file in files:
    #         if file==filename:
    #             new_file_path=(root + '/' + str(file))
    #             break
    #####################################################
    base_ground_file_path = (os.path.join(base_image_path, filename))
    if os.path.exists(base_ground_file_path):

        data_label = [image_name, width, height, object_name, int(xmin), int(ymin), int(xmax), int(ymax),score ]
        if xmin < 0 or ymin < 0:
            # if xmin < 0 or ymin < 0 or xmax>500 or ymax>500:
            print(filename)
            continue
        bblabel.append(data_label)
        # source=os.path.join(source, filename)
        # if not os.path.exists(desitnation):
        # shutil.move(source,desitnation)
        shutil.copy(os.path.join(source, filename), os.path.join(desitnation, filename))
    #
columns = ['filename', 'width', 'height', 'class', 'xmin', 'ymin', 'xmax', 'ymax', 'score']

df = pd.DataFrame(bblabel, columns=columns)
df.to_csv('/home/mayank_s/Desktop/custom_farm_front_tl/farm_parking_1/farm_custom_parking_1_final.csv', index=False)
# print('total imaages are',counter1)
# y = datasets.iloc[:, 4].values
