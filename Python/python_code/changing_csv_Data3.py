import os

import pandas as pd

# csv_path="/home/mayanksati/Documents/csv/BBD_val_traffic_light_signle.csv"
# csv_path="/home/mayanksati/Documents/Rosbag_files/short_range_images/combined.csv"
# csv_path="/home/mayank_sati/Desktop/TL_seoul_Train.csv"
# dir_path='/home/mayank_sati/Desktop/sorting_light'
# dir_path = '/home/mayank_sati/Documents/datsets/seol/images/'
# csv_path="/home/mayanksati/Documents/csv/BBD_val_traffic_light_signle.csv"
# csv_path="/home/mayanksati/Documents/Rosbag_files/short_range_images/combined.csv"
csv_path = "/home/mayank_s/datasets/all_labels/ddltld/csv/driveU_val_filter.csv"
# csv_path="/home/mayanksati/PycharmProjects/Data_Science_new/Deep learning/Load Datasets/BBD_daytime_train.csv"
base_image_path = '/home/mayank_sati/Desktop/farm_train'
datasets = pd.read_csv(csv_path)
# df=data[(data['xmin']!=data['xmax']) & (data['ymin']!=data['ymax'])]
data = datasets.iloc[:].values
# image_folder_path="/home/mayanksati/Documents/datasets/BDD/bdd100k/images/100k/train"
# left_limit=right_limit=200
# upper_limit=360
# size_limit=30
# save_path = "/home/mayanksati/Desktop/bdd_crop_roi/"
# loop=0
# counter=0
# counter1=0
id_list = [1001, 1002, 1003]
bblabel = []
new_file_path = ''
for values in data:
    filename, image_width, image_height, object_name, xmin, ymin, xmax, ymax,class_id,_,_ = values
    # filename, time_stamp, width, height, obj_class, xmin, ymin, xmax, ymax, obj_id, x_pos, y_pos = values

    # filename="/home/ubuntu/Documents/traffic_combined/gwmdata/gwm_sq_tf/"+filename
    # filename="/home/mayanksati/Desktop/square_tf/traffic_light_combined_to_train/"+filename
    ################################################333
    # class_id_split = class_id.split("")
    digits = [int(x) for x in str(class_id)]
    if digits[0]==1:
        data_label = [filename, image_width, image_height, object_name, xmin, ymin, xmax, ymax,class_id]
        if xmin < 0 or ymin < 0:
            # if xmin < 0 or ymin < 0 or xmax>500 or ymax>500:
            print(filename)
            continue
        bblabel.append(data_label)

columns = ['filename', 'width', 'height', 'class', 'xmin', 'ymin', 'xmax', 'ymax','class_id']

df = pd.DataFrame(bblabel, columns=columns)
# df.to_csv('ddld_front_filter_train.csv', index=False)
df.to_csv('/home/mayank_s/datasets/all_labels/ddltld/csv/ddld_front_filter_val.csv', index=False)
# print('total imaages are',counter1)
# y = datasets.iloc[:, 4].values
