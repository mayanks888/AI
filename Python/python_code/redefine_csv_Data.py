import pandas as pd

# csv_path="/home/mayanksati/Documents/csv/BBD_val_traffic_light_signle.csv"
# csv_path="/home/mayanksati/Documents/Rosbag_files/short_range_images/combined.csv"
# csv_path="/home/mayank_sati/Desktop/TL_seoul_Train.csv"
# dir_path='/home/mayank_sati/Desktop/sorting_light'
dir_path = '/home/mayank_sati/Documents/datsets/seol/images/'
# csv_path="/home/mayanksati/Documents/csv/BBD_val_traffic_light_signle.csv"
# csv_path="/home/mayanksati/Documents/Rosbag_files/short_range_images/combined.csv"
csv_path = "/home/mayank_s/codebase/AI/Machine_learning/Python/python_code/redefined.csv"
# csv_path="/home/mayanksati/PycharmProjects/Data_Science_new/Deep learning/Load Datasets/BBD_daytime_train.csv"
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
bblabel = []
new_file_path = ''
for values in data:
    filename, image_width, image_height, object_name, xmin, ymin, xmax, ymax,x,r = values
    # filename="/home/ubuntu/Documents/traffic_combined/gwmdata/gwm_sq_tf/"+filename
    # filename="/home/mayanksati/Desktop/square_tf/traffic_light_combined_to_train/"+filename
    ################################################333
    image_name1 = filename.split("/")[-1]
    image_name2 = filename.split("/")[-2]

    folder_path = filename.split("/")[-1]
    new_file_path = (dir_path + str(image_name2) + '/' + str(image_name1))
    # new
    # for root, dirs, files in os.walk(dir_path):
    #     for file in files:
    #         if file==filename:
    #             new_file_path=(root + '/' + str(file))
    #             break
    #####################################################

    data_label = [new_file_path, image_width, image_height, object_name, xmin, ymin, xmax, ymax]
    if xmin < 0 or ymin < 0:
        # if xmin < 0 or ymin < 0 or xmax>500 or ymax>500:
        print(filename)
        continue
    bblabel.append(data_label)
#
columns = ['filename', 'width', 'height', 'class', 'xmin', 'ymin', 'xmax', 'ymax']

df = pd.DataFrame(bblabel, columns=columns)
df.to_csv('seol_myayank_train.csv', index=False)
# print('total imaages are',counter1)
# y = datasets.iloc[:, 4].values
