import glob
import os

import natsort
import pandas as pd

filename = []
width = 0
height = 0
Class = []
xmin = []
ymin = []
xmax = []
ymax = []
light_color = []
bblabel = []
loop = 1

############################################33
file_dir = '/media/mayank_sati/DATA/datasets/one_shot_datasets/Farmington/images/new_farm_imageS_with_correct_box_n_qual/tenflow_api_result/result_at_250000_and_farm_added/pred_at_.7tr'
filelist = glob.glob(os.path.join(file_dir, '*.txt'))
filelist = natsort.natsorted(filelist, reverse=False)
if len(filelist) == 0:
    print ('No .txt files found in the specified dir!')
for files in filelist:
    # if os.path.exists(labelfilename):
    title, ext = os.path.splitext(os.path.basename(files))
    print (title)
    with open(files) as f:
        for (i, line) in enumerate(f):
            file_data = line.strip().split()
            img_name = title + ".jpg"
            # obj_class=file_data[0]
            obj_class = "traffic_light"
            xmin = int(float(file_data[2]))
            ymin = int(float(file_data[3]))
            xmax = int(float(file_data[4]))
            ymax = int(float(file_data[5]))
            score = (float(file_data[1]))
            # time_stamp = title.split("_")[0]
            # x_pos = title.split("_")[1]
            # y_pos = title.split("_")[2]
            # obj_id = file_data[5]
            # if (obj_id==1001 or obj_id==1003 or obj_id==1004):
            # if (obj_id == 2099):
            #     continue
            data_label = [img_name, width, height, obj_class, xmin, ymin, xmax, ymax, score]
            bblabel.append(data_label)
            #################################################
            # temp_path = "/home/mayank_sati/Documents/datsets/Rosbag_files/xshui_all/s-n-2"
            # temp_path_2 = "/home/mayank_sati/Documents/datsets/Rosbag_files/xshui_all/s-n-2_crop"
            # if not os.path.exists(temp_path_2):
            #     print("base_path folder not present. Creating New folder...")
            #     os.makedirs(temp_path_2)
            #
            # temp_path = temp_path + '/' + img_name
            # img = cv2.imread(temp_path)
            # frame = img[int(ymin):int(ymax), int(xmin):int(xmax)]
            # # frame = img[int(ymin-100):int(ymax+100), int(xmin-100):int(xmax+100)]
            # # frame = img[int(y[value][1]):int(y[value][3]), int(y[value][0]):int(y[value][2])]
            # # frame = img[int(y[value][1]-40):int(y[value][3]+40), int(y[value][0]-30):int(y[value][2]+30)]
            # temp_path_2 = temp_path_2 + '/' + img_name
            # cv2.imwrite(temp_path_2, frame)
# columns = ['img_name', 'time_stamp', 'width', 'height', 'obj_class', 'xmin', 'ymin', 'xmax', 'ymax', 'obj_id', 'x_pos', 'y_pos']
columns = ['filename', 'width', 'height', 'class', 'xmin', 'ymin', 'xmax', 'ymax', 'score']

df = pd.DataFrame(bblabel, columns=columns)
# print("into csv file")
df.to_csv(
    '/media/mayank_sati/DATA/datasets/one_shot_datasets/Farmington/images/new_farm_imageS_with_correct_box_n_qual/tenflow_api_result/result_at_250000_and_farm_added/prediction_farm.csv',
    index=False)
# csv_name = file_dir.split("/")[-1]
# csv_name = file_dir + ".csv"
# df.to_csv(csv_name, index=False)

#
# for section in cfg:
#     print(section)
#     loop += 1
#     xmax = section['boxes'][0]['boxes']['xmax']
#     xmin = section['boxes'][0]['boxes']['xmin']
#     ymax = section['boxes'][0]['boxes']['ymax']
#     ymin = section['boxes'][0]['boxes']['ymin']
#     height = section['boxes'][0]['boxes']['height']
#     width = section['boxes'][0]['boxes']['width']
#     # ___________________________________________
#     # playing with file
#     file_path = section['path']
#     file_name = file_path.split("/")[-1]
#     new_path = image_path + "/" + file_name
#     file_path = new_path
#     data_label = [file_path, width, height, object_name, xmin, ymin, xmax, ymax]
#     if not ((xmin == xmax) and (ymin == ymax)):
#         bblabel.append(data_label)
#         # print()
#     # if loop>=1018:
#     #     break
#
# columns = ['filename', 'width', 'height', 'class', 'xmin', 'ymin', 'xmax', 'ymax']
#
# df = pd.DataFrame(bblabel, columns=columns)
# print("into csv file")
# df.to_csv('square3_3.csv', index=False)
