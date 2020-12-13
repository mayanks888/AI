import os
import natsort
import pandas as pd
# labelname = self.imagename + '.txt'
labelfilename = '/media/mayank_sati/DATA/mayank_data/apollo/ob_apollo_build/mayank_apollo/mynew_big.txt'

# input_folder=''

# for root, _, filenames in os.walk(input_folder):
#             if (len(filenames) == 0):
#                 print("Input folder is empty")
#             # time_start = time.time()
#             filenames = natsort.natsorted(filenames, reverse=False)
#             for filename in filenames:
#                 filename=
bblabel=[]
if os.path.exists(labelfilename):
    with open(labelfilename) as f:
        for (i, line) in enumerate(f):
            data = line.strip().split('_')

            data_label = [data[0],data[1],data[2]]
            bblabel.append(data_label)

columns = ['timestamp', 'x_pos', 'y_pos']
df = pd.DataFrame(bblabel, columns=columns)
df.to_csv('time_stam_pos_farm.csv', index=False)
        # tmp = self.deconvert(yolo_data[1:])
            # tmp = [int(float(loop)) for loop in data[1:-1]]

#
# def find_template(ref_id,ref_pos,csv_name):
#     # df = pd.read_csv("xshui (copy).csv")
#     csv_path="/media/mayank_sati/DATA/datasets/gwm_specific/xshui_all/"+csv_name
#     df = pd.read_csv(csv_path)
#     # df = pd.read_csv("/home/mayank_sati/Documents/datsets/Rosbag_files/xshui_all/s-n-1.csv")
#     # data=pd.read_csv('/home/mayank-s/Desktop/Link to Datasets/aptiveBB/reddy.csv')
#
#     # ref_id = 1001
#     # ref_pos = [369152.294564, 4321492.85272]
#
#     # grouped = df.groupby('img_name', sort=True)
#     mygroup = df.groupby(['obj_id'], sort=True)
#     # return [data(filename, gb.get_group(x)) for filename, x in zip(gb.groups.keys(), gb.groups)]
#     mydata = mygroup.groups
#     concerned_list = mydata[ref_id].values
#     data = df.values[concerned_list]
#     ###############################33333
#     # calculating eucldean distance
#     x_dif = abs(data[:, 10] - ref_pos[0])
#     y_dif = abs(data[:, 11] - ref_pos[1])
#     total_dif = x_dif + y_dif
#
#     min_dist_val=np.amin(total_dif)
#     print(min_dist_val)
#     if min_dist_val > 0.8:
#         # print(np.amin(total_dif)
#         # flag=True
#         return 1,1
#     #######################################
#     index = np.argmin(total_dif)
#     img_name = data[index][0]
#     bbox = data[index][5:9]
#     # print(img_name)
#     # print(bbox)q
#     return(img_name,bbox)
