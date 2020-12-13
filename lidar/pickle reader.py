import pickle

import pandas as pd

# path='/home/mayanksati/Documents/point_clouds/result.pkl'
# path='/home/mayanksati/Documents/point_clouds/step_296960/result.pkl'
path = '/home/mayanksati/Documents/point_clouds/KITTI_DATASET_ROOT/kitti_dbinfos_train.pkl'
pickle_in = open(path, "rb")
example_dict = pickle.load(pickle_in)
print(1)
print(1)
bblabel = []
path = '/home/mayanksati/Documents/point_clouds/step_296960/000016.txt'
image_list = [line.strip().split(' ') for line in open(path)]
# image_list = [line.strip().split(' ')[0] for line in open(path)]
file_name = '/home/mayanksati/Documents/point_clouds/KITTI_DATASET_ROOT/testing/image_2/000016.png'
for value in image_list:
    # value
    # a=value[4]
    # b=float(a)
    # c=int(b)
    ##################################################33
    xmin = int(float(value[4]))
    ymin = int(float(value[5]))
    xmax = int(float(value[6]))
    ymax = int(float(value[7]))
    loc_x = int(float(value[11]))
    loc_y = int(float(value[12]))
    loc_z = int(float(value[12]))
    width = 520
    height = 1025
    # coordinate = [xmin, ymin, xmax, ymax, class_num]
    # object_name=object_name+"_"+light_color
    object_name = "car"
    data_label = [file_name, width, height, object_name, xmin, ymin, xmax, ymax, loc_x, loc_y, loc_z]
    # data_label = [file_name, width, height, object_name, xmin, ymin, xmax, ymax]
    if not ((xmin == xmax) and (ymin == ymax)):
        bblabel.append(data_label)
        # print()

columns = ['filename', 'width', 'height', 'class', 'xmin', 'ymin', 'xmax', 'ymax', 'loc_x', 'loc_y', 'loc_z']

df = pd.DataFrame(bblabel, columns=columns)
df.to_csv('pointpillar16.csv')
####################################################
print(1)
