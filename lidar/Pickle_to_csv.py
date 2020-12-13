import pickle

import pandas as pd

map_location = 'cpu'
# torch.load(map_location='cpu')
# path='/home/mayanksati/Documents/point_clouds/result.pkl'
# path='/home/mayanksati/Documents/point_clouds/step_296960/result.pkl'
path = '/home/mayanksati/Documents/point_clouds/read_pt_pickle.pkl'
pickle_in = open(path, "rb")
example_dict = pickle.load(pickle_in)
images_no = 0
width = 500
height = 1200
object_name = 'car'
bblabel = []
images_name = '/home/mayanksati/Documents/point_clouds/KITTI_DATASET_ROOT/imaes_for_showingbb/'  # /0.png'
images_name = '/home/mayanksati/Documents/point_clouds/KITTI_DATASET_ROOT/pycharm_work/point_cloud/00000'  # /0.png'
for example in example_dict:
    example
    loop = 0
    for boxes in example['bbox']:
        boxes
        file_name = images_no
        xmin = int(float(boxes[0]))
        ymin = int(float(boxes[1]))
        xmax = int(float(boxes[2]))
        ymax = int(float(boxes[3]))
        score = example['score'][loop]
        loc_x = (float(example['location'][loop][0]))
        loc_y = (float(example['location'][loop][1]))
        loc_z = (float(example['location'][loop][2]))
        if score > 0.5:
            file_name = images_name + str(images_no) + '.bin'
            # file_name=images_name+str(images_no)+'.png'
            data_label = [file_name, width, height, object_name, xmin, ymin, xmax, ymax, loc_x, loc_y, loc_z]
            # data_label = [file_name, width, height, object_name, xmin, ymin, xmax, ymax]#, loc_x, loc_y, loc_z]
            bblabel.append(data_label)
        loop += 1
    images_no += 1

# columns=['filename','width', 'height','class','xmin','ymin','xmax','ymax']
columns = ['filename', 'width', 'height', 'class', 'xmin', 'ymin', 'xmax', 'ymax', 'loc_x', 'loc_y', 'loc_z']

df = pd.DataFrame(bblabel, columns=columns)
# df.to_csv('ppillar2.csv',index=False)
df.to_csv('ppillar_bin.csv', index=False)
