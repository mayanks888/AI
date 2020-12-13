#never user the json file from the coco result, they something like (80class to 90 object thus destroy the whole thing somehow)
# so try to create you own csv from prediction and then create txt file to compar properly
from collections import namedtuple
import os
import pandas as pd



# csv_path='BBD_daytime_val.csv'
csv_path='/home/mayank_s/datasets/bdd/training_set/only_csv/front_light_bdd.csv'
#where to place
save_path = "/home/mayank_s/datasets/bdd/training_set/train_bdd_front_light"
data = pd.read_csv(csv_path)
print(data.head())

with open("bdd100k_single_tl.names") as f:
# with open("bdd_names_list.txt") as f:
  obj_list = f.readlines()
## remove whitespace characters like `\n` at the end of each line
  obj_list = [x.strip() for x in obj_list]


def split(df, group):
    data = namedtuple('data', ['filename', 'object'])
    # filename='img_name'
    # data = namedtuple('data', ['img_name', 'obj_class'])
    gb = df.groupby(group)
    return [data(filename, gb.get_group(x)) for filename, x in zip(gb.groups.keys(), gb.groups)]


grouped = split(data, 'filename')

if not os.path.exists(save_path):
  os.makedirs(save_path)
counter=0
for group in grouped:

        # filename = group.filename.encode('utf8')
            filename = group.filename
            # @@@@@@@@@@@@@
            # # this is only specific to coco 2014 validation
            # # test_string = 'COCO_val2014_'
            # K = '0'
            # # No. of zeros required
            # N = 12-len(str(filename))
            # # using format()
            # # Append K character N times
            # temp = '{:<' + K + str(len(test_string) + N) + '}'
            # res = temp.format(test_string)
            # # @@@@@@@@@@@@@@@@@
            filename=str(filename).split(".jpg")[0]
            txt_file_name=str(filename)+".txt"
            # txt_file_name=txt_file_name+".txt"
            txt_file_path=save_path+"/"+txt_file_name
            with open(txt_file_path, 'w') as f:
                for index, row in group.object.iterrows():
                    # bbox_x_normalized, bbox_y_normalized, bbox_width_normalized, bbox_height_normalized
                    bbox_x_normalized= (row['bbox_x_normalized'])
                    bbox_y_normalized=(row['bbox_y_normalized'])
                    bbox_width_normalized=(row['bbox_width_normalized'])
                    bbox_height_normalized=(row['bbox_height_normalized'])
                    # score=row['conf']
                    # obj_name=row["subclass"]
                    obj_name=row["class"]
                    obj_id = obj_list.index(obj_name)
                    if len(obj_name.split(" ")) > 1:
                        print(obj_name)
                        obj_name = obj_name.replace(" ", "_")
                        print(obj_name)
                    # f.write(str(bboxcls) + " " + " ".join([str(a) for a in bb]) + '\n')
                    bb = ((bbox_x_normalized), (bbox_y_normalized), (bbox_width_normalized),(bbox_height_normalized))

                    f.write(str(obj_id) + " " + " ".join([str(a) for a in bb]) + '\n')
        #
        # else:
        #     print(filename)
