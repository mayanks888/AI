import pandas as pd
import shutil
# csv_path="/home/mayank_s/datasets/bdd/training_set/only_csv/BBD_user_Select_lights.csv"
# datasets = pd.read_csv(csv_path)
#
# cool=datasets[(datasets['tl_save_or_not'] == "Y")]
#
# # df = pd.DataFrame(bblabel, columns=columns)
# cool.to_csv('front_light_bdd.csv', index=False)
1
original='/home/mayank_s/Desktop/yolo_onnx/2.jpg'
target='/home/mayank_s/Desktop/dummy'

# shutil.move(original,target)
shutil.copy(original, target)