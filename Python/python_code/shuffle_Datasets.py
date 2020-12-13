import pandas as pd
from sklearn.utils import shuffle

data = pd.read_csv(
    '/home/mayank_sati/Documents/datsets/Rosbag_files/short_range_images/for retina/gwm_seol_train_filter_Combined.csv')
df = shuffle(data)

df = pd.DataFrame(df)
df.to_csv(
    '/home/mayank_sati/Documents/datsets/Rosbag_files/short_range_images/for retina/gwm_seol_train_filter_Combined_shuffle.csv',
    index=False)
# df.to_csv('/home/
