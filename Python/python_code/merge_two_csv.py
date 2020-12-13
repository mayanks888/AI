import pandas as pd

# file_name1="/home/mayanksati/Desktop/csv_merge/re3_box.csv"
# file_name2="/home/mayanksati/Desktop/csv_merge/re3_box_2.csv"

f1 = '/home/mayank_sati/Documents/datsets/Rosbag_files/short_range_images/for retina/gwm_data_all_forRetina_osl.csv'
f2 = '/home/mayank_sati/Documents/datsets/Rosbag_files/short_range_images/for retina/seol_myayank_train.csv'

# f1='/home/mayank_sati/Documents/datsets/Rosbag_files/short_range_images/2018-11-24-09-25-36_copy/re1.csv'
# f2='/home/mayank_sati/Documents/datsets/Rosbag_files/short_range_images/2018-11-24-09-31-33/re3_3.csv'
# f3="/home/mayank_sati/Documents/datsets/Rosbag_files/short_range_images/2018-11-24-09-44-06/re3_4.csv"
# f4="/home/mayank_sati/Documents/datsets/Rosbag_files/short_range_images/2018-11-24-09-45-27/re3_5.csv"
# f5="/home/mayank_sati/Documents/datsets/Rosbag_files/short_range_images/2018-11-24-09-47-09/re3_6.csv"
# # f6="/home/mayank_sati/Documents/datsets/Rosbag_files/square data/square_data2-2/square2_2.csv"
# f7="/home/mayank_sati/Documents/datsets/Rosbag_files/square data/square_data2-3/square2_3.csv"
# f8="/home/mayank_sati/Documents/datsets/Rosbag_files/square data/square_data3-1/square3_1.csv"
# f9="/home/mayank_sati/Documents/datsets/Rosbag_files/square data/square_data3-2/square3_2.csv"
# f10="/home/mayank_sati/Documents/datsets/Rosbag_files/square data/square_data3-3/square3_3.csv"

df1 = pd.read_csv(f1)
df2 = pd.read_csv(f2)
# df3=pd.read_csv(f3)
# df4=pd.read_csv(f4)
# df5=pd.read_csv(f5)
# df6=pd.read_csv(f6)
# df7=pd.read_csv(f7)
# df8=pd.read_csv(f8)
# df9=pd.read_csv(f9)
# df10=pd.read_csv(f10)

frames = [df1, df2]  # ,df3,df4,df5]#,df6,df7,df7,df9,df10]

result = pd.concat(frames)
# columns = ['filename', 'width', 'height', 'class', 'xmin', 'ymin', 'xmax', 'ymax']
# df = pd.DataFrame(frames, columns=columns )
df = pd.DataFrame(result)
df.to_csv(
    '/home/mayank_sati/Documents/datsets/Rosbag_files/short_range_images/for retina/gwm_seol_train_filter_Combined.csv',
    index=False)
# df.to_csv('/home/mayank_sati/Documents/datsets/Rosbag_files/short_range_images/square_mayank.csv',index=False)
