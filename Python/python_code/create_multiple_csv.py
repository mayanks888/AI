import os

import pandas as pd

# file_name1="/home/mayanksati/Desktop/csv_merge/re3_box.csv"
# # file_name2="/home/mayanksati/Desktop/csv_merge/re3_box_2.csv"
#
f1 = '/home/mayank_sati/Desktop/point_csv/pointpillar3.csv'
f2 = '/home/mayank_sati/Desktop/point_csv/pointpillar7.csv'
f3 = "/home/mayank_sati/Desktop/point_csv/pointpillar9.csv"
# f4="/home/mayanksati/PycharmProjects/Data_Science_new/lidar/point_csv/pointpillar10.csv"
# f5="/home/mayanksati/PycharmProjects/Data_Science_new/lidar/point_csv/pointpillar11.csv"
# f6="/home/mayanksati/PycharmProjects/Data_Science_new/lidar/point_csv/pointpillar12.csv"
# f7="/home/mayanksati/PycharmProjects/Data_Science_new/lidar/point_csv/pointpillar13.csv"
# f8="/home/mayanksati/PycharmProjects/Data_Science_new/lidar/point_csv/pointpillar13.csv"
# f9="/home/mayanksati/PycharmProjects/Data_Science_new/lidar/point_csv/pointpillar16.csv"

df1 = pd.read_csv(f1)
df2 = pd.read_csv(f2)
df3 = pd.read_csv(f3)
# df4=pd.read_csv(f4)
# df5=pd.read_csv(f5)
# df6=pd.read_csv(f6)
# df7=pd.read_csv(f7)
# df8=pd.read_csv(f8)
# df9=pd.read_csv(f9)


frames = [df1, df2, df3]

result = pd.concat(frames)
# columns = ['filename', 'width', 'height', 'class', 'xmin', 'ymin', 'xmax', 'ymax']
# df = pd.DataFrame(frames, columns=columns )
df = pd.DataFrame(result)
# df.to_csv('/home/mayank_sati/Desktop/point_csv',index=False)

frames = []
input_folder = '/home/mayank_sati/Desktop/point_csv'
for root, _, filenames in os.walk(input_folder):
    if (len(filenames) == 0):
        print("Input folder is empty")
        break
    for filename in filenames:
        file_path = input_folder + "/" + filename
        df_new = pd.read_csv(file_path)
        # df.append(df_new, ignore_index=True)
        df.append(df_new)
        # frames=frames.append(df)

# result = pd.concat(frames)
df = pd.DataFrame(result)
df.to_csv('coo_compined.csv', index=False)
