import os

import pandas as pd

# file_name1="/home/mayanksati/Desktop/csv_merge/re3_box.csv"
# file_name2="/home/mayanksati/Desktop/csv_merge/re3_box_2.csv"
input_folder = "/home/mayanksati/Documents/Rosbag_files/short_range_images/only_csv"
combined_csv = []
for root, _, filenames in os.walk(input_folder):
    if (len(filenames) == 0):
        print("Input folder is empty")

    for filename in filenames:
        print(filename)
        csv_path = input_folder + "/" + filename
        df1 = pd.read_csv(csv_path)
        combined_csv.append(df1)

# f1='/home/mayanksati/PycharmProjects/Data_Science_new/lidar/point_csv/pointpillar3.csv'
# f2='/home/mayanksati/PycharmProjects/Data_Science_new/lidar/point_csv/pointpillar7.csv'
# f3="/home/mayanksati/PycharmProjects/Data_Science_new/lidar/point_csv/pointpillar9.csv"
# f4="/home/mayanksati/PycharmProjects/Data_Science_new/lidar/point_csv/pointpillar10.csv"
# f5="/home/mayanksati/PycharmProjects/Data_Science_new/lidar/point_csv/pointpillar11.csv"
# f6="/home/mayanksati/PycharmProjects/Data_Science_new/lidar/point_csv/pointpillar12.csv"
# f7="/home/mayanksati/PycharmProjects/Data_Science_new/lidar/point_csv/pointpillar13.csv"
# f8="/home/mayanksati/PycharmProjects/Data_Science_new/lidar/point_csv/pointpillar13.csv"
# f9="/home/mayanksati/PycharmProjects/Data_Science_new/lidar/point_csv/pointpillar16.csv"
#
# df1=pd.read_csv(f1)
# df2=pd.read_csv(f2)
# df3=pd.read_csv(f3)
# df4=pd.read_csv(f4)
# df5=pd.read_csv(f5)
# df6=pd.read_csv(f6)
# df7=pd.read_csv(f7)
# df8=pd.read_csv(f8)
# df9=pd.read_csv(f9)
#
# frames = [df1,df2,df3,df4,df5,df6,df7,df7,df9]

result = pd.concat(combined_csv)
# columns = ['filename', 'width', 'height', 'class', 'xmin', 'ymin', 'xmax', 'ymax']
# df = pd.DataFrame(frames, columns=columns )
df = pd.DataFrame(result)
# df.to_csv('/home/mayanksati/PycharmProjects/Data_Science_new/lidar/point_csv/pointpil.csv',index=False)
df.to_csv('combined.csv', index=False)
