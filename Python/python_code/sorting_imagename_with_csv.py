import os
import natsort
import pandas as pd
import numpy as np
import cv2
# labelname = self.imagename + '.txt'
labelfilename = '/media/mayank_sati/DATA/mayank_data/apollo/ob_apollo_build/mayank_apollo/mynew_big.txt'


input_folder='/home/mayank_sati/Desktop/images'
csv_path="/home/mayank_sati/Desktop/time_stam_pos_farm.csv"
new_path="/media/mayank_sati/DATA/datasets/one_shot_datasets/Farmington/images/2019-09-27-14-39-41_new"
df = pd.read_csv(csv_path)
counter=0
for root, _, filenames in os.walk(input_folder):
            if (len(filenames) == 0):
                print("Input folder is empty")
            # time_start = time.time()
            filenames = natsort.natsorted(filenames, reverse=False)
            for filename in filenames:
                filename1=(filename.split("."))
                timpe_val=int(filename1[0])*.000000001
                print(timpe_val)
#####################################################################3
                # mygroup = df.groupby(['timestamp'], sort=True)
                # # return [data(filename, gb.get_group(x)) for filename, x in zip(gb.groups.keys(), gb.groups)]
                # mydata = mygroup.groups
                # concerned_list = mydata[timpe_val].values
                # data = df.values[concerned_list]


                all_timestam= (df.iloc[:,0].values)
                x_dif = abs(all_timestam - timpe_val)

                index = np.argmin(x_dif)

                xval=df.iloc[index,1]
                yval=df.iloc[index,2]
                1
                file_name=str(filename1[0])+"_"+str(xval)+"_"+str(yval)+'.jpg'
                # img_name = data[index][0]

                ##################################3333
                file_name=new_path+"/"+file_name
                print(file_name)
                counter+=1
                print(counter)
                img=cv2.imread(input_folder+"/"+filename)
                cv2.imwrite( file_name,img)


