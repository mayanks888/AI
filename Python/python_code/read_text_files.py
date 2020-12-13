import os

import pandas as pd

# input_folder="/home/mayank_sati/Documents/ob_apollo_build/jin_apollo/modules/perception/traffic_light/rectify"
# input_folder="/home/mayank_sati/Documents/ob_apollo_build/jin_apollo"
input_folder = "/home/mayank_sati/Documents/ob_apollo_build (copy)/jin_apollo"
bblabel = []
for root, _, filenames in os.walk(input_folder):
    if (len(filenames) == 0):
        print("Input folder is empty")
    for filename in filenames:

        try:
            file_path = root + "/" + filename
            with open(file_path) as fp:
                line = fp.readline()
                cnt = 1
                while line:
                    # print("Line {}: {}".format(cnt, line.strip()))
                    # if "AERROR" or "AWARN" in line:
                    if "AWARN" in line:
                        print("Line {}: {}".format(cnt, line.strip()))
                        data_label = [file_path, line.strip()]
                        bblabel.append(data_label)

                    if "AERROR" in line:
                        print("Line {}: {}".format(cnt, line.strip()))
                        data_label = [file_path, line.strip()]
                        bblabel.append(data_label)

                    line = fp.readline()
                    cnt += 1

        except:
            1
            # print(filename)

columns = ['filename', "error"]

df = pd.DataFrame(bblabel, columns=columns)
df.to_csv('error_6.csv')
