import os
import shutil
point = ["samples", "sweeps"]
# point = [ "sweeps"]
# topic=["CAM_FRONT_LEFT","CAM_BACK_LEFT","CAM_BACK","CAM_BACK_RIGHT","CAM_FRONT_RIGHT","CAM_FRONT"]
# topic=["LIDAR_TOP","RADAR_BACK_LEFT","RADAR_BACK_RIGHT","RADAR_FRONT","RADAR_FRONT_LEFT","RADAR_FRONT_RIGHT"]
topic=["LIDAR_TOP","RADAR_BACK_LEFT","RADAR_BACK_RIGHT","RADAR_FRONT","RADAR_FRONT_LEFT","RADAR_FRONT_RIGHT","CAM_FRONT_LEFT","CAM_BACK_LEFT","CAM_BACK","CAM_BACK_RIGHT","CAM_FRONT_RIGHT","CAM_FRONT"]

# src_path_demo = "/home/user/Documents/nuscenes/v1.0-mini"
# src_path_1 = "/media/mayank_sati/wd_2tb/myjob/datasets/detection/nuscenes/nuscenes_all/all/v1"
# src_path_2 = "/media/mayank_sati/wd_2tb/myjob/datasets/detection/nuscenes/nuscenes_all/all/v2"
# src_path_3 = "/home/mayank_sati/Documents/datasets/v3"
# src_path_4 = "/home/mayank_sati/Documents/datasets/v4"
src_path_5 = "/home/mayank_sati/Documents/datasets/v5"
# src_path_6 = "/home/mayank_sati/Documents/datasets/v6"
src_path_7 = "/home/mayank_sati/Documents/datasets/v7"
# src_path_8 = "/home/mayank_sati/Documents/datasets/V8"
# src_path_9 = "/home/mayank_sati/Documents/datasets/nuscenes/v9"
# src_path_10 = "/home/mayank_sati/Documents/datasets/nuscenes/v10/data"
destination_path = "/media/mayank_sati/wd_2tb/myjob/datasets/detection/nuscenes/v1.0-trainval"
# destination_path = "RADAR_FRONT_RIGHT"

# for root,_, filenames in os.walk(source_path):
#     if (len(filenames) == 0):
#         print("Input folder is empty")
#     for filename in filenames:
#         print (filenames)

from os import walk


def copy_data(source_path, destination_path):
    # print("source path " , source_path)
    # print("destination path " , destination_path)
    for (dirpath, dirnames, filenames) in walk(source_path):
        for dir in dirnames:
            if dir in topic:
                # print(dir)
                # print(dirpath)
                for (des_dirpath, des_dirnames, des_filenames) in walk(destination_path):
                    for desdir in des_dirnames:
                        # if desdir == "RADAR_FRONT":
                        #     1
                            # print(desdir)
                        if desdir == dir:
                            print(desdir)
                            # print(des_dirpath)
                            #################################333333
                            src = os.path.join(dirpath, dir)
                            dest = os.path.join(des_dirpath, desdir)
                            # if not os.path.exists(dest):
                            #     os.makedirs(dest)
                            # print(src, '\n', dest)
                            ###########################################
                            src_files = os.listdir(src)
                            for file_name in src_files:
                                # file_name=file_name+"new"
                                full_file_name = os.path.join(src, file_name)
                                if (os.path.isfile(full_file_name)):
                                    # print("source path ", source_path)
                                    # print(full_file_name)
                                    # shutil.copy(full_file_name, dest)
                                    checkdestination_path=os.path.join(dest, file_name)
                                    if os.path.exists(checkdestination_path):
                                        # os.remove(checkdestination_path)
                                        continue
                                    else:
                                        print(file_name, "was not present")
                                        shutil.copy(full_file_name, dest)
                                        # shutil.move(full_file_name, dest)


# file = [src_path_1, src_path_2,src_path_3, src_path_4,src_path_5, src_path_6,src_path_7, src_path_8,src_path_9, src_path_10]
file = [src_path_5,src_path_7]
# file = [src_path_demo]

for file_source in file:
    print(file_source)
    for foldername in point:
        print(foldername)
        file_source1 = os.path.join(file_source, foldername)
        destination_path1 = os.path.join(destination_path, foldername)
        copy_data(file_source1, destination_path1)
