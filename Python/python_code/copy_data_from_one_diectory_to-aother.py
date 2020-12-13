import os
import shutil
point = ["samples", "sweeps"]
topic=["CAM_FRONT_LEFT","CAM_BACK_LEFT","CAM_BACK","CAM_BACK_RIGHT","CAM_FRONT_RIGHT","CAM_FRONT"]

src_path_1 = "/home/mayank_s/datasets/nuscene/dummy/1"
# src_path_1 = "/media/ads/DATA/mayank_folder_dont_delete/datasets/nuscene/v1.0_trainval_01"
src_path_2 = "/media/ads/DATA/mayank_folder_dont_delete/datasets/nuscene/v1.0_trainval_02"
src_path_3 = "/media/ads/DATA/mayank_folder_dont_delete/datasets/nuscene/v1.0_trainval_03"
src_path_4 = "/media/ads/DATA/mayank_folder_dont_delete/datasets/nuscene/v1.0_trainval_04"
src_path_5 = "/media/ads/DATA/mayank_folder_dont_delete/datasets/nuscene/v1.0_trainval_05"
src_path_6 = "/media/ads/DATA/mayank_folder_dont_delete/datasets/nuscene/v1.0_trainval_06"
src_path_7 = "/media/ads/DATA/mayank_folder_dont_delete/datasets/nuscene/v1.0_trainval_07"
src_path_8 = "/media/ads/DATA/mayank_folder_dont_delete/datasets/nuscene/v1.0_trainval_08"
src_path_9 = "/media/ads/DATA/mayank_folder_dont_delete/datasets/nuscene/v1.0_trainval_09"
src_path_10 = "/media/ads/DATA/mayank_folder_dont_delete/datasets/nuscene/v1.0_trainval_10"
# destination_path = "/media/ads/DATA/mayank_folder_dont_delete/datasets/nuscene/all"
destination_path = "/home/mayank_s/datasets/nuscene/dummy/output"

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
                print(dir)
                # print(dirpath)
                for (des_dirpath, des_dirnames, des_filenames) in walk(destination_path):
                    for desdir in des_dirnames:
                        # if desdir == "RADAR_FRONT":
                        #     1
                            # print(desdir)
                        if desdir == dir:
                            # print(desdir)
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
                                    shutil.copy(full_file_name, dest)


# file = [src_path_1, src_path_2,src_path_3, src_path_4,src_path_5, src_path_6,src_path_7, src_path_8,src_path_9, src_path_10]
file = [src_path_1]

for file_source in file:
    # print(file_source)
    for foldername in point:
        file_source1 = os.path.join(file_source, foldername)
        destination_path1 = os.path.join(destination_path, foldername)
        copy_data(file_source1, destination_path1)
