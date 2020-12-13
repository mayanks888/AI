import os
import shutil

output_folder = '/home/mayank_sati/Desktop/run_del/create_annot/farm_eval'

copy_path = '/media/mayank_sati/DATA/datasets/one_shot_datasets/Farmington/images/2019-09-27-14-39-41_test_scaled'

# input_folder='/media/mayank_sati/DATA/datasets/one_shot_datasets/Farmington/images/2019-09-27-14-39-41_train'
input_folder = '/home/mayank_sati/Desktop/run_del/create_annot/farm_test'
for root, _, filenames in os.walk(input_folder):
    if (len(filenames) == 0):
        print("Input folder is empty")
        # return 1
    # time_start = time.time()
    for filename in filenames:
        # print("file : {fn}".format(fn=filename), '\n')
        # myfile_name=filename.split("_")[0]
        #
        # # if myfile_name in copy_path:
        # iamge_namenew=myfile_name+'.jpg'

        iamge_namenew = filename
        base_file_path = (os.path.join(copy_path, iamge_namenew))
        if os.path.exists(base_file_path):
            output_path = (os.path.join(output_folder, filename))
            shutil.copyfile(base_file_path, output_path)
        else:
            print(iamge_namenew)
        # cv2.imwrite(output_path, imageToPredict)

        #
        # shutil.copyfile('path/to/1.jpg', 'new/path/to/1.jpg')
