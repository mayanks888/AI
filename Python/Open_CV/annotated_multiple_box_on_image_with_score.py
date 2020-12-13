import os
import cv2
import pandas as pd

# root=''
# root = "/media/mayank_sati/DATA/datasets/one_shot_datasets/Farmington/images/new_farm_imageS_with_correct_box_n_qual/farm_eval"
# # csv_path="/home/mayanksati/Documents/Rosbag_files/short_range_images/gwm_data_train.csv"
# csv_path = "/media/mayank_sati/DATA/datasets/one_shot_datasets/Farmington/images/new_farm_imageS_with_correct_box_n_qual/tenflow_api_result/result_at_250000_and_farm_added/prediction_farm.csv"

csv_path='/home/mayank_s/codebase/others/centernet/mayank/CenterNet/src/centernet_prediction_val.csv'
root='/home/mayank_s/datasets/bdd/bdd100k_images/bdd100k/images/100k/val'
saving_path="/home/mayank_sati/Desktop/farm_pred/"
data = pd.read_csv(csv_path)
mydata = data.groupby('filename')
print(data.groupby('class').count())
len_group = mydata.ngroups
mygroup = mydata.groups
new = data.groupby(['filename'])['class'].count()
###############################################3333

x = data.iloc[:, 0].values
y = data.iloc[:, 4:9].values
##################################################
loop = 1
for da1 in sorted(mygroup.keys()):
    index = mydata.groups[da1].values
    da = os.path.join(root, da1)
    # image_scale = cv2.imread(image_path, 1)
    image_scale = cv2.imread(da, 1)
    for read_index in index:
        print(index)
        top = (int(y[read_index][0]), int(y[read_index][3]))
        bottom = (int(y[read_index][2]), int(y[read_index][1]))
        cv2.rectangle(image_scale, pt1=top, pt2=bottom, color=(0, 255, 0), thickness=2)
        cv2.putText(image_scale, str(round(y[read_index][4],2)),(int((y[read_index][0] + y[read_index][2]) / 2), int(y[read_index][1] - 10)), cv2.FONT_HERSHEY_SIMPLEX, .5, (0, 255, 0), lineType=cv2.LINE_AA)
        # str(round(y[read_index][4]))
        # y[read_index][0]
        print(da)
        # break
    # cv2.imshow('streched_image', image_scale)
    # # output_folder_path="/home/mayank-s/PycharmProjects/Datasets/aptive/object_detect/output/output_merccedes.png"
    # # filepath=output_folder_path+my_image+".png"
    # # cv2.imwrite(filepath,my_image)
    # cv2.waitKey(1000)
    # output_path = (os.path.join(saving_path, str(loop)+".jpg"))
    # loop += 1
    # cv2.imwrite(output_path, image_scale)
    # cv2.destroyAllWindows()
    # val += 1
    # print(val)
    cv2.imshow('streched_image', image_scale)
    ch = cv2.waitKey(0)
    if ch & 0XFF == ord('q'):
        cv2.destroyAllWindows()
    # cv2.waitKey(1)
    # cv2.destroyAllWindows()
    output_path = saving_path + da1
    # output_path = (os.path.join(saving_path, da))
    #
    # cv2.imwrite(output_path, image_scale)
    # cv2.destroyAllWindows()

# cv2.imwrite("../d