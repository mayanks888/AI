#remember the output of json to csv will genreate the bbox in (xmin,ymin,xmax,yman format)
# and from json format it came like ( xmin, ymin , width, height)
import os
import cv2
import pandas as pd
import tqdm
# csv_path='yolo1.csv'
# csv_path='BBD_daytime_train.csv'
csv_path='/home/mayank_s/Desktop/custom_farm_front_tl/farm_all_front_datasets/farm_final_imges.csv'
# csv_path='/home/mayank_s/datasets/bdd/training_set/only_csv/front_light_bdd.csv'
# root='/home/mayank_s/datasets/bdd/bdd100k_images/bdd100k/images/100k/train'
root='/home/mayank_s/Desktop/custom_farm_front_tl/farm_all_front_datasets/farm_final_imges'
saving_path = "/home/mayank_s/Desktop/custom_farm_front_tl/farmington_all/checkout"
data = pd.read_csv(csv_path)
# mydata = data.groupby('img_name')
mydata = data.groupby(['filename'], sort=True)
# print(data.groupby('class').count())
len_group = mydata.ngroups
mygroup = mydata.groups
###############################################3333
x = data.iloc[:, 0].values
y = data.iloc[:, 4:8].values
z = data.iloc[:, -1].values
##################################################
loop = 0
for ind, da1 in enumerate(sorted(mygroup.keys())):
    loop += 1
    # print(da1)
    index = mydata.groups[da1].values
    ###########33
    # @@@@@@@@@@@@@
    # this is only specific to coco 2014 validation
    # test_string = 'COCO_val2014_'
    # K = '0'
    # # No. of zeros required
    # N = 12 - len(str(da1))
    # # using format()
    # # Append K character N times
    # temp = '{:<' + K + str(len(test_string) + N) + '}'
    # res = temp.format(test_string)
    # txt_file_name = res + str(da1) + ".jpg"
    ##############3
    da = os.path.join(root, da1)
    # image_scale = cv2.imread(image_path, 1)
    image_scale = cv2.imread(da, 1)
    # try:
    height = image_scale.shape[0]
    width = image_scale.shape[1]
    for read_index in index:
        # print(index)
        top = (int(y[read_index][0]), int(y[read_index][3]))
        bottom = (int(y[read_index][2]), int(y[read_index][1]))
        cv2.rectangle(image_scale, pt1=top, pt2=bottom, color=(0, 255, 0), thickness=2)
        cv2.putText(image_scale, str(z[read_index]),(int((y[read_index][0] + y[read_index][2]) / 2), int(y[read_index][1] - 10)), cv2.FONT_HERSHEY_SIMPLEX, .5, (255, 255, 0), lineType=cv2.LINE_AA)

        if (y[read_index][2] > width) or y[read_index][3] > height:
            print(da1)
    # except:
    #     print(da1)

    flag = True
    if flag:
        cv2.imshow('streched_image', image_scale)
        ch = cv2.waitKey(1)
    #     if ch & 0XFF == ord('q'):
    #         cv2.destroyAllWindows()
    # cv2.waitKey(1)
    # cv2.destroyAllWindows()
    output_path = saving_path + da1
    # cv2.imwrite(output_path, image_scale)
    # cv2.destroyAllWindows()



