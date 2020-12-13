import os

import cv2
import pandas as pd


def increase_bounding_box_scale(img, mybbox, scale_width, scale_height):
    print("old box", mybbox)
    bbox_info = mybbox
    height, width, depth = img.shape

    # x_y_min = bbox_info[:, 0]  # this is the array of all x min and y min in boxes
    # x_y_max = bbox_info[:, 1]  # this is a array of  al x max and y max in boxes
    # diff = x_y_max - x_y_min  # finding xmax - xmin and ymax - ymin

    centr_box = (int((bbox_info[0] + bbox_info[2]) / 2), int((bbox_info[1] + bbox_info[3]) / 2))
    temp_width = bbox_info[2] - bbox_info[0]
    temp_height = bbox_info[3] - bbox_info[1]

    bbox_info[1] = int(bbox_info[1] - (scale_height * (temp_height / 2)))
    bbox_info[3] = int(bbox_info[3] + (scale_height * (temp_height / 2)))

    bbox_info[0] = int(bbox_info[0] - (scale_width * (temp_width / 2)))
    bbox_info[2] = int(bbox_info[2] + (scale_width * (temp_width / 2)))

    temp_width = bbox_info[2] - bbox_info[0]
    temp_height = bbox_info[3] - bbox_info[1]

    if bbox_info[1] < 0:
        bbox_info[3] = bbox_info[3] + abs(bbox_info[1])
        bbox_info[1] = 0

    if bbox_info[3] > height:
        bbox_info[1] = bbox_info[1] - abs(bbox_info[1] - height)
        bbox_info[3] = height

    if bbox_info[0] < 0:
        bbox_info[2] = bbox_info[2] + abs(bbox_info[0])
        bbox_info[0] = 0

    if bbox_info[2] > width:
        bbox_info[0] = bbox_info[0] - abs(bbox_info[2] - width)
        bbox_info[3] = width

    return bbox_info


def increase_bounding_box_scale_diff_apr(img, mybbox, scale_width, scale_height):
    # print("old box",mybbox)
    bbox_info = mybbox
    height, width, depth = img.shape

    # x_y_min = bbox_info[:, 0]  # this is the array of all x min and y min in boxes
    # x_y_max = bbox_info[:, 1]  # this is a array of  al x max and y max in boxes
    # diff = x_y_max - x_y_min  # finding xmax - xmin and ymax - ymin
    centr_box = (int((bbox_info[0] + bbox_info[2]) / 2), int((bbox_info[1] + bbox_info[3]) / 2))
    temp_width = bbox_info[2] - bbox_info[0]
    temp_height = bbox_info[3] - bbox_info[1]

    bbox_info[1] = int(bbox_info[1] - (scale_height * (temp_height / 2)))
    bbox_info[3] = int(bbox_info[3] + (scale_height * (temp_height / 2)))

    bbox_info[0] = int(bbox_info[0] - (scale_width * (temp_width / 2)))
    bbox_info[2] = int(bbox_info[2] + (scale_width * (temp_width / 2)))

    temp_width = bbox_info[2] - bbox_info[0]
    temp_height = bbox_info[3] - bbox_info[1]

    if bbox_info[1] < 0:
        bbox_info[3] = bbox_info[3] - abs(bbox_info[1])
        bbox_info[1] = 0

    if bbox_info[3] > height:
        bbox_info[1] = bbox_info[1] + abs(bbox_info[1] - height)
        bbox_info[3] = height

    if bbox_info[0] < 0:
        bbox_info[2] = bbox_info[2] - abs(bbox_info[0])
        bbox_info[0] = 0

    if bbox_info[2] > width:
        bbox_info[0] = bbox_info[0] + abs(bbox_info[2] - width)
        bbox_info[2] = width

    return bbox_info


def increase_bounding_box_scale_diff_aproach_and_set_lim500(img, mybbox, scale_width, scale_height, box_wd_limit=500,
                                                            box_ht_limit=500):
    # print("old box",mybbox)

    bbox_info = mybbox
    height, width, depth = img.shape

    # x_y_min = bbox_info[:, 0]  # this is the array of all x min and y min in boxes
    # x_y_max = bbox_info[:, 1]  # this is a array of  al x max and y max in boxes
    # diff = x_y_max - x_y_min  # finding xmax - xmin and ymax - ymin
    centr_box = (int((bbox_info[0] + bbox_info[2]) / 2), int((bbox_info[1] + bbox_info[3]) / 2))
    temp_width = bbox_info[2] - bbox_info[0]
    temp_height = bbox_info[3] - bbox_info[1]

    bbox_info[1] = int(bbox_info[1] - (scale_height * (temp_height / 2)))
    bbox_info[3] = int(bbox_info[3] + (scale_height * (temp_height / 2)))

    bbox_info[0] = int(bbox_info[0] - (scale_width * (temp_width / 2)))
    bbox_info[2] = int(bbox_info[2] + (scale_width * (temp_width / 2)))

    temp_width = bbox_info[2] - bbox_info[0]
    temp_height = bbox_info[3] - bbox_info[1]

    if temp_width > box_wd_limit:
        crop_diff = (temp_width - box_wd_limit)
        bbox_info[0] = bbox_info[0] + (crop_diff / 2)
        bbox_info[2] = bbox_info[2] - (crop_diff / 2)

    if temp_height > box_ht_limit:
        crop_diff = (temp_height - box_ht_limit)
        bbox_info[1] = bbox_info[1] + (crop_diff / 2)
        bbox_info[3] = bbox_info[3] - (crop_diff / 2)

    if bbox_info[1] < 0:
        bbox_info[3] = bbox_info[3] - abs(bbox_info[1])
        bbox_info[1] = 0

    if bbox_info[3] > height:
        bbox_info[1] = bbox_info[1] + abs(bbox_info[1] - height)
        bbox_info[3] = height

    if bbox_info[0] < 0:
        bbox_info[2] = bbox_info[2] - abs(bbox_info[0])
        bbox_info[0] = 0

    if bbox_info[2] > width:
        bbox_info[0] = bbox_info[0] + abs(bbox_info[2] - width)
        bbox_info[2] = width

    return bbox_info


def get_iou(bb1, bb2):
    """
    Calculate the Intersection over Union (IoU) of two bounding boxes.

    Parameters
    ----------
    bb1 : dict
        Keys: {'x1', 'x2', 'y1', 'y2'}
        The (x1, y1) position is at the top left corner,
        the (x2, y2) position is at the bottom right corner
    bb2 : dict
        Keys: {'x1', 'x2', 'y1', 'y2'}
        The (x, y) position is at the top left corner,
        the (x2, y2) position is at the bottom right corner

    Returns
    -------
    float
        in [0, 1]
    """
    assert bb1['x1'] < bb1['x2']
    assert bb1['y1'] < bb1['y2']
    assert bb2['x1'] < bb2['x2']
    assert bb2['y1'] < bb2['y2']

    # determine the coordinates of the intersection rectangle
    x_left = max(bb1['x1'], bb2['x1'])
    y_top = max(bb1['y1'], bb2['y1'])
    x_right = min(bb1['x2'], bb2['x2'])
    y_bottom = min(bb1['y2'], bb2['y2'])

    if x_right < x_left or y_bottom < y_top:
        return 0.0

    # The intersection of two axis-aligned bounding boxes is always an
    # axis-aligned bounding box
    intersection_area = (x_right - x_left) * (y_bottom - y_top)

    # compute the area of both AABBs
    bb1_area = (bb1['x2'] - bb1['x1']) * (bb1['y2'] - bb1['y1'])
    bb2_area = (bb2['x2'] - bb2['x1']) * (bb2['y2'] - bb2['y1'])

    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = intersection_area / float(bb1_area + bb2_area - intersection_area)
    assert iou >= 0.0
    assert iou <= 1.0
    return iou


def bb_intersection_over_union(boxA, boxB):
    # determine the (x, y)-coordinates of the intersection rectangle
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    # compute the area of intersection rectangle
    # interArea = (xB - xA) * (yB - yA)
    interArea = max((xB - xA), 0) * max((yB - yA), 0)
    # compute the area of both the prediction and ground-truth
    # rectangles
    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])

    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = interArea / float(boxAArea + boxBArea - interArea)

    # return the intersection over union value
    return iou


# root=''
# root = "/home/mayank_sati/Desktop/one_Shot_learning/xshui"
root = "/media/mayank_sati/DATA/datasets/one_shot_datasets/Farmington/images/2019-09-27-14-39-41_train"

# root = "/media/mayank_sati/DATA/datasets/one_shot_datasets/Farmington/images/2019-09-27-14-39-41_test"
# csv_path="/home/mayanksati/Documents/Rosbag_files/short_range_images/gwm_data_train.csv"
# csv_path = '/home/mayank_sati/Desktop/git/2/AI/Annotation_tool_V3/system/Labels/xshui.csv'
csv_path = '/media/mayank_sati/DATA/datasets/one_shot_datasets/Farmington/images/for_farming_single_light/2019-09-27-14-39-41_train.csv'
# csv_path = '/media/mayank_sati/DATA/datasets/one_shot_datasets/Farmington/images/new_farm_imageS_with_correct_box_n_qual/2019-09-27-14-39-41_test.csv'
saving_path = "/home/mayank_sati/Desktop/mycropped_2/"
data = pd.read_csv(csv_path)
# mydata = data.groupby('img_name')
mydata = data.groupby(['img_name'], sort=True)
# print(data.groupby('class').count())
len_group = mydata.ngroups
mygroup = mydata.groups
# new = data.groupby(['img_name'])['class'].count()
###############################################3333
x = data.iloc[:, 0].values
y = data.iloc[:, 5:10].values
##################################################
loop = 0
listbox = []
smal_box = []
mycount = 0
for ind, da1 in enumerate(sorted(mygroup.keys())):
    listbox = []
    # print(ind)
    index = mydata.groups[da1].values
    da = os.path.join(root, da1)
    # image_scale = cv2.imread(image_path, 1)
    image_scale = cv2.imread(da, 1)
    for read_index in index:
        # print(index)
        old_bbox_val = y[read_index].copy()
        new_box = increase_bounding_box_scale_diff_aproach_and_set_lim500(image_scale, old_bbox_val, scale_width=0,
                                                                          scale_height=0)
        # new_box = increase_bounding_box_scale(image_scale, old_bbox_val, scale_width=20, scale_height=10)

        listbox.append(new_box)
        # smal_box=[]
    for lis in listbox:
        image_scale = cv2.imread(da, 1)
        # 3##############################################################################
        loop += 1
        new_image_name = da1.split(".jpg")[0]
        file_name = new_image_name + "_" + str(loop) + ".jpg"
        save_file_path = os.path.join(saving_path, file_name)

        # 3##############################################################################
        for read_index in index:
            local_bbox_val = y[read_index].copy()

            top = (lis[0], lis[3])
            bottom = (lis[2], lis[1])
            #
            # cv2.rectangle(image_scale, pt1=top, pt2=bottom, color=(0, 255, 0), thickness=2)

            iou = bb_intersection_over_union(local_bbox_val, lis)
            # iou=get_iou(local_bbox_val,lis)
            frame = image_scale[int(lis[1]):int(lis[3]), int(lis[0]):int(lis[2])]

            # frame = img[int(bbox_info[1] - temp_scale_2):int(bbox_info[3] + temp_scale_2), int(bbox_info[0] - temp_scale_2):int(bbox_info[2] + temp_scale_2)]
            # print(iou)
            if iou > 0:

                # smal_box.append(local_bbox_val)
                temp_width = local_bbox_val[2] - local_bbox_val[0]
                temp_height = local_bbox_val[3] - local_bbox_val[1]

                xmin_new = local_bbox_val[0] - lis[0]
                ymin_new = local_bbox_val[1] - lis[1]
                xmax_new = xmin_new + temp_width
                ymax_new = ymin_new + temp_height

                # top = (local_bbox_val[0], local_bbox_val[3])
                # bottom = (local_bbox_val[2], local_bbox_val[1])
                # smal_box.append()
                top = (xmin_new, ymax_new)
                bottom = (xmax_new, ymin_new)
                # cv2.rectangle(frame, pt1=top, pt2=bottom, color=(0, 0, 255), thickness=2)
                ##################33333
                object_name = "traffic_light"
                data_label = [file_name, 0, 0, object_name, xmin_new, ymin_new, xmax_new, ymax_new]
                # data_label = [file_name, width, height, object_name, xmin, ymin, xmax, ymax]
                if not ((xmin_new == xmax_new) and (ymin_new == ymax_new)):
                    if not (xmin_new < 0 or ymin_new < 0):
                        if not (xmax_new > int(frame.shape[1]) or ymax_new > frame.shape[0]):
                            # if not (y[read_index][2] > width) or y[read_index][3] > height:
                            smal_box.append(data_label)
                            cv2.imwrite(save_file_path, frame, [cv2.IMWRITE_JPEG_QUALITY, 100])
                            print(file_name)
                            print(mycount)
                            mycount += 1

                # 3##########################3

        # image_scale1=image_scale.copy()
        # cv2.imshow('streched_image', frame)
        #
        # ch = cv2.waitKey(100)
        # # cv2.destroyAllWindows()
        # if ch & 0XFF == ord('q'):
        #     cv2.destroyAllWindows()

columns = ['filename', 'width', 'height', 'class', 'xmin', 'ymin', 'xmax', 'ymax']
df = pd.DataFrame(smal_box, columns=columns)
df.to_csv('myfarm_crop.csv', index=False)
# exit()


# top = (y[read_index][0], y[read_index][3])
# bottom = (y[read_index][2], y[read_index][1])
# cv2.rectangle(image_scale, pt1=top, pt2=bottom, color=(0, 255, 0), thickness=2)
# # cv2.putText(image_scale, y[read_index][4], ((y[read_index][0]+y[read_index][2])/2, y[read_index][1]), cv2.CV_FONT_HERSHEY_SIMPLEX, 2, 255)
# cv2.putText(image_scale, str(y[read_index][4]),
#             ((y[read_index][0] + y[read_index][2]) / 2, y[read_index][1] - 10), cv2.FONT_HERSHEY_SIMPLEX, .5,
#             (0, 255, 0), lineType=cv2.LINE_AA)
# y[read_index][0]
# print(da)
# break
# if ind>0:
#     cv2.imshow('streched_image', image_scale)
#     ch = cv2.waitKey(1)
#     if ch & 0XFF == ord('q'):
#         cv2.destroyAllWindows()
# # cv2.waitKey(1)
# # cv2.destroyAllWindows()
# output_path = saving_path + da1
#
# cv2.imwrite(output_path, image_scale)
# cv2.destroyAllWindows()


'''
# dataset = pd.read_csv("SortedXmlresult.csv")
# /home/mayank-s/PycharmProjects/Datasets/Berkely DeepDrive/bdd100k/images/100k/train
# /home/mayank-s/PycharmProjects/Datasets/Berkely DeepDrive/berkely_train.csv
# csv_path="/home/mayanksati/PycharmProjects/Data_Science_new/lidar/point_csv/pointpil.csv"
# csv_path="/home/mayanksati/PycharmProjects/Data_Science_new/Python/python_code/combined.csv"
# root="/home/mayanksati/Documents/Rosbag_files/short_range_images/gwm_data_train.csv"
# root="/home/mayanksati/Documents/datasets/BDD/ubuntu_aws_instance/Mayank_datastore/rosbag_images/2018-11-24-09-48-37/Baidu_TL_dataset1"
root = "/media/mayank_sati/DATA/datasets/one_shot_datasets/Farmington/images/2019-09-27-14-39-41_train"
# csv_path="/home/mayanksati/Documents/Rosbag_files/short_range_images/gwm_data_train.csv"
csv_path = "/media/mayank_sati/DATA/datasets/one_shot_datasets/Farmington/images/for_farming_single_light/2019-09-27-14-39-41_train.csv"
# csv_path=('/home/mayanksati/Documents/csv/traffic_light_square.csv')
# csv_path=('/home/mayanksati/PycharmProjects/Data_Science_new/Deep learning/Load Datasets/traffic_light.csv')
# root='/home/mayanksati/PycharmProjects/models/tensorflow/re3-tensorflow-master/demo/GWM_dataset'
dataset = pd.read_csv(csv_path)
print(dataset.head())
# dataset=dataset.iloc[np.random.permutation(len(dataset))]
x = dataset.iloc[:, 0].values
y = dataset.iloc[:, 5:9].values

# for loop,xml_aray in zip(x, y):

z = dataset.iloc[:, 3].values
for val,loop in enumerate(x):
    print(loop)
    image_path = os.path.join(root, loop)
    image_scale = cv2.imread(image_path, 1)
    new_box = increase_bounding_box_scale_diff_apr(image_scale, y[val], scale_width=8, scale_height=2.5)
    # [Xmin,ymin,xmax,ymax]
    # top = (y[val, 0], y[val, 3])
    # bottom = (y[val, 2], y[val, 1])
    top = (new_box[0], new_box[3])
    bottom = (new_box[2], new_box[1])

    height=abs(new_box[3]-new_box[1])
    width = abs((new_box[2]- new_box[0]))

    print(width ," X ", height)
    # image_path=os.path.join(root,loop+".jpg")

    # print(my_image[0,0])
    # image_scale=cv2.resize(m_image,dsize=(800,600), interpolation=cv2.INTER_NEAREST)
    # cv2.rectangle(image_scale, pt1=(125,84),pt2=(159,38),color= (0,255,0), thickness=2)
    cv2.rectangle(image_scale, pt1=top, pt2=bottom, color=(0, 255, 0), thickness=2)
    cv2.circle(image_scale, center=(452, 274), radius=10, color=(0, 255, 0))
    # cv2.putText(image_scale, z[val], (20, 20), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), lineType=cv2.LINE_AA)
    # cv2.imshow('streched_image', image_scale)
    # output_folder_path="/home/mayank-s/PycharmProjects/Datasets/aptive/object_detect/output/output_merccedes.png"
    # filepath=output_folder_path+my_image+".png"
    # cv2.imwrite(filepath,my_image)
    # ch = cv2.waitKey(1)  # refresh after 1 milisecong
    # if ch & 0XFF == ord('q'):
    #     cv2.destroyAllWindows()
    # cv2.destroyAllWindows()
    # val += 1
    # print(val)


    # new_box=increase_bounding_box_scale_diff_apr(image_scale,y[val],scale_width=2.5,scale_height=2.5)'''
