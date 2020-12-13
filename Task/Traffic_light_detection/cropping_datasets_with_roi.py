import os

import cv2
import pandas as pd
from scipy.stats import kurtosis, skew


# def cropak(frame,lenght, hieght):
#     frame = frame[int(bbox_crop[1]):int(bbox_crop[1] + 500),
#                  int(bbox_crop[0]):int(bbox_crop[0] + 500)]  # forces to be square (keep y)
#     # save image
#     save_path = "./Baidu_TL_dataset1/baid_" + str(count) + ".jpg"
#     cv2.imwrite(save_path, frame)
#
# def crop(imPath,resize_width=256,resize_height=256,new_width=224,new_height=224):
#     im = Image.open(imPath)
#     im = im.resize((resize_width,resize_height),Image.ANTIALIAS)
#
#     #central crop 224,224
#     width, height = im.size   # Get dimensions
#
#     left = (width - new_width)/2
#     top = (height - new_height)/2
#     right = (width + new_width)/2
#     bottom = (height + new_height)/2
#
#     im = im.crop((left, top, right, bottom))
#     image_array = np.array(im)
#     image_array = np.rollaxis(image_array,2,0)
#     image_array = image_array/255.0
#     image_array = image_array * 2.0 - 1.0
#     return image_array

def crop(imPath, xmin, ymin, xmax, ymax, height_crop_ratio, width_crop_ratio):
    width = xmax - xmin
    height = ymax - ymin
    im = cv2.imread(imPath, 1)
    # im = Image.open(imPath)
    roi_xmin = xmin - ((width_crop_ratio * width) / 2)
    if roi_xmin <= 0:
        roi_xmin = 0
    roi_xmax = xmax + ((width_crop_ratio * width) / 2)
    if roi_xmax > 1280:
        roi_xmax = 1280
    roi_ymin = ymin - ((height_crop_ratio * height) / 2)
    if roi_ymin <= 0:
        roi_ymin = 0
    roi_ymax = ymax + ((height_crop_ratio * height) / 2)
    if roi_ymax > 720:
        roi_ymax = 720
    ###############################################333
    # if roi_xmin<=0:
    #     roi_xmin=0
    #     roi_max=roi_xmax+roi_xmin
    ###################################################
    roi_width = roi_xmax - roi_xmin
    roi_height = roi_ymax - roi_ymin

    # frame = im[int(roi_ymax):int(roi_ymax+roi_height), int(roi_xmax):int(roi_xmax+roi_width)]
    frame = im[int(roi_ymin):int(roi_ymin + roi_height), int(roi_xmin):int(roi_xmin + roi_width)]
    # cv2.imshow("Short", frame)
    # cv2.waitKey(200)
    return (frame, roi_xmin, roi_ymin, roi_xmax, roi_ymax)


# csv_path="/home/mayanksati/Documents/csv/BBD_val_traffic_light_signle.csv"
csv_path = "/home/mayanksati/Documents/csv/BBD_daytime_train_final.csv"
# csv_path="/home/mayanksati/PycharmProjects/Data_Science_new/Deep learning/Load Datasets/BBD_daytime_train.csv"
datasets = pd.read_csv(csv_path)
# df=data[(data['xmin']!=data['xmax']) & (data['ymin']!=data['ymax'])]
data = datasets.iloc[:].values
image_folder_path = "/home/mayanksati/Documents/datasets/BDD/bdd100k/images/100k/train"
left_limit = right_limit = 200
upper_limit = 360
size_limit = 30
save_path = "/home/mayanksati/Desktop/bdd_crop_roi/"
loop = 0
counter = 0
counter1 = 0
bblabel = []
for values in data:
    filename, image_width, image_height, object_name, xmin, ymin, xmax, ymax = values
    width_box = xmax - xmin
    height_box = ymax - ymin
    # print(values)
    counter += 1
    print(counter)
    # if (xmin > left_limit & xmax < (image_width-right_limit) & ymax<upper_limit & image_width >size_limit & image_height>size_limit):
    # if (xmin > left_limit and xmax < (image_width-right_limit and width> size_limit and height>size_limit)):
    # if (width_box > size_limit or height_box > size_limit,xmin > 200 or xmax < (image_width-right_limit)):# or xmin > left_limit):
    if (width_box < 30 or height_box < 30 or xmin < 100 or xmax > 1200):
        continue
    image_path = (os.path.join(image_folder_path, filename, ))
    counter1 += 1
    # print(counter1)
    try:
        my_frame, roi_xmin, roi_ymin, roi_xmax, roi_ymax = crop(image_path, xmin, ymin, xmax, ymax, height_crop_ratio=7,
                                                                width_crop_ratio=10)
        xmin_n = int(xmin - roi_xmin)
        ymin_n = int(ymin - roi_ymin)
        xmax_n = int(xmin_n + width_box)
        ymax_n = int(ymin_n + height_box)
        ########################################333
        top = (xmin_n, ymax_n)
        bottom = (xmax_n, ymin_n)
        # cv2.rectangle(my_frame, pt1=top, pt2=bottom, color=(0, 255, 0), thickness=2)
        # cv2.imshow('MyImage', my_frame)
        # cv2.waitKey(1)
        # ************************************************88
        gray = cv2.cvtColor(my_frame, cv2.COLOR_BGR2GRAY)
        oneDarray = gray.flatten()
        kurtValue = kurtosis(oneDarray)
        if kurtValue > 1:
            continue
        skewValue = skew(oneDarray)
        # cv2.putText(my_frame, str((kurtValue)), (50, 50), cv2.FONT_HERSHEY_SIMPLEX, .750, (0, 0, 255), lineType=cv2.LINE_AA)
        # cv2.putText(my_frame, str((skewValue)), (50, 100), cv2.FONT_HERSHEY_SIMPLEX, .750, (0, 0, 255), lineType=cv2.LINE_AA)
        # eccentricity = region.eccentricity
        ##########################################33
        # ymax_n=roi_ymax-ymax
        filename = filename + str(loop)
        output_path = save_path + filename + ".jpg"
        cv2.imwrite(output_path, my_frame)
        img_read = cv2.imread(output_path)
        im_width = img_read[1]
        im_height = img_read[0]
        data_label = [output_path, im_width, im_height, object_name, xmin_n, ymin_n, xmax_n, ymax_n]
        bblabel.append(data_label)
        print("the total images", loop)
        loop += 1
    except IOError:
        print("failed...")

print('total imaages are', loop)

columns = ['filename', 'width', 'height', 'class', 'xmin', 'ymin', 'xmax', 'ymax']

df = pd.DataFrame(bblabel, columns=columns)
df.to_csv('bbtrain_crop.csv')
# print('total imaages are',counter1)
# y = datasets.iloc[:, 4].values
