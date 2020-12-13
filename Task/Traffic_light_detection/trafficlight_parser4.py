import glob
import os

import cv2
import numpy as np
# import ruamel.yaml as yaml
import yaml
from scipy.stats import kurtosis, skew
from skimage import io
from skimage.measure import label, regionprops

inputpath = './road04_seg/Label/'
inputpath1 = './road04_seg/ColorImage/'
outputpath = './trafficlight-data-2/'
file_counter = 0

for dirnames in os.listdir(inputpath):
    filepath = inputpath + dirnames + "/Camera 5/"
    for labelfile in glob.glob(filepath + "*_bin.png"):
        colorimagefile = labelfile.replace("Label", "ColorImage")
        colorimagefile = colorimagefile.replace('_bin.png', '.jpg')
        label_img = io.imread(labelfile)
        color_img = io.imread(colorimagefile)
        label_img_cv = cv2.imread(labelfile)
        color_img_cv = cv2.imread(colorimagefile)
        image_height = color_img.shape[0]
        image_width = color_img.shape[1]
        print(file_counter)
        # if (file_counter==1000):
        #     break
        if (np.any(label_img == 81)) == 1:
            label_img[label_img != 81] = 0
            label_img[label_img == 81] = 255
            label_image = label(label_img)
            list1 = []
            list2 = []
            ival = 0
            label_val = 0
            for region in regionprops(label_image):
                if region.area >= 4000:
                    miny, minx, maxy, maxx = region.bbox
                    print(ival)
                    ival = ival + 1

                    factor_w = 3
                    factor_h = 1
                    width = maxx - minx
                    height = maxy - miny
                    width_dash = width * factor_w
                    height_dash = height * factor_h
                    xnewmin = max(0, (minx - width_dash))
                    ynewmin = max(0, (miny - height_dash))
                    xnewmax = min(image_width, (maxx + width_dash))
                    ynewmax = min(image_height, (maxy + height_dash))
                    tmpwidth = xnewmax - xnewmin
                    tmpheight = ynewmax - ynewmin
                    color_img_crop = color_img[ynewmin:ynewmax, xnewmin:xnewmax]
                    # gray = cv2.cvtColor(color_img_crop, cv2.COLOR_BGR2GRAY)
                    # cv2.imshow("Input", gray)
                    # cv2.waitKey(0)
                    gray = cv2.cvtColor(color_img_crop, cv2.COLOR_BGR2GRAY)
                    oneDarray = gray.flatten()
                    kurtValue = kurtosis(oneDarray)
                    skewValue = skew(oneDarray)
                    eccentricity = region.eccentricity
                    if (eccentricity < 0.94 or kurtValue > 8 or skewValue > 2):
                        continue
                    if ((minx - xnewmin) < 10 or (miny - ynewmin) < 10 or (maxx - xnewmin) > (tmpwidth - 10) or (
                            maxy - ynewmin) > (tmpheight - 10)):
                        continue

                    # cv2.putText(color_img_crop, str(kurtValue), (50, 200), cv2.FONT_HERSHEY_SIMPLEX, 0.5,(255, 0, 255), 2)
                    # cv2.putText(color_img_crop, str(skewValue), (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5,(255, 0, 255), 2)

                    orientation = region.orientation * 180.0 / 3.1412
                    # eccentricity = region.eccentricity
                    # cv2.putText(color_img_crop,str(eccentricity),(50,200),cv2.FONT_HERSHEY_SIMPLEX,1,(255,0,255),2)
                    # cv2.imshow("Input",color_img_crop)
                    # cv2.waitKey(0)
                    # if(eccentricity >= 0.94):
                    image_name = colorimagefile.split("/")[5]
                    label_name = '_' + str(label_val) + '.jpg'
                    image_name = image_name.replace(".jpg", label_name)
                    path = outputpath + image_name
                    # mCrop = cv2.rectangle(color_img_crop, (minx-xnewmin, miny-ynewmin), (maxx-xnewmin, maxy-ynewmin), (255, 0, 255), 4)
                    # cv2.imwrite(path, mCrop)
                    io.imsave(path, color_img_crop)
                    label_val = label_val + 1
                    case = {"label": "Traffic_light", "occluded": False, 'xmin': minx - xnewmin, 'ymin': miny - ynewmin,
                            'xmax': maxx - xnewmin, 'ymax': maxy - ynewmin}
                    data3 = dict(boxes=case, path=path)
                    # list1.append(data3)
                    # data4 = dict(boxes=list1)
                    list2.append(data3)
                    print(list2)
                    with open('./roadseg_04_5_new1.yaml', 'a') as outfile:
                        yaml.dump(list2, outfile)
            # file_counter = file_counter + 1
