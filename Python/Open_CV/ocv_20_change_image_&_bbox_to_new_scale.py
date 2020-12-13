import cv2


def bbox_size_change(data_label, new_hight, new_width):
    width = data_label[5]
    height = data_label[4]

    xmin = data_label[0]
    ymin = data_label[1]
    xmax = data_label[2]
    ymax = data_label[3]

    h = ymax - ymin
    w = xmax - xmin
    # which cell this obj falls into
    centerx = (xmax + xmin) / 2.0
    centery = (ymax + ymin) / 2.0
    # 448 is size of the input image
    newx = (new_width / width) * centerx
    newy = (new_hight / height) * centery

    h_new = h * (new_hight / height)
    w_new = w * (new_width / width)

    new_xmin = int(newx - (w_new / 2))
    new_xmax = int(newx + (w_new / 2))
    new_ymin = int(newy - (h_new / 2))
    new_ymax = int(newy + (h_new / 2))

    return [new_xmin, new_ymin, new_xmax, new_ymax]


xmin, ymin, xmax, ymax = 1156, 136, 1191, 210
top1 = (xmin, ymax)
bottom1 = (xmax, ymin)

# this is how to draw bounding box
'''x1,y1 ------
    |          |
    |          |
    |          |
    --------x2,y2'''
###########################################################3
# this is for normal image
my_image = cv2.imread('ocv_20_test.jpg', 1)
cv2.rectangle(my_image, pt1=top1, pt2=bottom1, color=(0, 255, 0), thickness=2)
cv2.imshow('Normal image', my_image)
cv2.waitKey(5000)
cv2.destroyAllWindows()

####################################################################333
# Let change for the rescale image

image_scale = cv2.resize(my_image, dsize=(300, 500), interpolation=cv2.INTER_NEAREST)
height = my_image.shape[0]
width = my_image.shape[1]

data_label = [xmin, ymin, xmax, ymax, height, width]

dat_return = bbox_size_change(data_label, new_hight=300, new_width=500)
nxmin, nymin, nxmax, nymax = dat_return
top = (nxmin, nymax)
bottom = (nxmax, nymin)

cv2.rectangle(image_scale, pt1=top, pt2=bottom, color=(0, 0, 255), thickness=2)
cv2.imshow('rescale image', image_scale)
cv2.waitKey(5000)
cv2.destroyAllWindows()
