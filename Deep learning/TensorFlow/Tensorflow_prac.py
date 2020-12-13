import tensorflow as tf
from matplotlib import pyplot as plt

filename = tf.placeholder(tf.string, name="inputFile")
fileContent = tf.read_file(filename, name="loadFile")
image = tf.image.decode_jpeg(fileContent, name="decodeJpeg")
resize_bilinear = tf.image.resize_images(image, size=[256, 256], method=tf.image.ResizeMethod.BILINEAR)
resize_bicubic = tf.image.resize_images(image, size=[256, 256], method=tf.image.ResizeMethod.BICUBIC)
resize_nearest_neighbor = tf.image.resize_images(image, size=[256, 256], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
resize_area = tf.image.resize_images(image, size=[256, 256], method=tf.image.ResizeMethod.AREA)
resize_image_with_crop_or_pad = tf.image.resize_image_with_crop_or_pad(image, target_height=300, target_width=300)

sess = tf.Session()
# feed_dict = {filename: "/home/mayanksati/PycharmProjects/Data_Science_new/Task/object_detector_2/training/images_all/3.jpg"}
feed_dict = {
    filename: "/home/mayanksati/PycharmProjects/Data_Science_new/Task/object_detector_2/training/save_cropped_images_300/crop_img_70.jpg"}
with sess.as_default():
    actualImage = image.eval(feed_dict)
    plt.imshow(actualImage)
    plt.title("original image")
    plt.show()

    actual_resize_image_with_crop_or_pad = resize_image_with_crop_or_pad.eval(feed_dict)
    plt.imshow(actual_resize_image_with_crop_or_pad)
    plt.title("resize_image_with_crop_or_pad")
    plt.show()

    ## Resized images will be distorted if their original aspect ratio is not the same as size.
    actual_resize_bilinear = resize_bilinear.eval(feed_dict)
    plt.imshow(actual_resize_bilinear)
    plt.title("biliner resize")
    plt.show()

    actual_resize_nearest_neighbor = resize_nearest_neighbor.eval(feed_dict)
    plt.imshow(actual_resize_nearest_neighbor)
    plt.title("nearest neighbor resize")
    plt.show()

    ## Resized images will be distorted if their original aspect ratio is not the same as size.
    actual_resize_area = resize_area.eval(feed_dict)
    plt.imshow(actual_resize_area)
    plt.title("area resize")
    plt.show()
