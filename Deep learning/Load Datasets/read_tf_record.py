import cv2
import numpy as np
import tensorflow as tf
from PIL import Image

# train_record = '/home/deepaknayak/Documents/TF-records/Apolloscape/scene-parsing/instance-level_pixel-level/data/sample-tf/trains.record'
train_record = '/home/mayank_sati/Documents/datsets/Rosbag_files/short_range_images/gwm_sq_traffic_light_train_all_image_filter_final.record'


def read_and_decode(filename_queue):
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)
    features = tf.parse_single_example(
        serialized_example,
        features={
            'image/height': tf.FixedLenFeature([], tf.int64),
            'image/width': tf.FixedLenFeature([], tf.int64),
            'image/filename': tf.FixedLenFeature([], tf.string),
            'image/source_id': tf.FixedLenFeature([], tf.string),
            'image/encoded': tf.FixedLenFeature([], tf.string),
            'image/format': tf.FixedLenFeature([], tf.string),
            'image/object/bbox/xmin': tf.VarLenFeature(tf.float32),
            'image/object/bbox/xmax': tf.VarLenFeature(tf.float32),
            'image/object/bbox/ymin': tf.VarLenFeature(tf.float32),
            'image/object/bbox/ymax': tf.VarLenFeature(tf.float32),
            'image/object/class/text': tf.VarLenFeature(tf.string),
            'image/object/class/label': tf.VarLenFeature(tf.int64),
        })
    image = tf.image.decode_png(features['image/encoded'])
    label = tf.cast(features['image/object/class/label'], tf.int32)
    height = tf.cast(features['image/height'], tf.int32)
    width = tf.cast(features['image/width'], tf.int32)
    xmin = tf.cast(features['image/object/bbox/xmin'], tf.float32)
    xmax = tf.cast(features['image/object/bbox/xmax'], tf.float32)
    ymin = tf.cast(features['image/object/bbox/ymin'], tf.float32)
    ymax = tf.cast(features['image/object/bbox/ymax'], tf.float32)
    class_text = tf.cast(features['image/object/class/text'], tf.string)

    return image, height, width, label, xmin, xmax, ymin, ymax, class_text


def get_all_records(FILE):
    with tf.Session() as sess:
        filename_queue = tf.train.string_input_producer([FILE])
        image, height, width, label, xmin, xmax, ymin, ymax, class_text = read_and_decode(filename_queue)
        image = tf.reshape(image, tf.stack([height, width, 3]))
        # image.set_shape([600,400,3])
        init_op = tf.global_variables_initializer()
        sess.run(init_op)
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)
        d = 0
        for j in range(100000):
            example, l, h, w, xn, xx, yn, yx, ct = sess.run(
                [image, label, height, width, xmin, xmax, ymin, ymax, class_text])
            img = Image.fromarray(example)
            img = np.asarray(img, np.uint8)
            print("Train Image:", j + 1)
            print(l)
            if l.values.size > 0:
                for i in range(l.values.size):
                    if l.values[i] == 1:
                        labellight = 'Traffic_light'
                    # elif l.values[i]==2:
                    #     labellight='YELLOW'
                    # elif l.values[i]==3:
                    #     labellight='GREEN'
                    # elif l.values[i]==4:
                    #     labellight='GreenRight'
                    # elif l.values[i]==5:
                    #     labellight='RedLeft'
                    # elif l.values[i]==6:
                    #     labellight='RedRight'
                    # elif l.values[i]==7:
                    #     labellight='Yellow'
                    # elif l.values[i]==8:
                    #     labellight='off'
                    # elif l.values[i]==9:
                    #     labellight='RedStraight'
                    # elif l.values[i]==10:
                    #     labellight='GreenStraight'
                    # elif l.values[i]==11:
                    #     labellight='GreenStraightLeft'
                    # elif l.values[i]==12:
                    #     labellight='GreenStraightRight'
                    # elif l.values[i]==13:
                    #     labellight='RedStraightLeft'
                    # elif l.values[i]==14:
                    #     labellight='RedStraightRight'
                    xminimum = xn.values[i] * w
                    xmaximum = xx.values[i] * w
                    yminimum = yn.values[i] * h
                    ymaximum = yx.values[i] * h
                    # print("Label:",labellight)
                    # print("Xmin:",xminimum)
                    # print("Ymin:",yminimum)
                    # print("Xmax:",xmaximum)
                    # print("Ymax:",ymaximum)
                    cv2.rectangle(img, (int(xminimum), int(yminimum)), (int(xmaximum), int(ymaximum)), (255, 0, 0), 3)
                    cv2.putText(img, str(l.values[i]), (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 0, 0),
                                lineType=cv2.LINE_AA)

            print("Writing Image:", j + 1)
            filename = "images" + str(j) + ".jpg"
            print(filename)
            path = "/home/mayank_sati/Desktop/dump/" + filename
            print(j)
            cv2.imwrite(path, img)

        coord.request_stop()
        coord.join(threads)


get_all_records(train_record)
