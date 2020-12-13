######################################################################################
# A script to run object detection model
#
# Run command : python3 object_detector.py "input_path" "output_path"
# example:
#
# # run command: python3 aptiv_object_detector.py --input_path "/home/object_detector/input" --output_path "/home/object_detector/output"
#
# @author Mayank Sati/Ashis Samal
#
# library used:

import argparse
import os
import sys
import time

# 1.Protobuf 3.0.0
# 2.Python - tk
# 3.Pillow 1.0
# 4.lxml
# 5.Matplotlib
# 6.Tensorflow CPU/GPU - 1.8.0
# 7.Cython
# 8.contextlib2
# 9.0pencv-python 3.4.1.15
######################################################################################
import cv2
import numpy as np
import tensorflow as tf

sys.path.append("..")
# Import utilites
from utils import label_map_util
# from utils import visualization_utils as vis_util
import natsort


class Object_detect():

    def __init__(self, MODEL_NAME, label_path, NUM_CLASSES):
        self.MODEL_NAME = MODEL_NAME
        self.label_path = label_path
        self.NUM_CLASSES = NUM_CLASSES

    def Laod_model_parameter(self):
        CWD_PATH = os.getcwd()
        # Path to frozen detection graph .pb file, which contains the model that is usedfor object detection.
        PATH_TO_CKPT = os.path.join(CWD_PATH, self.MODEL_NAME, 'frozen_inference_graph_new.pb')
        # PATH_TO_CKPT = os.path.join(CWD_PATH, self.MODEL_NAME, 'frozen_inference_graph_14000.pb')
        # PATH_TO_CKPT = os.path.join(CWD_PATH, self.MODEL_NAME, 'frozen_inference_graph.pb')
        # PATH_TO_CKPT = os.path.join(CWD_PATH, self.MODEL_NAME, 'frozen_inference_graph_75000.pb')
        # / home / ubuntu / PycharmProjects / ob_2 / training / ssd_resnet_50_fpn / retiananet / frozen_inference_graph_75000.pb
        # PATH_TO_CKPT = "/mnt/datastore/groups/ai/Mayank_datastore/frozen_inference_graph_20000.pb"
        # Path to label map file
        # PATH_TO_LABELS="/home/ubuntu/Documents/data/bdd_traffic_label_map_single.pbtxt"
        PATH_TO_LABELS = os.path.join(CWD_PATH, self.label_path)
        # dictionary mapping integers to appropriate string labels would be fine
        label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
        categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=self.NUM_CLASSES,
                                                                    use_display_name=True)
        self.category_index = label_map_util.create_category_index(categories)

        # / home / ubuntu / PycharmProjects / ob_2 / training / bdd_traffic_label_map_simgle.pbtxt

        # Load the Tensorflow model into memory.
        detection_graph = tf.Graph()
        with detection_graph.as_default():
            od_graph_def = tf.GraphDef()
            with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
                serialized_graph = fid.read()
                od_graph_def.ParseFromString(serialized_graph)
                tf.import_graph_def(od_graph_def, name='')
            self.sess = tf.Session(graph=detection_graph)

        # Define input and output tensors (i.e. data) for the object detection classifier
        # Input tensor is the imagessd_mobilenet_v2
        self.image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
        # Output tensors are the detection boxes, scores, and classes
        # Each box represents a part of the image where a particular object was detected
        self.detection_boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
        # Each score represents level of confidence for each of the objects.
        # The score is shown on the result image, together with the class label.
        self.detection_scores = detection_graph.get_tensor_by_name('detection_scores:0')
        self.detection_classes = detection_graph.get_tensor_by_name('detection_classes:0')
        # Number of objects detected
        self.num_detections = detection_graph.get_tensor_by_name('num_detections:0')
        # './training/color_model/model_b_new_12_04.h'
        self.mymodel = tf.keras.models.load_model('./training/color_model/model_b_new_12_04.h')
        print(1)

    def find_detection(self, input_folder, output_folder):
        """Function to detect images in input folder and save annotated image in an output directory.

            Args:
                input_folder        : Input  file directory.

                output_folder       : Output directory to save the image with object detected.

            Returns:
                None"""
        if not os.path.exists(input_folder):
            print("Input folder not found")
            return 1

        if not os.path.exists(output_folder):
            print("Output folder not present. Creating New folder...")
            os.makedirs(output_folder)
        # traffic_light = ["red", "green", "black"]
        traffic_light = ["black", "red", "yellow", "green"]
        # mymodel = tf.keras.models.load_model(
        #     "/home/mayank_sati/Documents/ob_apollo_build/AI-tl_detector/src/augmented_tl_detector/src/saved_models/new_model/model_b_lr-4_ep150_ba32.h")
        # test_image=image

        for root, _, filenames in os.walk(input_folder):
            if (len(filenames) == 0):
                print("Input folder is empty")
                return 1
            filenames = natsort.natsorted(filenames, reverse=False)

            # time_start = time.time()
            for filename in filenames:
                # try:
                print("\nCreating object detection for file : {fn}".format(fn=filename))

                file_path = (os.path.join(root, filename))
                image = cv2.imread(file_path, 1)
                image_expanded = np.expand_dims(image, axis=0)
                res = (str(image.shape[1]) + "*" + str(image.shape[0]))
                time_start = time.time()
                # Perform the actual detection by running the model with the image as input
                (boxes, scores, classes, num) = self.sess.run(
                    [self.detection_boxes, self.detection_scores, self.detection_classes, self.num_detections],
                    feed_dict={self.image_tensor: image_expanded})
                #################################################################################3333
                # our code
                rows, cols, channels = image.shape
                numboxes = 1
                if int(numboxes) >= 1:
                    for i in range(0, int(numboxes)):
                        score = float(np.squeeze(scores)[i])
                        bbox = [float(v) for v in np.squeeze(boxes)[i]]
                        if score > 0.3:
                            x_top_left = bbox[1] * cols
                            y_top_left = bbox[0] * rows
                            x_bottom_right = bbox[3] * cols
                            y_bottom_right = bbox[2] * rows
                            width = x_bottom_right - x_top_left
                            height = y_bottom_right - y_top_left
                            #########################################################333

                            traffic_light = ["black", "red", "yellow", "green"]
                            frame = image[int(y_top_left):int(y_bottom_right), int(x_top_left):int(x_bottom_right)]
                            test_image = cv2.resize(frame, (32, 32))

                            test_image = tf.keras.preprocessing.image.img_to_array(test_image)
                            test_image = np.expand_dims(test_image, axis=0)
                            test_image /= 255
                            result = self.mymodel.predict(test_image)
                            # print("My model is:" ,mymodel.summary())
                            light_color = traffic_light[result.argmax()]
                            print('color of light was', light_color)
                            #
                            # light_color = light_color+"\n"
                            # with open('somefile4.txt', 'a') as the_file:
                            #     the_file.write(light_color)
                            # # '_______________________________________'

                            # cv2.imshow('images', frame)
                            # cv2.waitKey(1)
                            # cv2.destroyAllWindows()
                            # # cv2.waitKey(10)
                            top = (int(x_top_left), int(y_top_left))
                            bottom = (int(x_bottom_right), int(y_bottom_right))
                            cv2.rectangle(image, pt1=top, pt2=bottom, color=(0, 255, 0), thickness=2)
                            # cv2.putText(image_np, light_color, (250, 90), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0),lineType=cv2.LINE_AA)
                            ################################################################

                            time_end = time.time()
                            print("time consumed", (time_end - time_start) * 1000)
                            # print("time consumed", (time_end - time_start) * 1000)
                            output_path = (os.path.join(output_folder, filename))

                            cv2.putText(image, str(round((time_end - time_start) * 1000)), (50, 50),
                                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), lineType=cv2.LINE_AA)
                            cv2.putText(image, res, (50, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0),
                                        lineType=cv2.LINE_AA)
                            cv2.putText(image, light_color, (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0),
                                        lineType=cv2.LINE_AA)

                            # Clean up



                else:
                    1
                cv2.imwrite(output_path, image)
                cv2.destroyAllWindows()
                # vis_util.save_image_array_as_png(image, output_folder)
            # except IOError:
            #     print("Existing Object Detection...")
            # except:
            #     print('ERROR...object detection failed for filename: {fn}, Check file type... '.format(fn=filename),'\n')
            # else:
            #     1
            #     print("Object Detected  successfully !", '\n')
            #

            print('\n', "Path for output annotated images :", output_folder)
            # print("Object Detected successfully !", '\n')


def parse_args():
    # file_path="/home/mayanksati/PycharmProjects/Data_Science_new/Task/object_detector_2/training/images_all"

    # file_path="/home/ubuntu/Documents/traffic_light/save_cropped_images_300"
    # file_path="/home/mayank_sati/pycharm_projects/Data_Science_new/Task/object_detector_2/training/save_cropped_images_300"
    # file_path="/home/mayank_sati/Documents/datsets/Rosbag_files/short_range_images/2018-11-24-09-44-06/Baidu_TL_dataset1"
    # file_path="/home/mayank_sati/Documents/datsets/Rosbag_files/short_range_images/2018-11-24-09-25-36_copy/GWM_dataset"
    file_path = "/home/mayank_sati/Documents/datsets/Rosbag_files/square data/square_data0-1/GWM_dataset"
    # file_path="/mnt/datastore/groups/ai/Mayank_datastore/crop_r_313"
    """Parse input arguments."""
    parser = argparse.ArgumentParser(description='Object detection on imgage started..')
    # parser.add_argument('--input_path', help="Input Folder")
    # parser.add_argument('--output_path', help="Output folder")
    # parser.add_argument('--input_path', help="Input Folder",default='/home/mayanksati/PycharmProjects/Data_Science_new/Task/object_detector_2/training/traffic_multiple_withoutANnotation/MKZ056_436_1543022148_1543022246')
    parser.add_argument('--input_path', help="Input Folder", default=file_path)
    # parser.add_argument('--input_path', help="Input Folder",default='/home/mayanksati/PycharmProjects/Data_Science_new/Task/object_detector_2/training/images_all')
    parser.add_argument('--output_path', help="Output folder",
                        default='/home/mayank_sati/pycharm_projects/Data_Science_new/Task/object_detector_2/training/output')

    args = parser.parse_args()
    return args


# MODEL_NAME = 'training/ssd_resnet_50_fpn/faster_rcnn_gwm'
# MODEL_NAME = '/home/mayank_sati/pycharm_projects/Data_Science_new/Task/object_detector_2/training/ssd_resnet_50_fpn/faster_rcnn_gwm/fixed_sized_500'
# MODEL_NAME = '/home/mayank_sati/pycharm_projects/Data_Science_new/Task/object_detector_2/training/ssd_resnet_50_fpn/faster_rcnn_gwm/multicolor_square_traffic'
MODEL_NAME = 'training/ssd_resnet_50_fpn/faster_rcnn_gwm/fixed_sized_500'
PATH_TO_LABELS = "training/bdd_traffic_label_map_simgle.pbtxt"
# PATH_TO_LABELS="training/multicolor_square.pbtxt"
NUM_CLASSES = 1

args = parse_args()
print('\n', "Starting  Objects Detection on Image...", "\n")
print('Reading images from path :', args.input_path)
model = Object_detect(MODEL_NAME, PATH_TO_LABELS, NUM_CLASSES)
model.Laod_model_parameter()
ret = model.find_detection(args.input_path, args.output_path)
if ret == 1:
    print("\n", "File Error.....")
