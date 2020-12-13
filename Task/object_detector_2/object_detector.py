
import argparse

# General libs
import numpy as np
import os
import sys
import tensorflow as tf
import cv2
# import PIL
import time
import time


class tensorflow_obj():
    def __init__(self):
        # ## Initial msg
        # rospy.loginfo('  ## Starting ROS Tensorflow interface ##')
        # ## Load a (frozen) Tensorflow model into memory.
        detection_graph = tf.Graph()
        with detection_graph.as_default():
            od_graph_def = tf.GraphDef()
            with tf.gfile.GFile("/home/mayank_sati/codebase/docker_ob/output/frozen_inference_graph.pb", 'rb') as fid:
            # with tf.gfile.GFile("./object_detection/mayank_pb/frozen_inference_graph_new.pb", 'rb') as fid:
                serialized_graph = fid.read()
                od_graph_def.ParseFromString(serialized_graph)
                tf.import_graph_def(od_graph_def, name='')

        # ## Loading label map
        # Label maps map indices to category names, so that when our convolution network predicts `5`, we know that this corresponds to `airplane`.  Here we use internal utility functions, but anything 			that returns a dictionary mapping integers to appropriate string labels would be fine
        # self.category_index = label_map_util.create_category_index_from_labelmap("./object_detection/Baidu_models/Baidu_ssd_model_0/haval_label_map.pbtxt", use_display_name=True)
        # self.category_index = label_map_util.create_category_index_from_labelmap("./object_detection/mayank_pb/bdd_traffic_label_map_simgle.pbtxt", use_display_name=True)
        # ## Get Tensors to run from model
        self.image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
        # Each box represents a part of the image where a particular object was detected.
        self.boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
        # Each score represent how level of confidence for each of the objects.
        # Score is shown on the result image, together with the class label.
        self.scores = detection_graph.get_tensor_by_name('detection_scores:0')
        self.classes = detection_graph.get_tensor_by_name('detection_classes:0')
        self.num_detections = detection_graph.get_tensor_by_name('num_detections:0')

        # # Tensorflow Session opening: Creates a session with log_device_placement set to True.
        # ## Session configuration
        config=tf.ConfigProto(log_device_placement=True)
        config.log_device_placement = True
        config.gpu_options.allow_growth = True

        # ## Session openning
        try:
            with detection_graph.as_default():
                self.sess = tf.Session(graph=detection_graph, config = config)
                # rospy.loginfo('  ## Tensorflow session open: Starting inference... ##')
        except ValueError:
            print(1)
        global graph
        graph = tf.get_default_graph()

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

        for root, _, filenames in os.walk(input_folder):
            if (len(filenames) == 0):
                print("Input folder is empty")
                return 1
            time_start = time.time()
            for filename in filenames:
                # try:
                    print("Creating object detection for file : {fn}".format(fn=filename), '\n')

                    file_path = (os.path.join(root, filename))
                    image = cv2.imread(file_path, 1)
                    image_expanded = np.expand_dims(image, axis=0)

                    #######################################3333
                    rows, cols, channels = image.shape
                    image_np_expanded = np.expand_dims(image, axis=0)
                    # second = time.time()
                    ################################################

                    # Perform the actual detection by running the model with the image as input
                    # (boxes, scores, classes, num) = self.sess.run([self.detection_boxes, self.detection_scores, self.detection_classes, self.num_detections], feed_dict={self.image_tensor: image_expanded})
                    # print (self.num_detections,'\n')
                    # print(scores[scores > .2])
                    (boxes_out, scores_out, classes_out, num_detections_out) = self.sess.run([self.boxes, self.scores, self.classes, self.num_detections], feed_dict={self.image_tensor: image_np_expanded})
                    numboxes = np.squeeze(num_detections_out)
                    print(scores_out[scores_out > .2])
                    # print("num_detection are",num_detections_out)
                    # numboxes = 300

                    # print("confidence score are",scores_out[:, 0:8])
                    # tl_bbox = Float32MultiArray()
                    # if int(numboxes) >= 1:
                    #     tmp = -np.ones(7 * int(numboxes) + 1)

                    for i in range(0, int(numboxes)):
                        score = float(np.squeeze(scores_out)[i])
                        bbox = [float(v) for v in np.squeeze(boxes_out)[i]]
                        if score > 0.1:
                            x_top_left = bbox[1] * cols
                            y_top_left = bbox[0] * rows
                            x_bottom_right = bbox[3] * cols
                            y_bottom_right = bbox[2] * rows
                            width = x_bottom_right - x_top_left
                            height = y_bottom_right - y_top_left

                            # tmp[7] = .999

            # time_end = time.time()
            # print("Object Detection on above images completed successfully !", '\n')
            # # sec = timedelta(seconds=int(time_end - time_start))
            # # d = datetime(1, 1, 1) + sec
            # print("Time Consumed - hours:{th} - Minutes:{mn} - Second:{sc}".format(th=d.hour, mn=d.minute,
            #                                                                        sc=d.second))
            # print('\n', "Path for output annotated images :", output_folder)
            # # print("Object Detected successfully !", '\n')


def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser(description='Object detection on imgage started..')
    # parser.add_argument('--input_path', help="Input Folder")
    # parser.add_argument('--output_path', help="Output folder")
    # parser.add_argument('--input_path', help="Input Folder", default='/home/mayanksati/PycharmProjects/Data_Science_new/Task/object_detector_2/training/input')
    parser.add_argument('--input_path', help="Input Folder",
                        default='/media/mayank_sati/DATA/datasets/traffic_light/BDD/bdd100k/images/10k/test')
    parser.add_argument('--output_path', help="Output folder",
                        default='/home/mayank_sati/Desktop/cool')

    args = parser.parse_args()
    return args



def main():
    MODEL_NAME = '/home/mayank_sati/codebase/docker_ob/output'
    PATH_TO_LABELS = "training/bdd_traffic_label_map_simgle.pbtxt"
    NUM_CLASSES = 1

    args = parse_args()
    print('\n', "Starting  Objects Detection on Image...", "\n")
    print('Reading images from path :', args.input_path)
    model = tensorflow_obj()
    ret = model.find_detection(args.input_path, args.output_path)
    if ret == 1:
        print("\n", "File Error.....")



if __name__ == '__main__':
    main()
