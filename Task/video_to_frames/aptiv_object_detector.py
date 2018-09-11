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
#library used:

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
import os
import argparse
import time
import numpy as np
import sys
from datetime import datetime, timedelta
import tensorflow as tf
sys.path.append("..")
# Import utilites
from utils import label_map_util
from utils import visualization_utils as vis_util

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

class Object_detect():

    def __init__(self,MODEL_NAME,label_path,NUM_CLASSES):
        self.MODEL_NAME=MODEL_NAME
        self.label_path=label_path
        self.NUM_CLASSES=NUM_CLASSES

    def Laod_model_parameter(self):
        CWD_PATH = os.getcwd()
        # Path to frozen detection graph .pb file, which contains the model that is usedfor object detection.
        PATH_TO_CKPT = os.path.join(CWD_PATH, self.MODEL_NAME, 'frozen_inference_graph.pb')
        # Path to label map file
        PATH_TO_LABELS = os.path.join(CWD_PATH,self.label_path)
        # dictionary mapping integers to appropriate string labels would be fine
        label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
        categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=self.NUM_CLASSES,
                                                                    use_display_name=True)
        self.category_index = label_map_util.create_category_index(categories)

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
        # Input tensor is the image
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



    def find_detection(self,input_folder,output_folder,maxframes="None"):
        """Function to detect images in input folder and save annotated image in an output directory.

            Args:
                input_folder        : Input  file directory.

                output_folder       : Output directory to save the image with object detected.

                maxframes           : Number of frames to capture if input is a video file

            Returns:
                None"""
        if not os.path.exists(input_folder):
            print("Input folder not found")
            return 1

        if not os.path.exists(output_folder):
            print("Output folder not present. Creating New folder...")
            os.makedirs(output_folder)

        for root,_, filenames in os.walk(input_folder):
            if (len(filenames) == 0):
                print("Input folder is empty")
                return 1
            time_start = time.time()
            for filename in filenames:
                try:
                    file_path = (os.path.join(root, filename))
                    cap = cv2.VideoCapture()
                    cap.open(file_path)
                    frameCount = cap.get(cv2.CAP_PROP_FRAME_COUNT)
                    if not cap.isOpened():
                        print("Failed to open input file : {fn}, check file type".format(fn=filename))
                        raise IOError
                    elif frameCount == 1:
                        print("Creating object detection on Image file : {fn}".format(fn=filename), '\n')
                        image = cv2.imread(file_path, 1)
                        image_expanded = np.expand_dims(image, axis=0)
                        # Perform the actual detection by running the model with the image as input
                        (boxes, scores, classes, num) = self.sess.run(
                            [self.detection_boxes, self.detection_scores, self.detection_classes, self.num_detections],
                            feed_dict={self.image_tensor: image_expanded})
                        # Draw the results of the detection (aka 'visulaize the results')
                        vis_util.visualize_boxes_and_labels_on_image_array(
                            image,
                            np.squeeze(boxes),
                            np.squeeze(classes).astype(np.int32),
                            np.squeeze(scores),
                            self.category_index,
                            use_normalized_coordinates=True,
                            line_thickness=5,
                            min_score_thresh=0.80)
                        output_path = (os.path.join(output_folder, filename))
                        cv2.imwrite(output_path, image)
                        # Clean up
                        cv2.destroyAllWindows()
                    else:
                        print("Creating object detection on Video file : {fn}".format(fn=filename))
                        frame_width = int(cap.get(3))
                        frame_height = int(cap.get(4))
                        output_path = (os.path.join(output_folder, filename))
                        out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), 10,
                                              (frame_width, frame_height))
                        print(
                            "TotalFrame : {tf} - Frame_width : {fw} - Frame Height : {fh} - Frame Rate(FPS) : {fp} ".format(
                                tf=frameCount, fw=cap.get(cv2.CAP_PROP_FRAME_WIDTH),
                                fh=cap.get(cv2.CAP_PROP_FRAME_HEIGHT),
                                fp=cap.get(cv2.CAP_PROP_FPS)))

                        frameId = 0
                        skipDelta = 0

                        if not maxframes == "None":
                            if frameCount > maxframes:
                                skipDelta = (frameCount / maxframes)
                                print("Maxframes is set to : {mf}".format(mf=maxframes))
                                print("Skip frames delta is : {d}".format(d=int(skipDelta)),'\n')
                            else:
                                print('Max frame {mf} cannot exceed total frame'.format(mf=maxframes),'\n')
                                # raise IOError


                        while frameId < frameCount:
                            ret, frame = cap.read()
                            if not ret:
                                print("Failed to read the frame {f}".format(f=frameId))
                                continue
                            # Acquire frame and expand frame dimensions to have shape: [1, None, None, 3]
                            frame_expanded = np.expand_dims(frame, axis=0)
                            # Perform the actual detection by running the model with the image as input
                            (boxes, scores, classes, num) = self.sess.run(
                                [self.detection_boxes, self.detection_scores, self.detection_classes,
                                 self.num_detections],
                                feed_dict={self.image_tensor: frame_expanded})
                            # Draw the results of the detection (aka 'visulaize the results')
                            vis_util.visualize_boxes_and_labels_on_image_array(
                                frame,
                                np.squeeze(boxes),
                                np.squeeze(classes).astype(np.int32),
                                np.squeeze(scores),
                                self.category_index,
                                use_normalized_coordinates=True,
                                line_thickness=8,
                                min_score_thresh=0.80)
                            # All the results have been drawn on the frame, so it's time to display it.
                            out.write(frame)
                            frameId += int(1 + skipDelta)
                            cap.set(cv2.CAP_PROP_POS_FRAMES, frameId)

                        # Clean up
                        out.release()
                        cap.release()
                        cv2.destroyAllWindows()

                    # vis_util.save_image_array_as_png(image, output_folder)
                except IOError:
                    print("Existing Object Detection...","\n")
                except:
                    print('ERROR...Object detection failed for filename: {fn}, Check file type... '.format(fn=filename),'\n')
                # else:
                #     print("Object Detected  successfully !", '\n')

            time_end = time.time()
            print('\n',"Object detection on above image/video files completed successfully !", '\n')
            sec = timedelta(seconds=int(time_end - time_start))
            d = datetime(1, 1, 1) + sec
            print("Time Consumed - hours:{th} - Minutes:{mn} - Second:{sc}".format(th=d.hour, mn=d.minute,
                                                                                   sc=d.second))
            print('\n',"Path for output annotated file :", output_folder)



def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser(description='Object detection on imgage started..')
    # parser.add_argument('--input_path', help="Input Folder")
    # parser.add_argument('--output_path', help="Output folder")
    # parser.add_argument('--maxframes',type=int,help="Output max number of frames")
    parser.add_argument('--input_path', help="Input Folder",
                        default='/home/mayank-s/PycharmProjects/Datasets/aptive/object_detect/input')
    parser.add_argument('--output_path', help="Output folder",
                        default='/home/mayank-s/PycharmProjects/Datasets/aptive/object_detect/output')
    parser.add_argument('--maxframes', help="Max_frames", default=5)
    args = parser.parse_args()
    return args


MODEL_NAME = 'ssd_mobilenet_v2'
PATH_TO_LABELS="labelmap.pbtxt"
NUM_CLASSES=9

args=parse_args()
print('\n',"Starting  Objects Detection...","\n")
print('Reading files from path :', args.input_path,'\n')
model = Object_detect(MODEL_NAME,PATH_TO_LABELS,NUM_CLASSES)
model.Laod_model_parameter()
if args.maxframes:
	ret = model.find_detection(args.input_path,args.output_path,args.maxframes)
else:
    ret=model.find_detection(args.input_path, args.output_path)
if ret==1:
    print("\n","File Error.....")
