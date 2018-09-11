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
from PIL import Image
import h5py
import io
import tempfile
sys.path.append("..")
# Import utilites
from utils import label_map_util
from utils import visualization_utils as vis_util


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



    def find_detection(self,input_file,output_folder=None):
        """Function to detect images in input folder and save annotated image in an output directory.

            Args:
                input_folder        : Input  file directory.

                output_folder       : Output directory to save the image with object detected.

            Returns:
                None"""
        # if not os.path.exists(input_folder):
        #     print("Input folder not found")
        #     return 1

        # if not os.path.exists(output_folder):
        #     print("Output folder not present. Creating New folder...")
        #     os.makedirs(output_folder)

        # for root,_, filenames in os.walk(input_folder):
        #     if (len(filenames) == 0):
        #         print("Input folder is empty")
        #         return 1
        image_return=self.read_hdf5_file(input_file)
        # ___________________________________________________#
        # hdf5_output_path=os.path.join(output_folder,'output.hdf5')
        hdf5_output_path=input_file
        h5file = h5py.File(hdf5_output_path, 'r+')
        uint8_dt = h5py.special_dtype(vlen=np.dtype('uint8'))  # variable length uint8
        # vlen_int_dt = h5py.special_dtype(vlen=np.dtype(int))
        # test_group = h5file.create_group('test')
        test_group=h5file.get('test')
        # cool=test_group.items()
        # new=list(cool)
        # if 'output_images' in new:
        #     print ("hello")

        # del [test_group['output_images']
        # test_group['output_images'][:] = 0
        # test_images = test_group.create_dataset(name='images', shape=(np.shape(image_return)[0],), dtype=uint8_dt)
        test_images = test_group.require_dataset(name='output_images', shape=(len(image_return),), dtype=uint8_dt)
        ######################################################
        # ##
        time_start = time.time()
        dirpath = tempfile.mkdtemp()
        temp_file = dirpath +"tmp.jpg"
        for val, image_val in enumerate(image_return):
                #try:
                    print("Creating object detection for file ", '\n')
                    # print(image_return[0])
                    # image_val= image_return[len]
                    img = Image.open(io.BytesIO(image_val))
                    file_name = "file_" + str(val) + '.jpg'
                    # image_path = os.path.join(output_path, file_name)
                    # img.save(image_path, 'JPEG')

                    # ... do stuff with dirpath
                    # shutil.rmtree(dirpath)

                    img.save(temp_file)
                    # img.save("temp.jpg")

                    # # ______________________________________
                    #
                    # file_path = (os.path.join(root, filename))
                    image = cv2.imread(temp_file, 1)
                    image_expanded = np.expand_dims(image, axis=0)


                    # Perform the actual detection by running the model with the image as input
                    (boxes, scores, classes, num) = self.sess.run(
                        [self.detection_boxes, self.detection_scores, self.detection_classes, self.num_detections],
                        feed_dict={self.image_tensor: image_expanded})
                    # print (self.num_detections,'\n')

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
                    # output_path = (os.path.join(output_folder, file_name))
                    # All the results have been drawn on image. Now display the image.
                    # cv2.imshow('Object detector', image)
                    output_path = (os.path.join(output_folder, file_name))
                    cv2.imwrite(output_path, image)

                    test_images[val] = self.get_image_for_id(output_path)
                    # Press any key to close the image
                    # cv2.waitKey(1000)
                    # Clean up
                    cv2.destroyAllWindows()
                    # vis_util.save_image_array_as_png(image, output_folder)
                # except IOError:
                #     print("Existing Object Detection...")
                # except:
                #     1
                #     # print('ERROR...object detection failed for filename: {fn}, Check file type... '.format(fn=filename),'\n')
                # else:
                #     1
                #     #print("Object Detected  successfully !", '\n')

        time_end = time.time()
        print("Object Detection on above images completed successfully !", '\n')
        sec = timedelta(seconds=int(time_end - time_start))
        d = datetime(1, 1, 1) + sec
        print("Time Consumed - hours:{th} - Minutes:{mn} - Second:{sc}".format(th=d.hour, mn=d.minute,
                                                                                   sc=d.second))
        print('\n',"Path for output annotated images :", output_folder)
            # print("Object Detected successfully !", '\n')

    def read_hdf5_file(self,hdf5_path):
        Read_h5 = h5py.File(hdf5_path, 'r')
        image_data = np.array(Read_h5.get('test').get('input_images'))
        Read_h5.close()
        return image_data

    def get_image_for_id(self,fname):
        """Get image data as uint8 array for given image.
        Parameters
        ----------
        path : path of image directory
        -------
        image_data : array of uint8
            Compressed JPEG byte string represented as array of uint8.
        """
        # fname = os.path.join(path, image_id)
        with open(fname, 'rb') as in_file:
            data = in_file.read()
        # Use of encoding based on: https://github.com/h5py/h5py/issues/745
        return np.fromstring(data, dtype='uint8')



def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser(description='Object detection on imgage started..')
    # parser.add_argument('--input_path', help="Input Folder")
    # parser.add_argument('--output_path', help="Output folder")
    parser.add_argument('--input_path', help="Input Folder",
                        default='/home/mayank-s/Desktop/Link to Datasets/aptive/object_detect/Image_to_hdf51.hdf5')
    parser.add_argument('--output_path', help="Output folder",
                        default='/home/mayank-s/PycharmProjects/Datasets/aptive/object_detect/output')

    args = parser.parse_args()
    return args


MODEL_NAME = 'ssd_mobilenet_v2'
PATH_TO_LABELS="labelmap.pbtxt"
NUM_CLASSES=9

args=parse_args()
print('\n',"Starting  Objects Detection on Image...","\n")
print('Reading images from path :', args.input_path)
model = Object_detect(MODEL_NAME,PATH_TO_LABELS,NUM_CLASSES)
model.Laod_model_parameter()
ret = model.find_detection(args.input_path,args.output_path)
if ret==1:
    print("\n","File Error.....")
