######################################################################################
# A script to run object detection model
#
# Run command : python3 object_detector.py "input_path" "output_path(optional)"
# Note:
# output path(optional): this is a optional parameter if not given then only creates output in hdf5 file
# # run command: python3 aptiv_object_detector.py --input_path "/home/object_detector/input" --output_path "/home/object_detector/output"
#
# @author Mayank Sati/Ashis Samal
#
# library used:


# 1.Pillow 1.0
# 2. Numpy -any latest version

######################################################################################

import argparse
import datetime
import os
import time

import h5py
import numpy as np
from PIL import Image

ts = time.time()


class Image_To_HDF5():
    def image_hdf5(self, input_path, output_path):

        if not os.path.exists(input_path):
            print("Input file not found")
            return 1

        if not os.path.exists(output_path):
            print("Output path not present")
            return 1

        st = datetime.datetime.fromtimestamp(ts).strftime('%d_%m_%Y_%H_%M_%S')
        Hdf5_file_name = "INPUT_" + st + ".hdf5"
        print('Creating HDF5 dataset structure...\n')
        fname = os.path.join(output_path, Hdf5_file_name)
        voc_h5file = h5py.File(fname, 'w')
        uint8_dt = h5py.special_dtype(vlen=np.dtype('uint8'))  # variable length uint8
        test_group = voc_h5file.create_group('test')
        start = 0
        for root, _, filenames in os.walk(input_path):
            if (len(filenames) == 0):
                print("Input folder is empty")
                return 1
            test_images = test_group.create_dataset('input_images', (3,), maxshape=((None),), dtype=uint8_dt)
            for filename in filenames:
                try:
                    print('Processing : {fn} into hdf5 file'.format(fn=filename))
                    test_id = Image.open(os.path.join(input_path, filename))
                    test_id.close()
                    image_data = self.get_image_for_id(input_path, filename)
                    test_images.resize((start + 1,))
                    test_images[start] = image_data
                    start += 1
                    print(start)
                except:
                    print('Error in image conversion for file name {fn}, jumping to next file'.format(fn=filename))

        print('Closing HDF5 file.')
        voc_h5file.close()
        print('HDF5 created sucessfully')

    def get_image_for_id(self, path, image_id):
        fname = os.path.join(path, image_id)
        with open(fname, 'rb') as in_file:
            data = in_file.read()
        return np.fromstring(data, dtype='uint8')


def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser(description='Object detection on imgage started..')
    # parser.add_argument('--input_path', help="Input Folder")
    # parser.add_argument('--output_path', help="Output folder")
    parser.add_argument('--input_path', help="Input Folder",
                        default='/home/mayank-s/PycharmProjects/Datasets/aptive/object_detect/input')
    parser.add_argument('--output_path', help="Output folder",
                        default='/home/mayank-s/PycharmProjects/Datasets/aptive/object_detect/j')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()
    print("\nStarting  Image to HDF5 file Creation... \n")
    print('Reading hdf5 from path : {pt} \n'.format(pt=args.input_path))
    model = Image_To_HDF5()
    ret = model.image_hdf5(args.input_path, args.output_path)
    # ret = model.find_detection(args.input_path)
    if ret == 1:
        print("\nFile Error.....")
