import argparse
import os
from PIL import Image
import io
import h5py
import numpy as np
from collections import namedtuple, OrderedDict
import pandas as pd
import cv2
from batchup import data_source
# data=hp.File('data1.hdf5','r')




class Hdf5_to_Image():

    def read_hdf5(self,fname,image_type,output_path):

        if not os.path.exists(fname):
            print("Input file not found")
            return 1
        if not os.path.exists(output_path):
            print("Output folder not present. Creating New folder...")
            os.makedirs(output_path)
 # l
        print('Reading HDF5 dataset structure.')
        try:
            file_data=h5py.File(fname,'r')
        except IOError as e:
            print("I/O error : Input file is currupt")
            return 1
        except:
            print("Unexpected error:")
        try:
            dataset=file_data.get('test')

            if image_type =="input":
                image_data=dataset.get('input_images')
            elif image_type =="output":
                image_data = dataset.get('output_images')
            else:
                print('Select either  Input or output datasets from hdf5 file')
                raise IOError

            group_image=np.array(image_data)

            if len(group_image.shape)==0:
                print("{it} datasets is not present in hdf5 file".format(it=image_type))
                raise IOError

            file_data.close()

            if len(group_image)==0:
                print ("HDF5 file does not contain any data")
                raise IOError


            for loop,image_val in enumerate (group_image):
                # print(image_return[0])
                # image_val= image_return[len]
                img = Image.open(io.BytesIO(image_val))
                file_name="file_"+str(loop) +'.jpeg'
                image_path=os.path.join(output_path, file_name)
                img.save(image_path, 'JPEG')

                # img.show()
            print('Processing file..')
            print("Closing hdf5 file")
        except IOError:
            print("Existing HDF5 to image conversion...")
        except:
            print('Error in image extraction, check content of hdf5 file')
        else:
            print('HDF5 to image converted sucessfully')



def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser(description='Object detection on imgage started..')
    # parser.add_argument('--input_path', help="Input Folder")
    # parser.add_argument('--output_path', help="Output folder")
    parser.add_argument('--hdf5_path', help="input hdf5_path",default='/home/mayank-s/PycharmProjects/Datasets/aptive/object_detect/INPUT_10_08_2018_12_51_38.hdf5')
    parser.add_argument('--image_type', help="input_output_images",
                        default='output')
    parser.add_argument('--output_path', help="Output folder",
                        default='/home/mayank-s/PycharmProjects/Datasets/aptive/object_detect/output')
    args = parser.parse_args()
    return args

if __name__ == '__main__':

    args=parse_args()
    # print('\n',"Extracting Images from HDF5 file...","\n")
    print('Reading hdf5 from path :', args.hdf5_path)
    model = Hdf5_to_Image()
    ret = model.read_hdf5(args.hdf5_path,args.image_type,args.output_path)
    # ret = model.find_detection(args.input_path)
    if ret==1:
        print("\n","File Error.....")
