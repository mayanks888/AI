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


if __name__ == '__main__':
    voc_path = '/home/mayank-s/PycharmProjects/Datasets/Berkely_DeepDrive'
    train_csv_name = "berkely_train_for_hdf5.csv"
    train_image_path='/home/mayank-s/PycharmProjects/Datasets/Berkely_DeepDrive/bdd100k/images/100k/train'

    print('Reading HDF5 dataset structure.')
    fname = os.path.join(voc_path, 'My_developed.hdf5')

    my_data=h5py.File(fname,'r')
    at_item = my_data.attrs.values()#reading attributes
    print(list(at_item))
    val=my_data.get('train')
    image_data=val.get('images')
    # cool=list(image_data)
    image_return=np.array(image_data)
    print(image_return[0])
    # img = Image.open(io.BytesIO(image_val))
    # img.show()
   # print(list(data_return))

    bbox_data=val.get('boxes')
    box_return=np.array(bbox_data)
    print(box_return[0])
    my_data.close()
    print("closing hdf5 file")
    print(image_return[7])

'''for loop,bbox_val in enumerate (box_return):
# for loop, image_val in enumerate(image_return):
                # print(image_return[0])
                # image_val= image_return[len]
                # img = Image.open(io.BytesIO(image_val))
                # file_name="file_"+str(loop) +'.jpeg'
                print(bbox_val)
                # image_path=os.path.join(output_path, file_name)
                # img.save(imag'''


ds = data_source.ArrayDataSource([image_return, box_return])
    # Iterate over samples, drawing batches of 64 elements in random order
for (batch_X, batch_y) in ds.batch_iterator(batch_size=1, shuffle=True):#shuffle true will randomise every batch
    print(batch_X.shape)
    print(batch_y)
    data = batch_y.reshape((batch_y.shape[0], 1))
    print(data)
    #box=np.reshape(-1,5)
    
    # break


# e_path, 'JPEG')