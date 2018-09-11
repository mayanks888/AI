import argparse
import os
from PIL import Image
import io
import h5py
import numpy as np
from collections import namedtuple, OrderedDict
import pandas as pd
import cv2

classes=['person','motor','bus','car','bike','traffic light','traffic sign','truck', 'train']

def class_text_to_int(row_label):
    if row_label in classes:
        val=classes.index(row_label)
        return val+1
    else:
        print(row_label)
        print('classes does not exist in defined classes')
        raise IOError

def add_image_to_datasets(voc_path, group_val, images, start=0):
    # image_name = group_val.filename + ".jpg"
    image_name = group_val
    image_data = get_image_for_id(voc_path, image_name)
    images[start] = image_data


def add_to_dataset(voc_path, group_val, images, boxes, start=0):

    image_name = group_val.filename + ".jpg"
    image_data = get_image_for_id(voc_path, image_name)
    images[start] = image_data
    image_boxes = get_boxes_for_id(group_val)
    boxes[start]=image_boxes


def get_boxes_for_id(group):
    boxes = []
    for index, row in group.object.iterrows():
        # print(index)
        # print (row)
        class_no = (class_text_to_int(row['class']))
        # name=row['class'].encode('utf8')
        xmin, xmax, ymin, ymax = row['xmin'], row['xmax'], row['ymin'], row['ymax']
        bbox = (class_no, xmin, ymin, xmax, ymax)
        boxes.extend(bbox)
    final_bbox = np.array(boxes)
    return final_bbox
    # train_boxes[loop] = final_bbox



def get_image_for_id(path, image_id):

    """Get image data as uint8 array for given image.
    Parameters
    ----------
    path : path of image directory
    -------
    image_data : array of uint8
        Compressed JPEG byte string represented as array of uint8.
    """
    fname = os.path.join(path, image_id)
    with open(fname, 'rb') as in_file:
        data = in_file.read()
    # Use of encoding based on: https://github.com/h5py/h5py/issues/745
    return np.fromstring(data, dtype='uint8')



def split(df, group):
    data = namedtuple('data', ['filename', 'object'])
    gb = df.groupby(group)
    return [data(filename, gb.get_group(x)) for filename, x in zip(gb.groups.keys(), gb.groups)]



if __name__ == '__main__':
    voc_path = '/home/mayank-s/Desktop/Link to Datasets/aptive/object_detect'
    train_csv_name = "berkely_train_for_hdf5.csv"
    image_path='/home/mayank-s/Desktop/Link to Datasets/aptive/object_detect/input'

    print('Creating HDF5 dataset structure.')
    fname = os.path.join(voc_path, 'Image_to_hdf51.hdf5')

    # csv_path=os.path.join(voc_path,train_csv_name)
    # # reading train csv file
    # datasets_data = pd.read_csv(csv_path)
    # grouped = split(datasets_data, 'filename')

    voc_h5file = h5py.File(fname, 'w')

    uint8_dt = h5py.special_dtype(vlen=np.dtype('uint8'))  # variable length uint8
    vlen_int_dt = h5py.special_dtype(vlen=np.dtype(int))  # variable length default int

    # # Creating struture for hdf5 fromat
    # train_group = voc_h5file.create_group('train')
    # val_group = voc_h5file.create_group('val')
    test_group = voc_h5file.create_group('test')


    # store class list for reference class ids as csv fixed-length numpy string
    # voc_h5file.attrs['classes'] = np.string_(str.join(',', classes))
    # total_train_ids=len(grouped)+5
    # total_train_ids=10942
    # val_ids=2000
    # test_ids=len(filenames)

    # store images as variable length uint8 arrays
    # train_images = train_group.create_dataset(
    #     'images', shape=(total_train_ids,), dtype=uint8_dt)
    # val_images = val_group.create_dataset(
    #     'images', shape=((val_ids),), dtype=uint8_dt)
    # test_images = test_group.create_dataset(
    #     'images', shape=((test_ids),), dtype=uint8_dt)

    # # store boxes as class_id, xmin, ymin, xmax, ymax
    # train_boxes = train_group.create_dataset(
    #     'boxes', shape=(total_train_ids,), dtype=vlen_int_dt)
    # val_boxes = val_group.create_dataset(
    #     'boxes', shape=((val_ids),), dtype=vlen_int_dt)
    # test_boxes = test_group.create_dataset(
    #     'boxes', shape=((test_ids),), dtype=vlen_int_dt)

    loop=0

    if not os.path.exists(image_path):
        print("Input folder not found")
        # return 1

    # if not os.path.exists(image_path):
    #     print('Output folder not present. Creating New folder...')
    #     os.makedirs(output_folder)
    start=0
    for root, _, filenames in os.walk(image_path):
        if (len(filenames) == 0):
            print("Input folder is empty")
            # return 1
        test_ids = len(filenames)
        test_images = test_group.create_dataset(name='input_images', shape=((test_ids),), dtype=uint8_dt)
        # test_images = test_group.create_dataset(name='images', shape=(3, 2), dtype=uint8_dt)
        for filename in filenames:
            # try:
                image_data = get_image_for_id(image_path, filename)
                test_images[start] = image_data
                # filename = filename.encode()
                # dt = np.dtype([(filename, np.unicode, 8)])
                # # np.fromstring(filename, dtype=np.uint8)
                # # test_images[start, 1] = filename
                # # test_images[start] = filename

                start+=1
                print(start)

   # test_group.H5IMlink_palette


    print('Closing HDF5 file.')
    voc_h5file.close()
    print('Done.')
