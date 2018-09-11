import os
import xml.etree.ElementTree as ET
import numpy as np
import cv2

labels = ["aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow", "diningtable", "dog",
          "horse", "motorbike", "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor"]

def generate_batch_data(vocPath,imageNameFile,batch_size,sample_number):
    """
    Args:
      vocPath: the path of pascal voc data
      imageNameFile: the path of the file of image names
      batchsize: batch size, sample_number should be divided by batchsize
    Funcs:
      A data generator generates training batch indefinitely
    """
    class_num = 20
    #Read all the data once and dispatch them out as batches to save time
    TotalimageList = prepareBatch(0,sample_number,imageNameFile,vocPath)

    while 1:
        batches = sample_number // batch_size
        for i in range(batches):
            images = []
            boxes = []
            sample_index = np.random.choice(sample_number,batch_size,replace=True)
            #sample_index = [3]
            for ind in sample_index:
                image = TotalimageList[ind]
                #print image.imgPath
                # image_array = crop_detection(image.imgPath,new_width=448,new_height=448)
                #image_array = np.expand_dims(image_array,axis=0)

                y = []
                for i in range(7):
                    for j in range(7):
                        box = image.boxes[i][j]
                        '''
                        ############################################################
                        #x,y,h,w,one_hot class label vector[0....0],objectness{0,1}#
                        ############################################################
                        '''
                        if(box.has_obj):
                            obj = box.objs[0]

                            y.append(obj.x)
                            y.append(obj.y)
                            y.append(obj.h)
                            y.append(obj.w)

                            labels = [0]*20
                            labels[obj.class_num] = 1
                            y.extend(labels)
                            y.append(1) #objectness
                        else:
                            y.extend([0]*25)
                y = np.asarray(y)
                #y = np.reshape(y,[1,y.shape[0]])

                images.append(image_array)
                boxes.append(y)
            #return np.asarray(images),np.asarray(boxes)
            yield np.asarray(images),np.asarray(boxes)

def prepareBatch(start,end,imageNameFile,vocPath):
    """
    Args:
      start: the number of image to start
      end: the number of image to end
      imageNameFile: the path of the file that contains image names
      vocPath: the path of pascal voc dataset
    Funs:
      generate a batch of images from start~end
    Returns:
      A list of end-start+1 image objects
    """
    HEIGHT=600
    WIDTH=1000
    CHANNEL=3
    label_dimension=5
    max_label_found=20
    batch_size = end - start
    Image_data = np.zeros((batch_size,HEIGHT,WIDTH,CHANNEL), dtype=np.float32)
    # labels = np.zeros( (0, label_dimension), dtype = np.float32 )
    boundingBX_labels = np.zeros((batch_size, max_label_found, label_dimension), dtype=np.float32)
    im_dims=np.zeros((batch_size,2), dtype=np.float32)
    imageList = []
    labels = ["aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow", "diningtable", "dog", "horse", "motorbike", "person", "pottedplant", "sheep", "sofa", "train","tvmonitor"]
    file = open(imageNameFile)
    imageNames = file.readlines()
    for i in range(start,end):
        imgName = imageNames[i].strip('\n')
        #
        # I'm chaning something in this'
        imgName1=imgName.split(" ")
        imgName=imgName1[0]
        # ____________________________
        imgPath = os.path.join(vocPath,'JPEGImages',imgName)+'.jpg'
        xmlPath = os.path.join(vocPath,'Annotations',imgName)+'.xml'


        img = cv2.imread(imgPath,1)
        image_scale = cv2.resize(img, dsize=(WIDTH,HEIGHT), interpolation=cv2.INTER_NEAREST)
        
        Image_data[i]=image_scale
        im_dims[i] = image_scale.shape[:2]
        #Remeber to change to different widht and height
        #labels=
        bblabel=parseXML(xmlPath,labels,HEIGHT,WIDTH)
        # imageList.append(img)
        bb=np.array(bblabel)
        boundingBX_labels[i,:len(bb),:]=bb

    return Image_data,boundingBX_labels,im_dims

def parseXML(xmlPath, labels,pixel_size_height,pixel_size_width):
    """
    Args:
      xmlPath: The path of the xml file of this image
      labels: label names of pascal voc dataset
      side: an image is divided into side*side grid
    """
    tree = ET.parse(xmlPath)
    root = tree.getroot()

    width = int(root.find('size').find('width').text)
    height = int(root.find('size').find('height').text)
    bblabel=[]
    for obj in root.iter('object'):
        class_num = labels.index(obj.find('name').text)#finding the index of label mention so as to store data into correct label
        object_name=obj.find('name').text
        bndbox = obj.find('bndbox')
        xmin = int(bndbox.find('xmin').text)
        ymin = int(bndbox.find('ymin').text)
        xmax = int(bndbox.find('xmax').text)
        ymax = int(bndbox.find('ymax').text)
        h = ymax - ymin
        w = xmax - xmin
        # which cell this obj falls into
        centerx = (xmax + xmin) / 2.0
        centery = (ymax + ymin) / 2.0
        #448 is size of the input image
        newx = (pixel_size_width / width) * centerx
        newy = (pixel_size_height / height) * centery

        h_new = h * (pixel_size_height / height)
        w_new = w * (pixel_size_width / width)

        new_xmin=int(newx-(w_new/2))
        new_xmax=int(newx+(w_new/2))
        new_ymin=int(newy-(h_new/2))
        new_ymax=int(newy+(h_new/2))

        coordinate = [new_xmin, new_ymin, new_xmax, new_ymax, class_num]
        result = [float(c) for c in coordinate]
        result = np.array(result, dtype=np.int32)
        bblabel.append(coordinate)
    return bblabel


#
# imageNameFile = "../../Datasets/VOCdevkit/VOC2012/ImageSets/Main/aeroplane_train.txt"
# vocPath       = "../../Datasets/VOCdevkit/VOC2012"
#
#
# #image_array,y = generate_batch_data(vocPath,imageNameFile,1,sample_number=200)
# Image_data,boundingBX_labels,im_dims=prepareBatch(0,2,imageNameFile,vocPath)
# print(Image_data,boundingBX_labels,im_dims)
