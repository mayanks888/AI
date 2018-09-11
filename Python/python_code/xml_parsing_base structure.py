import xml.etree.cElementTree as ET
import numpy as np
xml_source='C:/Users/mayank/Documents/Datasets/VOCtrainval_11-May-2012/VOCdevkit/VOC2012/Annotations/2007_000027.xml'
xml_source="C:/Users/mayank/Documents/Datasets/VOCtrainval_11-May-2012/VOCdevkit/VOC2012/Annotations/2007_001027.xml"
def parseXML(xmlPath, labels, side):
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

    for obj in root.iter('object'):
        class_num = labels.index(obj.find('name').text)#finding the index of label mention so as to store data into correct label
        bndbox = obj.find('bndbox')
        xmin = int(bndbox.find('xmin').text)
        ymin = int(bndbox.find('ymin').text)
        xmax = int(bndbox.find('xmax').text)
        ymax = int(bndbox.find('ymax').text)
        h = ymax - ymin
        w = xmax - xmin
        # objif = objInfo(xmin/448.0,ymin/448.0,np.sqrt(ymax-ymin)/448.0,np.sqrt(xmax-xmin)/448.0,class_num)

        # which cell this obj falls into
        centerx = (xmax + xmin) / 2.0
        centery = (ymax + ymin) / 2.0
        #448 is size of the input image
        newx = (448.0 / width) * centerx
        newy = (448.0 / height) * centery

        h_new = h * (448.0 / height)
        w_new = w * (448.0 / width)

        cell_size = 448.0 / side
        col = int(newx / cell_size)
        row = int(newy / cell_size)
        # print "row,col:",row,col,centerx,centery

        cell_left = col * cell_size
        cell_top = row * cell_size
        cord_x = (newx - cell_left) / cell_size
        cord_y = (newy - cell_top) / cell_size

        #objif = objInfo(cord_x, cord_y, np.sqrt(h_new / 448.0), np.sqrt(w_new / 448.0), class_num)
        #self.boxes[row][col].has_obj = True
        #self.boxes[row][col].objs.append(objif)

# mylabel='head'
side=7
mylabel = ["aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow", "diningtable", "dog", "horse", "motorbike", "person", "pottedplant", "sheep", "sofa", "train","tvmonitor"]
parseXML(xml_source,mylabel,side)
