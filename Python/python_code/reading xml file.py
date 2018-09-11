# this one was dedicated to pascal voc data  annotation
import xml.etree.cElementTree as et
#extracting xml file
tree=et.parse(source='C:/Users/mayank/Documents/Datasets/VOCtrainval_11-May-2012/VOCdevkit/VOC2012/Annotations/2007_000027.xml')
# get xml file data
root=tree.getroot()
print (root)

# read any text with thier label
print (root.find('size').find('width').text)
print(root.find('source').find('annotation').text)

for obj in root.iter('object'):
    print(obj.text)
# for child in root:
#