import cv2
import os
import xml.etree.ElementTree as ET
import  pandas as pd
import numpy as np
from keras.utils import np_utils
from keras import applications
from keras.callbacks import TensorBoard
from tensorflow.python import debug as tf_debug
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report,confusion_matrix
import tensorflow as tf
from batchup import data_source
import read_pascal_voc
import region_proposal_network
import anchor_target_layer
import region_proposal_network2
from keras.models import Model
from keras.layers import merge, Input
# *************************************************************************
# getting datasets from voc pascal
rpn=[]
feat_stride        = 16
anchor_scale       = [ 8, 16, 32 ]
Mode="train"
imageNameFile = "../../../Datasets/VOCdevkit/VOC2012/ImageSets/Main/aeroplane_train.txt"
vocPath       = "../../../Datasets/VOCdevkit/VOC2012"

Image_data,boundingBX_labels,im_dims=read_pascal_voc.prepareBatch(0,3,imageNameFile,vocPath)
# print(Image_data,boundingBX_labels,im_dims)
# *************************************************************************

# _____________________________________________________________________________
# num_classes=4
# # dataset = pd.read_csv("SortedXmlresult.csv")
# #
# # loading my prepared datasets
# # dataset = pd.read_csv("/home/mayank-s/PycharmProjects//Datasets/SortedXmlresult_linux.csv")
# dataset = pd.read_csv("../../../../Datasets/SortedXmlresult_linux.csv")
# x = dataset.iloc[:, 1].values
# y = dataset.iloc[:, 2].values
# y=dataset.iloc[:, 3:8].values
#
# # new_val_y=np.resize(y,(y.shape[0],1))
#
# #y=y.resize(y.shape[0],1)
# # this was used to categorise label if they are more than tow
# # y_test = np_utils.to_categorical(y, 2)#cotegorise label
#
# imagelist=[]
# for loop in x:
#     my_image=cv2.imread(loop,1)#reading my path of all address
#     image_scale=cv2.resize(my_image,dsize=(224,224),interpolation=cv2.INTER_NEAREST)#resisizing as per the vgg16 module
#     imagelist.append(image_scale)#added all pixel values in list
#
# image_list_array=np.array(imagelist)#convet list into array since all calculation required array input
#
# new_image_input,y=shuffle(image_list_array,y,random_state=4)#shuffle data (good practise)
#
# X_train, X_test, y_train, y_test = train_test_split(new_image_input, y, test_size = .20, random_state = 4)#splitting data (no need if test data is present
# _____________________________________________________________________________
# ### Placeholders
input_x = tf.placeholder(tf.float32, [None, None, None, 3])  # [ batch_size, height, width, channel]
gt_bbox = tf.placeholder(tf.int32, [None,20, 5])
im_dimsal = tf.placeholder(tf.int32, [None, 2])

x_image = tf.reshape(input_x,[-1,600,1000,3])
# input_tensor = tf.keras.Input(shape=(224, 224, 3))
# input_shape = Input(shape=(1,600,1000,3))
# inputshape=(1,600,1000,3)
base_model=tf.keras.applications.vgg16.VGG16(weights = "imagenet", include_top=False,input_tensor=x_image)#loading vgg16  model trained n imagenet datasets

last_layer = base_model.get_layer('block5_conv3').output#taking the previous layers from vgg16 officaal model
# features=last_layer
custom_vgg_model2=tf.keras.Model(x_image,last_layer)
#-------------------------------------------

rpn_spc_layer = tf.keras.layers.Conv2D(filters=512,kernel_size=(1, 1),strides=(1, 1), activation='relu', padding='same', name='rpn_conv_1')(last_layer)
rpn_cls_score_lr = tf.keras.layers.Conv2D(filters=18,kernel_size=(1, 1),strides=(1, 1), activation='relu', padding='same', name='rpn_class_score')(rpn_spc_layer)
rpn_bb_reg_lr = tf.keras.layers.Conv2D(filters=36,kernel_size=(1, 1),strides=(1, 1), activation='relu', padding='same', name='rpn_bbox')(rpn_spc_layer)

# custom_vgg_model2 = Model(inputs=x_image,outputs=rpn_spc_layer)

custom_vgg_model2=tf.keras.Model(x_image,rpn_bb_reg_lr)
# freeze all the layers except the dense layers
for layer in custom_vgg_model2.layers[:-2]:
	layer.trainable = False

custom_vgg_model2.summary()
# _______________________________________________________________________________
# region_proposal_network.RegionProposalNetwork(rpn_cls_score_lr, gt_bbox, im_dims, anchor_scale, Mode)
rpn_labels, rpn_bbox_targets, rpn_bbox_inside_weights, rpn_bbox_outside_weights = anchor_target_layer.anchor_target_layer(rpn_cls_score_lr, gt_bbox, im_dims, feat_stride,anchor_scale)

rpn_box_loss=region_proposal_network2.rpn_bbox_loss(rpn_bb_reg_lr, rpn_bbox_targets, rpn_bbox_inside_weights, rpn_bbox_outside_weights)

# rpn_box_loss=.1
rpn_cla_loss=region_proposal_network2.rpn_cls_loss(rpn_cls_score_lr, rpn_labels)

total_loss=tf.add(rpn_box_loss,rpn_cla_loss)

optimizer = tf.train.AdadeltaOptimizer(learning_rate=1)#Inistising your optimising functions
train = optimizer.minimize(total_loss)


'''# freeze all the layers except the dense layers
for layer in custom_vgg_model2.layers[:-5]:
	layer.trainable = False

print (custom_vgg_model2.summary())

out=custom_vgg_model2.output'''

# ---------------------------------------------------------------------------

# iou_loss=find_my_iou(y_true,out)
# loss=1-iou_loss
# loss=tf.losses.mean_squared_error(y_true,out)
# loss=tf.metrics.mean_iou(labels=y_true,predictions=out,num_classes=4)
# -------------------------------------------------------------------------------


# optimizer = tf.train.AdadeltaOptimizer(learning_rate=1)#Inistising your optimising functions
# train = optimizer.minimize(loss)#

#correct_prediction = tf.equal(tf.round(out),y_true)
# acc = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
# acc=loss
#acc=tf.metrics.accuracy(labels=y_true,predictions=out)
# intialiasing all variable

init=tf.global_variables_initializer()

epochs=2

sess=tf.Session()

sess.run(init)
#sess = tf_debug.TensorBoardDebugWrapperSession(sess, "localhost:7000")

# sess = tf_debug.TensorBoardDebugWrapperSession(sess, "localhost:7000")
# X_train, X_test, y_train, y_test = train_test_split(new_image_input, y, test_size = .10, random_state = 4)#splitting data (no need if test data is present

for loop in range(epochs):

    # ______________________________________________________________________
    # my batch creater
    # Construct an array data source
    ds = data_source.ArrayDataSource([Image_data, boundingBX_labels,im_dims])
    # Iterate over samples, drawing batches of 64 elements in random order
    for (image_input, gt_box,image_dim) in ds.batch_iterator(batch_size=1, shuffle=True):#shuffle true will randomise every batch
        # accuoutput=sess.run([rpn_labels], feed_dict={input_x: image_input, gt_bbox: gt_box, im_dimsal: image_dim})
        _,loss,target = sess.run([train,total_loss,rpn_bbox_targets], feed_dict={input_x: image_input, gt_bbox: gt_box, im_dimsal: image_dim})
        print ("the total loss is",loss)
        print ("max target value",np.max(target))

