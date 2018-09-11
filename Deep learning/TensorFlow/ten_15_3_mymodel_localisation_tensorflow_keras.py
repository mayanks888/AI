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

num_classes=4

# dataset = pd.read_csv("SortedXmlresult.csv")
#
# loading my prepared datasets
# dataset = pd.read_csv("/home/mayank-s/PycharmProjects//Datasets/SortedXmlresult_linux.csv")
dataset = pd.read_csv("../../../../Datasets/SortedXmlresult.csv")
x = dataset.iloc[:, 1].values
y = dataset.iloc[:, 2].values
y=dataset.iloc[:, 3:8].values

# new_val_y=np.resize(y,(y.shape[0],1))

#y=y.resize(y.shape[0],1)
# this was used to categorise label if they are more than tow
# y_test = np_utils.to_categorical(y, 2)#cotegorise label

imagelist=[]
for loop in x:
    my_image=cv2.imread(loop,1)#reading my path of all address
    image_scale=cv2.resize(my_image,dsize=(224,224),interpolation=cv2.INTER_NEAREST)#resisizing as per the vgg16 module
    # image_scale=image_scale/255
    imagelist.append(image_scale)#added all pixel values in list

image_list_array=np.array(imagelist)#convet list into array since all calculation required array input

new_image_input,y=shuffle(image_list_array,y,random_state=4)#shuffle data (good practise)

X_train, X_test, y_train, y_test = train_test_split(new_image_input, y, test_size = .20, random_state = 4)#splitting data (no need if test data is present

# ### Placeholders
input_x = tf.placeholder(tf.float32,shape=[None,224,224,3])
y_true = tf.placeholder(tf.float32,shape=[None,4])
# ### Layers
# x_image = tf.reshape(x,[-1,224,224,3])

# importing vgg base_model
# imput_shape=new_image_input[0].shape#good thing to know the shape of input array
x_image = tf.reshape(input_x,[-1,224,224,3])
# input_tensor = tf.keras.Input(shape=(224, 224, 3))
#
# inputshape=(224,224,3)
# base_model=tf.keras.applications.vgg16.VGG16(weights = "imagenet", include_top=True,input_tensor=x_image)#loading vgg16  model trained n imagenet datasets

conv_1_layer = tf.keras.layers.Conv2D(filters=32,kernel_size=(6, 6),strides=(1, 1), activation='relu', padding='same', name='block1_conv1')(x_image)
max_pool_1 = tf.keras.layers.MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool')(conv_1_layer)

conv_2_layer = tf.keras.layers.Conv2D(filters=64,kernel_size=(6, 6),strides=(1, 1), activation='relu', padding='same', name='block2_conv1')(max_pool_1)
max_pool_2 = tf.keras.layers.MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool')(conv_2_layer)

# conv_3_layer = tf.keras.layers.Conv2D(filters=64,kernel_size=(6, 6),strides=(1, 1), activation='relu', padding='same', name='block3_conv1')(max_pool_2)
# max_pool_3 = tf.keras.layers.MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool')(conv_3_layer)


x= tf.keras.layers.Flatten(name='flatten')(max_pool_2)
x = tf.keras.layers.Dense(128, activation='relu', name='fc1')(x)

hold_prob = tf.placeholder(tf.float32)#this made to pass the user defined value for dropout probabilty you could have also used contant value
full_one_dropout = tf.nn.dropout(x,keep_prob=hold_prob)

x = tf.keras.layers.Dense(128, activation='relu', name='fc2')(x)
x= tf.keras.layers.Dense(num_classes,activation='relu',name='fc3')(x)
last_layer_out = tf.keras.layers.Dense(num_classes,activation='relu',name='output')(x)
out=last_layer_out


# custom_vgg_model2 = base_model(input_shape,out)
# custom_vgg_model2 = tf.keras.Model(x_image,last_layer_out)
# print (custom_vgg_model2.summary())
# freeze all the layers except the dense layers
'''for layer in custom_vgg_model2.layers[:-4]:
	layer.trainable = False



out=custom_vgg_model2.output'''

# ---------------------------------------------------------------------------
def find_my_iou(data_ground, data_predicted):
    xminofmax = tf.maximum((data_ground[:, 0]), (data_predicted[:, 0]))
    yminofmax = tf.maximum(data_ground[:, 1], data_predicted[:, 1])
    xmaxofmin = tf.minimum(data_ground[:, 2], data_predicted[:, 2])
    ymaxofmin = tf.minimum(data_ground[:, 3], data_predicted[:, 3])

    # Sub=(xmaxofmin - xminofmax + 1)
    sub1 = tf.add(tf.subtract(xmaxofmin, xminofmax), 1)
    sub2 = tf.add(tf.subtract(ymaxofmin, yminofmax), 1)
    intercetion = tf.multiply(sub1, sub2)

    aog1 = tf.add(tf.abs(tf.subtract(data_ground[:, 0], data_ground[:, 2])), 1)

    aog2 = tf.add(tf.abs(tf.subtract(data_ground[:, 1], data_ground[:, 3])), 1)

    AOG = tf.multiply(aog1, aog2)

    aop1 = tf.add(tf.abs(tf.subtract(data_predicted[:, 0], data_predicted[:, 2])), 1)

    aop2 = tf.add(tf.abs(tf.subtract(data_predicted[:, 1], data_predicted[:, 3])), 1)

    AOP = tf.multiply(aog1, aog2)

    Union = tf.subtract(tf.add(AOG, AOP), intercetion)
    iou = tf.divide(intercetion, Union)
    mean_iou = tf.reduce_mean(iou)
    return mean_iou

iou_loss=find_my_iou(y_true,out)
loss=1-iou_loss
# loss=tf.losses.mean_squared_error(y_true,out)
# iou_val=tf.metrics.mean_iou(labels=y_true,predictions=out,num_classes=4)
# -------------------------------------------------------------------------------


optimizer = tf.train.AdadeltaOptimizer(learning_rate=1)#Inistising your optimising functions
train = optimizer.minimize(loss)#

#correct_prediction = tf.equal(tf.round(out),y_true)
# acc = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
acc=iou_loss
#acc=tf.metrics.accuracy(labels=y_true,predictions=out)
# intialiasing all variable

init=tf.global_variables_initializer()

epochs=20

sess=tf.Session()
sess.run(init)
# sess = tf_debug.TensorBoardDebugWrapperSession(sess, "localhost:7000")
# X_train, X_test, y_train, y_test = train_test_split(new_image_input, y, test_size = .10, random_state = 4)#splitting data (no need if test data is present

for loop in range(epochs):
    # ______________________________________________________________________
    # my batch creater
    # Construct an array data source
    ds = data_source.ArrayDataSource([X_train, y_train])
    # Iterate over samples, drawing batches of 64 elements in random order
    for (batch_X, batch_y) in ds.batch_iterator(batch_size=5, shuffle=True):#shuffle true will randomise every batch
        _,accuracy,output=sess.run([train,acc,out], feed_dict={input_x: batch_X, y_true: batch_y, hold_prob: 0.5})
        print("the train loss is:", accuracy)
        print('thre ytrue is ',batch_y)
        print("the output is",output)
         # ________________________________________________________________________

    # if loop % 1 == 0:
    #     print('Currently on step {}'.format(loop))
    #     print('Accuracy is:')
    #     # Test the Train Modelsess.run(train, feed_dict={x: batch_x, y_true: batch_y, hold_prob: 0.5})
    #     correct_prediction = tf.equal(tf.round(out), y_true)
    #     acc = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    #     print(sess.run(acc, feed_dict={x: X_test, y_true: y_test, hold_prob: 1.0}))

