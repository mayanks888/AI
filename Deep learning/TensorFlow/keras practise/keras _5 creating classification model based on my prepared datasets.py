import cv2
import os
import xml.etree.ElementTree as ET
import  pandas as pd
import numpy as np
from keras.utils import np_utils
from keras import applications
from keras.callbacks import TensorBoard
from keras.preprocessing.image import ImageDataGenerator
from keras import optimizers
from keras.models import Sequential, Model
from keras.layers import Dropout, Flatten, Dense
from keras import optimizers
from keras import backend as keras_backend
from keras.callbacks import TensorBoard
from keras.layers import merge, Input
from tensorflow.python import debug as tf_debug
from sklearn.utils import shuffle
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report,confusion_matrix

epochs=5
num_classes=1

# dataset = pd.read_csv("SortedXmlresult.csv")
#
# loading my prepared datasets
dataset = pd.read_csv("SortedXmlresult_linux.csv")
x = dataset.iloc[:, 1].values
y = dataset.iloc[:, 2].values
z=dataset.iloc[:, 3:8].values

# this was used to categorise label if they are more than tow
y_test = np_utils.to_categorical(y, 2)#cotegorise label

imagelist=[]
for loop in x:
    my_image=cv2.imread(loop,1)#reading my path of all address
    image_scale=cv2.resize(my_image,dsize=(224,224),interpolation=cv2.INTER_NEAREST)#resisizing as per the vgg16 module
    imagelist.append(image_scale)#added all pixel values in list

image_list_array=np.array(imagelist)#convet list into array since all calculation required array input

new_image_input,y=shuffle(image_list_array,y,random_state=4)#shuffle data (good practise)

X_train, X_test, y_train, y_test = train_test_split(new_image_input, y, test_size = .10, random_state = 4)#splitting data (no need if test data is present

# ____________________________________________________
# importing vgg base_model
# imput_shape=new_image_input[0].shape#good thing to know the shape of input array
input_shape = Input(shape=(224, 224, 3))
# inputshape=(224,224,3)
base_model=applications.vgg16.VGG16(weights = "imagenet", include_top=True,input_tensor=input_shape)#loading vgg16  model trained n imagenet datasets
# print (base_model.summary())
'''base_model.layers.pop()
out = base_model.add(Dense(1, activation='sigmoid', name='output'))
print (base_model.summary())'''

last_layer = base_model.get_layer('block5_pool').output#taking the previous layers from vgg16 officaal model
x= Flatten(name='flatten')(last_layer)
x = Dense(128, activation='relu', name='fc1')(x)
x = Dense(128, activation='relu', name='fc2')(x)
out = Dense(num_classes, activation='sigmoid', name='output')(x)
# custom_vgg_model2 = base_model(input_shape,out)
custom_vgg_model2 = Model(input_shape,out)

# freeze all the layers except the dense layers
for layer in custom_vgg_model2.layers[:-3]:
	layer.trainable = False

custom_vgg_model2.summary()
custom_vgg_model2.compile(loss='binary_crossentropy',optimizer='adadelta',metrics=['accuracy'])
#------------------------------------------------------
# Image argumentation

datagen = ImageDataGenerator(
    featurewise_center=True,
    featurewise_std_normalization=True,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True)

# compute quantities required for featurewise normalization
# (std, mean, and principal components if ZCA whitening is applied)
datagen.fit(X_train)

# fits the model on batches with real-time data augmentation:
custom_vgg_model2.fit_generator(datagen.flow(X_train, y_train, batch_size=32),
                    steps_per_epoch=len(X_train) / 32, epochs=epochs)



    # # cv2.rectangle(image_scale, pt1=(125,84),pt2=(159,38),color= (0,255,0), thickness=2)
    # cv2.rectangle(image_scale, pt1=top, pt2=bottom, color=(0, 255, 0), thickness=2)
    # cv2.imshow('streched image',image_scale)
    # cv2.waitKey(1000)
    # cv2.destroyAllWindows()
    # val+=1