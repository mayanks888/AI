import cv2
import os
import xml.etree.ElementTree as ET
import  keras
import  tensorflow as tf
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
from keras.preprocessing import image
epochs=1
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

X_train, X_test, y_train, y_test = train_test_split(new_image_input, z, test_size = .10, random_state = 4)#splitting data (no need if test data is present

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

# custom_vgg_model2 = base_model(input_shape,out)
# custom_vgg_model2 = Model(input_shape,out)

# _______________________________________________________________
# creating parallel layers for image localisations
normisation_output=Dense(4, activation='relu', name='norm_output')(x)
print (normisation_output)

custom_vgg_model2 = Model(input_shape,normisation_output)

# --------------------------------------------------------------------------

# freeze all the layers except the dense layers
for layer in custom_vgg_model2.layers[:-3]:
	layer.trainable = False


custom_vgg_model2.summary()
#------------------------------------------------------
# Image argumentation
datagen = ImageDataGenerator(rescale=1./255,
                                    shear_range=0.2,
                                    zoom_range=0.2,
                                    horizontal_flip=True)

# compute quantities required for featurewise normalization
# (std, mean, and principal components if ZCA whitening is applied)
datagen.fit(X_train)


# cb=TensorBoard(log_dir=("/home/mayank-s/PycharmProjects/Data_Science/output_graph/try5"))#,histogram_freq = 1, batch_size = 32,write_graph ="TRUE" )

# ______________
# checking my created loss function
def find_iou(groundbb, predicted_bb):
    data_ground = np.transpose(groundbb)
    data_predicted = np.transpose(predicted_bb)
    xminofmax = np.maximum(data_ground[0], data_predicted[0])
    yminofmax = np.maximum(data_ground[1], data_predicted[1])
    xmaxofmin = np.minimum(data_ground[2], data_predicted[2])
    ymaxofmin = np.minimum(data_ground[3], data_predicted[3])

    interction = ((xmaxofmin - xminofmax + 1) * (ymaxofmin - yminofmax + 1))
#   i have added 1 to all the equation save the equation from giving 0 iou value
#    AOG: area of ground box
    AOG = (np.abs(data_ground[0] - data_ground[2]) + 1) * (np.abs(data_ground[1] - data_ground[3]) + 1)
    #AOP:area of predicted box
    AOP = (np.abs(data_predicted[0] - data_predicted[2]) + 1) * (np.abs(data_predicted[1] - data_predicted[3]) + 1)
    union= (AOG + AOP) - interction
    iou = (interction /union)
    mean_iou = np.mean(iou)
    return (mean_iou)

output_lastlayer = custom_vgg_model2.layers[-1].output
loss_function=find_iou(y_train,output_lastlayer)
# _________________



custom_vgg_model2.compile(loss='mean_squared_error',optimizer='adadelta',metrics=['accuracy'])
# fits the model on batches with real-time data augmentation:
# custom_vgg_model2.fit_generator(datagen.flow(X_train, y_train, batch_size=32),
#                     steps_per_epoch=len(X_train) / 32, epochs=epochs)

custom_vgg_model2.fit_generator(datagen.flow(X_train, y_train, batch_size=32),
                    steps_per_epoch=5, epochs=epochs)

    # # cv2.rectangle(image_scale, pt1=(125,84),pt2=(159,38),color= (0,255,0), thickness=2)
    # cv2.rectangle(image_scale, pt1=top, pt2=bottom, color=(0, 255, 0), thickness=2)
    # cv2.imshow('streched image',image_scale)
    # cv2.waitKey(1000)
    # cv2.destroyAllWindows()
    # val+=1
test_image=image.load_img(path="../../../../Datasets/cat_dogs/test1/506.jpg",target_size=(224,224))
test_image=image.img_to_array(test_image)
test_image=np.expand_dims(test_image,axis=0)
result=custom_vgg_model2.predict(test_image)
# print (training_data.class_indices)
print(((result)))
# print('prediction',decode_predictions(result))