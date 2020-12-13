import os

import cv2
import tensorflow as tf


# from tf.keras.layers import Dense,Conv2D,MaxPooling2D,Flatten,AveragePooling2D,Dropout,BatchNormalization,Activation
# from tf.keras.layers import Dense
# import keras
# from keras.models import Model,Input
################################################################3
# import tensorflow.keras.layer.Dropout as Dropout
# import tensorflow.keras.layer.Conv2D as Conv2D
# import tensorflow.keras.layer.MaxPooling2D as MaxPooling2D
# import tensorflow.keras.layer.Flatten as Flatten
# import tensorflow.keras.layer.AveragePooling2D as AveragePooling2D
# import tensorflow.keras.layer.BatchNormalization as BatchNormalization
# import tensorflow.keras.layer.Activation as Activation
# import tensorflow.keras.layer.Dense as Dense
# import tensorflow.keras.models.Model as Model
# import tensorflow.keras.models.Input as Input

##################################################################

def Unit(x, filters, pool=False):
    res = x
    if pool:
        x = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(x)
        res = tf.keras.layers.Conv2D(filters=filters, kernel_size=[1, 1], strides=(2, 2), padding="same")(res)
    out = tf.keras.layers.BatchNormalization()(x)
    out = tf.keras.layers.Activation("relu")(out)
    out = tf.keras.layers.Conv2D(filters=filters, kernel_size=[3, 3], strides=[1, 1], padding="same")(out)

    out = tf.keras.layers.BatchNormalization()(out)
    out = tf.keras.layers.Activation("relu")(out)
    out = tf.keras.layers.Conv2D(filters=filters, kernel_size=[3, 3], strides=[1, 1], padding="same")(out)

    out = tf.keras.layers.add([res, out])

    return out


# Define the model


def MiniModel(input_shape):
    images = tf.keras.Input(input_shape)
    net = tf.keras.layers.Conv2D(filters=32, kernel_size=[3, 3], strides=[1, 1], padding="same")(images)
    net = Unit(net, 32)
    net = Unit(net, 32)
    net = Unit(net, 32)

    net = Unit(net, 64, pool=True)
    net = Unit(net, 64)
    net = Unit(net, 64)

    net = Unit(net, 128, pool=True)
    net = Unit(net, 128)
    net = Unit(net, 128)

    net = Unit(net, 256, pool=True)
    net = Unit(net, 256)
    net = Unit(net, 256)

    net = tf.keras.layers.BatchNormalization()(net)
    net = tf.keras.layers.Activation("relu")(net)
    net = tf.keras.layers.Dropout(0.25)(net)

    net = tf.keras.layers.AveragePooling2D(pool_size=(4, 4))(net)
    net = tf.keras.layers.Flatten()(net)
    net = tf.keras.layers.Dense(units=3, activation="softmax")(net)

    model = tf.keras.models.Model(inputs=images, outputs=net)

    return model


input_shape = (32, 32, 3)
model = MiniModel(input_shape)
print(model.summary())
# model.compile(optimizer=Adam(0.001),loss="categorical_crossentropy",metrics=["accuracy"])
model.compile(optimizer='adam', loss="categorical_crossentropy", metrics=["accuracy"])
###################################################################
# inputshape=(224,224,3)
# base_model=applications.vgg16.VGG16(weights = "imagenet", include_top=True,input_shape=inputshape)
# print (base_model.summary())
#
# # last_layer=base_model.layers('fc2')
# #now we are used to work on a sequantial layer in keras aso we will conver vgg16 model into sequantal model and use as our train model
# new_sequential=Sequential()
# print(type(base_model))
# for mylayer in base_model.layers:
#     mylayer.trainable=False#this is done to set the weight as predefined
#     new_sequential.add(mylayer)
#
# print(type(new_sequential))
#
# new_sequential.layers.pop()#remove my last layer
#
# new_sequential.add(Dense(output_dim=1,activation='sigmoid'))
#
#
# #now I'll bring my old parameter defined
#
# new_sequential.compile(optimizer="adam",loss='binary_crossentropy',metrics=['accuracy'])
#
# print (new_sequential.summary())
#######################################################################################
# image argumentation
# from tf.keras.preprocessing.image import ImageDataGenerator
tf.keras.preprocessing.image.ImageDataGenerator

train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1. / 255, shear_range=0.2, zoom_range=0.2,
                                                                horizontal_flip=True)

test_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1. / 255)
# this is for winodows path

training_data = train_datagen.flow_from_directory(directory="/home/mayank_sati/Desktop/trainig/train",
                                                  target_size=(32, 32), batch_size=8, class_mode='categorical')

# this is for winodows path

test_validation = test_datagen.flow_from_directory("/home/mayank_sati/Desktop/trainig/test", target_size=(32, 32),
                                                   batch_size=8, class_mode='categorical')

# keras_backend.set_session(tf_debug.TensorBoardDebugWrapperSession(tf.Session(),"localhost:7000"))
# linux
# cb=TensorBoard(log_dir=("/home/mayank-s/PycharmProjects/Data_Science/output_graph/try4"))#,histogram_freq = 1, batch_size = 32,write_graph ="TRUE" )
# @windows
# cb=TensorBoard(log_dir=("C:/Users/mayank/Documents/Data_Science/output_graph/try4"))

# model.fit_generator(generator=training_data, steps_per_epoch=5, epochs=50, validation_data=test_validation, validation_steps=5)

# model.fit_generator(
#         train_generator,
#         steps_per_epoch=nb_train_samples // batch_size,
#         epochs=epochs,
#         validation_data=validation_generator,
#         validation_steps=nb_validation_samples // batch_size)
# making new prediction
model.load_weights('first_try2.h5')
# model.save_weights('first_try2.h5')
import numpy as np

# from tfkeras.preprocessing import image

# test_image=image.load_img(path="C:/Users/mayank/Documents/Datasets/Cat_dogs/test1/509.jpg",target_size=(64,64))
# linux path
# test_image=tf.keras.preprocessing.image.load_img(path='/home/mayank_sati/Desktop/l/red_light173.jpg',target_size=(32,32))
# test_image=tf.keras.preprocessing.image.load_img(path='/home/mayank_sati/Desktop/l/green_light502.jpg',target_size=(32,32))
# test_image=tf.keras.preprocessing.image.img_to_array(test_image)
# test_image=np.expand_dims(test_image,axis=0)
# result=model.predict(test_image)
# print(1)
# dat= (training_data.class_indices)
# print ({v:k for k, v in dat.items()}[result.argmax()])
# # print((int(result)))

val = 0
input_folder = '/home/mayank_sati/Desktop/l'
test = True

if test == True:
    for root, _, filenames in os.walk(input_folder):
        if (len(filenames) == 0):
            print("Input folder is empty")
        # time_start = time.time()
        for filename in filenames:
            filename = input_folder + "/" + filename
            test_image = tf.keras.preprocessing.image.load_img(path=filename, target_size=(32, 32))
            test_image = tf.keras.preprocessing.image.img_to_array(test_image)
            test_image = np.expand_dims(test_image, axis=0)
            result = model.predict(test_image)
            print(1)
            dat = (training_data.class_indices)
            res = ({v: k for k, v in dat.items()}[result.argmax()])
            print(res)

            image_scale = cv2.imread(filename, 1)
            cv2.putText(image_scale, res, (10, 10), cv2.FONT_HERSHEY_SIMPLEX, .50, (0, 255, 0), lineType=cv2.LINE_AA)
            cv2.imshow('streched_image', image_scale)
            # output_folder_path="/home/mayank-s/PycharmProjects/Datasets/aptive/object_detect/output/output_merccedes.png"
            # filepath=output_folder_path+my_image+".png"
            # cv2.imwrite(filepath,my_image)
            # font = cv2.FONT_HERSHEY_SIMPLEX
            # cv2.putText(image_scale, z[val], (100, 200), font, 4, (0, 255, 0), 2, cv2.LINE_AA)
            # if val > 185:
            #     print(1)
            # cv2.waitKey(1000)
            ch = cv2.waitKey(1000)  # refresh after 1 milisecong
            if ch & 0XFF == ord('q'):
                cv2.destroyAllWindows()
            cv2.destroyAllWindows()
