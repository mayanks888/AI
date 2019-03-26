from keras import applications
from keras.callbacks import TensorBoard
from keras.preprocessing.image import ImageDataGenerator
from keras import optimizers
from keras.models import Sequential, Model
from keras.layers import Dropout, Flatten, Dense
from keras import optimizers
from keras import backend as keras_backend
from keras.callbacks import TensorBoard
from tensorflow.python import debug as tf_debug
from keras.layers import Dense,Conv2D,MaxPooling2D,Flatten,AveragePooling2D,Dropout,BatchNormalization,Activation
import keras
from keras.models import Model,Input
##################################################################

def Unit(x,filters,pool=False):
    res = x
    if pool:
        x = MaxPooling2D(pool_size=(2, 2))(x)
        res = Conv2D(filters=filters,kernel_size=[1,1],strides=(2,2),padding="same")(res)
    out = BatchNormalization()(x)
    out = Activation("relu")(out)
    out = Conv2D(filters=filters, kernel_size=[3, 3], strides=[1, 1], padding="same")(out)

    out = BatchNormalization()(out)
    out = Activation("relu")(out)
    out = Conv2D(filters=filters, kernel_size=[3, 3], strides=[1, 1], padding="same")(out)

    out = keras.layers.add([res,out])

    return out

#Define the model


def MiniModel(input_shape):
    images = Input(input_shape)
    net = Conv2D(filters=32, kernel_size=[3, 3], strides=[1, 1], padding="same")(images)
    net = Unit(net,32)
    net = Unit(net,32)
    net = Unit(net,32)

    net = Unit(net,64,pool=True)
    net = Unit(net,64)
    net = Unit(net,64)

    net = Unit(net,128,pool=True)
    net = Unit(net,128)
    net = Unit(net,128)

    net = Unit(net, 256,pool=True)
    net = Unit(net, 256)
    net = Unit(net, 256)

    net = BatchNormalization()(net)
    net = Activation("relu")(net)
    net = Dropout(0.25)(net)

    net = AveragePooling2D(pool_size=(4,4))(net)
    net = Flatten()(net)
    net = Dense(units=3,activation="softmax")(net)

    model = Model(inputs=images,outputs=net)

    return model


input_shape = (32,32,3)
model = MiniModel(input_shape)
print(model.summary())
# model.compile(optimizer=Adam(0.001),loss="categorical_crossentropy",metrics=["accuracy"])
model.compile(optimizer='adam',loss="categorical_crossentropy",metrics=["accuracy"])
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
#image argumentation
from keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(rescale=1./255, shear_range=0.2, zoom_range=0.2, horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1./255)
#this is for winodows path

training_data = train_datagen.flow_from_directory(directory=r"C:\Users\mayan\Desktop\training\Train_Data", target_size=(32, 32), batch_size=8, class_mode='categorical')

#this is for winodows path

test_validation = test_datagen.flow_from_directory(r"C:\Users\mayan\Desktop\training\testing",
                                                        target_size=(32, 32),
                                                        batch_size=8,
                                                        class_mode='categorical')


# keras_backend.set_session(tf_debug.TensorBoardDebugWrapperSession(tf.Session(),"localhost:7000"))
#linux
# cb=TensorBoard(log_dir=("/home/mayank-s/PycharmProjects/Data_Science/output_graph/try4"))#,histogram_freq = 1, batch_size = 32,write_graph ="TRUE" )
# @windows
# cb=TensorBoard(log_dir=("C:/Users/mayank/Documents/Data_Science/output_graph/try4"))

model.fit_generator(generator=training_data,
                            steps_per_epoch=5,
                            epochs=5,
                            validation_data=test_validation,
                            validation_steps=5)

# model.fit_generator(
#         train_generator,
#         steps_per_epoch=nb_train_samples // batch_size,
#         epochs=epochs,
#         validation_data=validation_generator,
#         validation_steps=nb_validation_samples // batch_size)
# making new prediction
model.save_weights('first_try.h5')
import numpy as np
from keras.preprocessing import image

# test_image=image.load_img(path="C:/Users/mayank/Documents/Datasets/Cat_dogs/test1/509.jpg",target_size=(64,64))
#linux path
test_image=image.load_img(path=r"C:\Users\mayan\Desktop\training\testing\cat\cat.11.jpg",target_size=(32,32))
test_image=image.img_to_array(test_image)
test_image=np.expand_dims(test_image,axis=0)
result=model.predict(test_image)
# print (training_data.class_indices)
# print((int(result)))
