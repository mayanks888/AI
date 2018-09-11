from keras import applications
from keras.callbacks import TensorBoard
from keras.preprocessing.image import ImageDataGenerator
from keras import optimizers
from keras.models import Sequential, Model
from keras.layers import Dropout, Flatten, Dense
from keras import optimizers
from keras.models import Model
from keras import backend as keras_backend
from keras.callbacks import TensorBoard
from keras.layers import merge, Input
from tensorflow.python import debug as tf_debug
input_shape = Input(shape=(224, 224, 3))
inputshape=(224,224,3)
base_model=applications.vgg16.VGG16(weights = "imagenet", include_top=True,input_tensor=input_shape)
print (base_model.summary())
'''base_model.layers.pop()
out = base_model.add(Dense(1, activation='sigmoid', name='output'))
print (base_model.summary())'''
# ____________________________________________________
# image_input = Input(shape=(224, 224, 3))

# model = VGG16(input_tensor=image_input, include_top=True,weights='imagenet')

# model.summary()
num_classes=1
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
#------------------------------------------------------




'''for mylayer in base_model.layers:
    mylayer.trainable=False#this is done to set the weight as predefined
    new_sequential.add(mylayer)

print(type(new_sequential))

new_sequential.layers.pop()#remove my last layer

new_sequential.add(Dense(output_dim=1,activation='sigmoid'))


#now I'll bring my old parameter defined

new_sequential.compile(optimizer="adam",loss='binary_crossentropy',metrics=['accuracy'])

print (new_sequential.summary())'''


custom_vgg_model2.compile(loss='binary_crossentropy',optimizer='adadelta',metrics=['accuracy'])

#image argumentation
from keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(rescale=1./255,
                                    shear_range=0.2,
                                    zoom_range=0.2,
                                    horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1./255)
#this is for winodows path

training_data = train_datagen.flow_from_directory(directory="../../../../Datasets/cat_dogs/train",
                                                    target_size=(224, 224),
                                                    batch_size=8,
                                                    class_mode='binary')

#this is for winodows path

test_validation = test_datagen.flow_from_directory("../../../../Datasets/cat_dogs/test",
                                                        target_size=(224, 224),
                                                        batch_size=8,
                                                        class_mode='binary')


# keras_backend.set_session(tf_debug.TensorBoardDebugWrapperSession(tf.Session(),"localhost:7000"))
#linux
# cb=TensorBoard(log_dir=("/home/mayank-s/PycharmProjects/Data_Science/output_graph/try4"))#,histogram_freq = 1, batch_size = 32,write_graph ="TRUE" )
# @windows
cb=TensorBoard(log_dir=("C:/Users/mayank/Documents/Data_Science/output_graph/try5"))

custom_vgg_model2.fit_generator(generator=training_data,
                            steps_per_epoch=500,
                            epochs=5,
                            validation_data=test_validation,
                            validation_steps=5,callbacks=[cb])


# making new prediction

import numpy as np
from keras.preprocessing import image

# test_image=image.load_img(path="C:/Users/mayank/Documents/Datasets/Cat_dogs/test1/509.jpg",target_size=(64,64))
#linux path
test_image=image.load_img(path="../../../../Datasets/cat_dogs/test1/525.jpg",target_size=(224,224))
test_image=image.img_to_array(test_image)
test_image=np.expand_dims(test_image,axis=0)
result=custom_vgg_model2.predict(test_image)
print (training_data.class_indices)
print((int(result)))
