from keras import applications
from keras.callbacks import TensorBoard
from keras.preprocessing.image import ImageDataGenerator
from keras import optimizers
from keras.models import Sequential, Model
from keras.layers import Dropout, Flatten, Dense
from keras import optimizers
from keras.applications.imagenet_utils import preprocess_input
from keras.applications.imagenet_utils import decode_predictions
import tensorflow as tf
from keras import backend as keras_backend
from keras.callbacks import TensorBoard
from tensorflow.python import debug as tf_debug

inputshape=(224,224,3)
base_model=tf.keras.applications.VGG16(weights = "imagenet", include_top=True,input_shape=inputshape)
# base_model=applications.vgg16.VGG16(weights = "imagenet", include_top=True,input_shape=inputshape)
print (base_model.summary())

# last_layer=base_model.layers('fc2')
#now we are used to work on a sequantial layer in keras aso we will conver vgg16 model into sequantal model and use as our train model
'''new_sequential=Sequential()
print(type(base_model))






#now I'll bring my old parameter defined

new_sequential.compile(optimizer="adam",loss='binary_crossentropy',metrics=['accuracy'])

print (new_sequential.summary())

#image argumentation
from keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(rescale=1./255,
                                    shear_range=0.2,
                                    zoom_range=0.2,
                                    horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1./255)
#this is for winodows path

training_data = train_datagen.flow_from_directory(directory="../../../Datasets/cat_dogs/train",
                                                    target_size=(224, 224),
                                                    batch_size=8,
                                                    class_mode='binary')

#this is for winodows path

test_validation = test_datagen.flow_from_directory("../../../Datasets/cat_dogs/test",
                                                        target_size=(224, 224),
                                                        batch_size=8,
                                                        class_mode='binary')


# keras_backend.set_session(tf_debug.TensorBoardDebugWrapperSession(tf.Session(),"localhost:7000"))

cb=TensorBoard(log_dir=("/home/mayank-s/PycharmProjects/Data_Science/output_graph/try4"))#,histogram_freq = 1, batch_size = 32,write_graph ="TRUE" )


new_sequential.fit_generator(generator=training_data,
                            steps_per_epoch=5,
                            epochs=5,
                            validation_data=test_validation,
                            validation_steps=5,callbacks=[cb])'''


# making new prediction

import numpy as np
from keras.preprocessing import image

# test_image=image.load_img(path="C:/Users/mayank/Documents/Datasets/Cat_dogs/test1/509.jpg",target_size=(64,64))
#linux path
test_image=image.load_img(path="../../../../Datasets/cat_dogs/test1/506.jpg",target_size=(224,224))
test_image=image.img_to_array(test_image)
test_image=np.expand_dims(test_image,axis=0)
result=base_model.predict(test_image)
# print (training_data.class_indices)
print(((result)))
print('prediction',decode_predictions(result))
