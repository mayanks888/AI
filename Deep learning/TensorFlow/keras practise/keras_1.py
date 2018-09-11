# import tensorflow.keras.models.Sequential  as tf
import tensorflow as tf

from keras.layers import Convolution2D
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.models import Sequential
from keras.layers import Flatten
from keras.layers import Dense
from keras import backend as keras_backend
from keras.callbacks import TensorBoard
from tensorflow.python import debug as tf_debug



# tf.keras.models.Sequential
#initialise sequential layer of cnn
myclassifier=Sequential()
# step 1: convolution layer

myclassifier.add(Conv2D(filters=32, kernel_size=(5, 5), strides=(1, 1),padding="Same",activation='relu',input_shape=(64,64,3)))
myclassifier.add(MaxPooling2D(pool_size=(2,2)))

myclassifier.add(Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1),padding="Same",activation='relu'))
myclassifier.add(MaxPooling2D(pool_size=(2,2)))
#convert matrix in single row matrix
myclassifier.add(Flatten())

#fully connected layer
myclassifier.add(Dense(output_dim=128, activation="relu"))

# output layer?
myclassifier.add(Dense(output_dim=1,activation='sigmoid'))

myclassifier.compile(optimizer="adam",loss='binary_crossentropy',metrics=['accuracy'])

print (myclassifier.summary())

#image argumentation
from keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(rescale=1./255,
                                    shear_range=0.2,
                                    zoom_range=0.2,
                                    horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1./255)
#this is for winodows path
'''training_data = train_datagen.flow_from_directory(directory="C:/Users/mayank/Documents/Datasets/Cat_dogs/train",
                                                    target_size=(64, 64),
                                                    batch_size=32,
                                                    class_mode='binary')'''
training_data = train_datagen.flow_from_directory(directory="../../../../Datasets/cat_dogs/train",
                                                    target_size=(64, 64),
                                                    batch_size=32,
                                                    class_mode='binary')

#this is for winodows path
'''test_validation = test_datagen.flow_from_directory("C:/Users/mayank/Documents/Datasets/Cat_dogs/test",
                                                        target_size=(64, 64),
                                                        batch_size=32,
                                                        class_mode='binary')'''
test_validation = test_datagen.flow_from_directory("../../../../Datasets/cat_dogs/test",
                                                        target_size=(64, 64),
                                                        batch_size=32,
                                                        class_mode='binary')


# keras_backend.set_session(tf_debug.TensorBoardDebugWrapperSession(tf.Session(),"localhost:7000"))

cb=TensorBoard(log_dir=("/home/mayank-s/PycharmProjects/Data_Science/output_graph/try4"))#,histogram_freq = 1, batch_size = 32,write_graph ="TRUE" )


myclassifier.fit_generator(generator=training_data,
                            steps_per_epoch=30,
                            epochs=10,
                            validation_data=test_validation,
                            validation_steps=200,callbacks=[cb])


# making new prediction

import numpy as np
from keras.preprocessing import image

# test_image=image.load_img(path="C:/Users/mayank/Documents/Datasets/Cat_dogs/test1/509.jpg",target_size=(64,64))
#linux path
test_image=image.load_img(path="../../Datasets/cat_dogs/test1/2.jpg",target_size=(64,64))
test_image=image.img_to_array(test_image)
test_image=np.expand_dims(test_image,axis=0)
result=myclassifier.predict(test_image)
print (training_data.class_indices)
print((int(result)))

# import tensorflow as tf
# from keras import backend as keras_backend
# from tensorflow.python import debug as tf_debug

# keras_backend.set_session(tf_debug.LocalCLIDebugWrapperSession(tf.Session()),"localhost:7000")
# keras_backend.set_session(tf_debug.TensorBoardDebugWrapperSession(tf.Session(),"localhost:7000"))

# Define your keras model, called "model".
# myclassifier.fit(...)  # This will break into the TFDBG CLI.
