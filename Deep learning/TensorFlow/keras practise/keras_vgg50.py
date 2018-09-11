from keras import applications
from keras.callbacks import TensorBoard
from keras.preprocessing.image import ImageDataGenerator
from keras import optimizers
from keras.models import Sequential, Model
from keras.layers import Dropout, Flatten, Dense
from keras import optimizers
from keras.applications.imagenet_utils import preprocess_input
from keras.applications.imagenet_utils import decode_predictions


inputshape=(224,224,3)
base_model=applications.resnet50.ResNet50(weights = "imagenet", include_top=True,input_shape=inputshape)
print (base_model.summary())

# last_layer=base_model.layers('fc2')
#now we are used to work on a sequantial layer in keras aso we will conver vgg16 model into sequantal model and use as our train model

# making new prediction

import numpy as np
from keras.preprocessing import image

# test_image=image.load_img(path="C:/Users/mayank/Documents/Datasets/Cat_dogs/test1/509.jpg",target_size=(64,64))
#linux path
test_image=image.load_img(path="../../../Datasets/cat_dogs/test1/241.jpg",target_size=(224,224))
test_image=image.img_to_array(test_image)
test_image=np.expand_dims(test_image,axis=0)
test_image=preprocess_input(test_image)#process image input as per the model used(caffe use different notation of image)
result=base_model.predict(test_image)
# print (training_data.class_indices)
print((len(result)))
print('prediction',decode_predictions(result))
