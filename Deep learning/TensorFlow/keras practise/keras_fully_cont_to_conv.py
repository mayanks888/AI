from keras.models import Sequential
from keras.layers.convolutional import Convolution2D
from keras.engine import InputLayer
import keras
from tensorflow.python.keras.models import load_model
import tensorflow as tf
# import logging
# import tensorflow as tf
# from tensorflow.compat.v1 import graph_util
# from tensorflow.python.keras import backend as K
# from tensorflow.keras.models import  Sequential
# from tensorflow.keras.layers import InputLayer
# from tensorflow.keras.
# # from tensorflow import keras

def to_fully_conv(model):

    new_model = Sequential()

    input_layer = InputLayer(input_shape=(None, None, 3), name="input_new")

    new_model.add(input_layer)

    for layer in model.layers:

        if "Flatten" in str(layer):
            flattened_ipt = True
            f_dim = layer.input_shape

        elif "Dense" in str(layer):

            input_shape = layer.input_shape
            output_dim =  layer.get_weights()[1].shape[0]
            W,b = layer.get_weights()

            if flattened_ipt:
                shape = (f_dim[1],f_dim[2],f_dim[3],output_dim)
                new_W = W.reshape(shape)
                new_layer = Convolution2D(output_dim,
                                          (f_dim[1],f_dim[2]),
                                          strides=(1,1),
                                          activation=layer.activation,
                                          padding='valid',
                                          weights=[new_W,b])
                flattened_ipt = False

            else:
                shape = (1,1,input_shape[1],output_dim)
                new_W = W.reshape(shape)
                new_layer = Convolution2D(output_dim,
                                          (1,1),
                                          strides=(1,1),
                                          activation=layer.activation,
                                          padding='valid',
                                          weights=[new_W,b])


        else:
            new_layer = layer

        new_model.add(new_layer)

    return new_model

# tf.compat.v1.disable_eager_execution()

model = keras.applications.vgg16.VGG16()
# filename='/home/mayank_sati/codebase/python/camera/tensorflow/Food_DB_Demo (1)/h5_weoght/RX_model_food_with_dropout.h5'
# model = load_model(filename)
#
# model = keras.models.load_model(filename)

model.summary()

new_model = to_fully_conv(model)
new_model.summary()
