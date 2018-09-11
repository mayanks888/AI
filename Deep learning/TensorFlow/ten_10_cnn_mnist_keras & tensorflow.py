import cv2
import numpy as np
import tensorflow as tf
import sklearn.preprocessing
import pandas as pd
from sklearn.preprocessing import LabelEncoder,OneHotEncoder
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical
from keras.utils import np_utils
from batchup import data_source
from tensorflow.python import debug as tf_debug
# linux

# data=pd.read_csv('../../../../Datasets/MNIST_data/test_image (copy).csv')
# label=pd.read_csv('../../../../Datasets/MNIST_data/test_label (copy).csv')

data=pd.read_csv('../../../../Datasets/MNIST_data/train_image.csv')
label=pd.read_csv('../../../../Datasets/MNIST_data/train_label.csv')

test_feature=pd.read_csv('../../../../Datasets/MNIST_data/test_image.csv')
test_label=pd.read_csv('../../../../Datasets/MNIST_data/test_label.csv')

#winodws

# data=pd.read_csv(r"C:\Users\mayank\Documents\Datasets\MNIST_data\test_image.csv")
# label=pd.read_csv(r"C:\Users\mayank\Documents\Datasets\MNIST_data\test_label.csv")
# print (data.head())
# print(label.head())
# '____________________________________________________________'
# to read particular row in datasets
# reading in opencv
'''single_image= data.iloc[0]
single_image_array=np.array(single_image,dtype='uint8')
single_image_array=single_image_array.reshape(28,28)
cv2.imshow("image",single_image_array)
cv2.waitKey(0)
cv2.destroyAllWindows()'''
# '____________________________________________________________'

#dataset = pd.read_csv("SortedXmlresult_linux.csv")
feature_input = data.iloc[:,:].values
y = label.iloc[:,:].values
#to check for any null value in datasets
# print(data.isnull().sum())
# print(label.isnull().sum())
# ________________________________________________________________
# scaling features area image argumentation later we will add more image argumantation function
scaled_input = np.asfarray(feature_input/255.0)# * 0.99) +0.01

# this was used to categorise label if they are more than tow
# '_---______________________________________________' \
#one hot encode label data
y_train = np_utils.to_categorical(y, 10)
# print(y_test)
# '_---______________________________________________'
# new_image_input,y=shuffle(image_list_array,y,random_state=4)#shuffle data (good practise)

# X_train, X_test, y_train, y_test = train_test_split(new_image_input, y, test_size = .10, random_state = 4)#splitting data (no need if test data is present
# '_---______________________________________________'
# scaling and one hot encode applied on a test datasets
feature_test = test_feature.iloc[:,:].values
label_test = test_label.iloc[:,:].values
scaled_test = np.asfarray(feature_test/255.0)
y_test = np_utils.to_categorical(label_test, 10)
# '_---______________________________________________'


# ### Placeholders
x = tf.placeholder(tf.float32,shape=[None,784])
y_true = tf.placeholder(tf.float32,shape=[None,10])
# ### Layers
x_image = tf.reshape(x,[-1,28,28,1])

conv_1_layer = tf.keras.layers.Conv2D(filters=32,kernel_size=(6, 6),strides=(1, 1), activation='relu', padding='same', name='block1_conv1')(x_image)
max_pool_1 = tf.keras.layers.MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool')(conv_1_layer)

conv_2_layer = tf.keras.layers.Conv2D(filters=64,kernel_size=(6, 6),strides=(1, 1), activation='relu', padding='same', name='block2_conv1')(max_pool_1)
max_pool_2 = tf.keras.layers.MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool')(conv_2_layer)

flat_layer=tf.keras.layers.Flatten(name='flatten')(max_pool_2)

first_dense=tf.keras.layers.Dense(1024, activation="relu",name="flat1")(flat_layer)

hold_prob = tf.placeholder(tf.float32)#this made to pass the user defined value for dropout probabilty you could have also used contant value
full_one_dropout = tf.nn.dropout(first_dense,keep_prob=hold_prob)

output_layer =tf.keras.layers.Dense(10, activation='relu', name='output1')(full_one_dropout)
# _______________________________________________________
# input_tensor = tf.keras.Input(shape=(28,28,1))
#
# custom_vgg_model2 = tf.keras.Model(x,output_layer1)
#
# # Trying different things
# # freeze all the layers except the dense layers
#
# for layer in custom_vgg_model2.layers[:-3]:
# 	layer.trainable = False
#
# output_layer=custom_vgg_model2.output
 # ---------------------------------------------------------------------------
# first method
softmax_output=tf.nn.softmax(logits=output_layer)
entropy_loss_per_row=(y_true * tf.log(softmax_output))#formula for cross entropy
# formula for cross entropy (L=−∑i=0kyilnp^i)
sum_loss_per_row = (-tf.reduce_sum(entropy_loss_per_row,axis=1))#"-" you have to check(axis=1 since I was row wise sum)
# loss_per_row = (-tf.reduce_sum(y_true * tf.log(softmax_output),[1]))#formula for cross entropy
cross_entropy_mean=tf.reduce_mean(sum_loss_per_row)
# -------------------------------------------------------------------------------
# Second method(direct)
# cross_entropy_mean = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_true,logits=y_pred))
# -------------------------------------------------------------------------------
# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

# optimisation: this is to learning check differnt optimiser
optimizer = tf.train.AdamOptimizer(learning_rate=0.0001)#Inistising your optimising functions
train = optimizer.minimize(cross_entropy_mean)#this will trigger backward propogation for learning
# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
train_matches = tf.equal(tf.argmax(output_layer, 1), tf.argmax(y_true, 1))
acc = tf.reduce_mean(tf.cast(train_matches, tf.float32))

    # _____


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
    ds = data_source.ArrayDataSource([scaled_input, y_train])
    # Iterate over samples, drawing batches of 64 elements in random order
    for (batch_X, batch_y) in ds.batch_iterator(batch_size=1000, shuffle=True):#shuffle true will randomise every batch
    # for (batch_X, batch_y) in ds.batch_iterator(batch_size=1000, shuffle=np.random.RandomState(12345)):
        _,accuracy=sess.run([train,acc], feed_dict={x: batch_X, y_true: batch_y, hold_prob: 0.5})
        print("the train accuracy is:", accuracy)
         # ________________________________________________________________________

    # batch_x1, batch_y1 = tf.train.batch([feature_input, y_train], batch_size=50)
    # batch_x,batch_y=sess.run(batch_x1,batch_y1)
    # sess.run(train, feed_dict={x: batch_x, y_true: batch_y, hold_prob: 0.5})

    if loop % 1 == 0:
        print('Currently on step {}'.format(loop))
        print('Accuracy is:')
        # Test the Train Modelsess.run(train, feed_dict={x: batch_x, y_true: batch_y, hold_prob: 0.5})
        matches = tf.equal(tf.argmax(output_layer, 1), tf.argmax(y_true, 1))

        acc = tf.reduce_mean(tf.cast(matches, tf.float32))

        print(sess.run(acc, feed_dict={x: scaled_test, y_true: y_test, hold_prob: 1.0}))


