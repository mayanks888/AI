from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt
import numpy as np
import math
import numpy as np
import tensorflow as tf

'''
# this was used to create your own datasets (its a features in sklearn)
new_data=np.arange(0,10,2)
print(new_data)
data=make_blobs(50,2,2,random_state=75)
print(data)
feature=data[0]
labels=data[1]
x=new_data
y=(-.923*x)/.60083
plt.plot(x,y)
plt.scatter(feature[:,0],feature[:,1],c=labels)
plt.show()
wih=np.random.rand(3,3)-.5
wih = np.random.normal(0, pow(1, -0.5), (2,3))'''

# print (tf.__version__)#check tensor flow version
#this is normal tensorflow practise

'''con_data=tf.constant(.1,shape=(3,3),dtype=tf.float32)

sess= tf.Session()

new_data=sess.run(con_data)
print(new_data)'''

import cv2
import numpy as np
import tensorflow as tf
import sklearn.preprocessing
import pandas as pd
from sklearn.preprocessing import LabelEncoder,OneHotEncoder
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical
from keras.utils import np_utils
# linux
# data=pd.read_csv('../../../../Datasets/MNIST_data/train_image.csv')
# label=pd.read_csv('../../../../Datasets/MNIST_data/train_label.csv')

data=pd.read_csv('../../../../Datasets/MNIST_data/test_image.csv')
label=pd.read_csv('../../../../Datasets/MNIST_data/test_label.csv')
# winodws

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

# sess.run(init)

# X_train, X_test, y_train, y_test = train_test_split(new_image_input, y, test_size = .10, random_state = 4)#splitting data (no need if test data is present


# batch_x1, batch_y1 = tf.train.batch([feature_input, y_train], batch_size=5)
'''a=[13,1,2,2,6,5,5,5,5,6,69,55,6,656,58,9,4,3,5,5,5]
b=[13,1,2,2,6,5,5,5,5,6,69,55,6,656,58,9,9,3,5,5,5]
#_train=np.array(y_train)
y_train = zip(a, b)
batch_y1 = tf.train.batch(y_train, batch_size=3)#, enqueue_many=True, capacity=0)
# images, label_batch = tf.train.batch(
#         [feature_input, y_train],
#         batch_size=5,
#         num_threads=1,
#         capacity=4 + 3 * 5)
# batch_y1 = tf.train.batch(y_train, batch_size=5)
sess=tf.Session()

batch_y=sess.run(batch_y1)
# batch_x,batch_y=sess.run(images,label_batch)
print(batch_y)
# sess.run(train, feed_dict={x: batch_x, y_true: batch_y, hold_prob: 0.5})'''


from batchup import data_source

# Construct an array data source
ds = data_source.ArrayDataSource([scaled_input, y_train])
loop=0
# Iterate over samples, drawing batches of 64 elements in random order
for (batch_X, batch_y) in ds.batch_iterator(batch_size=100, shuffle=np.random.RandomState(12345)):
    loop+=1
    print(batch_X,'\n',batch_y)
    print (loop)