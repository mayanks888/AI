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


def create_weight(shape,var_name):
    init_random_dist = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(init_random_dist,name=var_name)
#we have to create this many function because of normal and convolution layers
def create_bias(shape,var_name):
    init_bias_vals = tf.constant(0.1, shape=shape)
    return tf.Variable(init_bias_vals,name=var_name)

def create_convd(x,W):
    return (tf.nn.conv2d(input=x,filter=W,strides=[1,1,1,1],padding="SAME"))

def max_pool(x):
    return tf.nn.max_pool(value=x, ksize=[1, 2, 2, 1],strides=[1, 2, 2, 1], padding='SAME')

def convolutional_layer(input_x, shape,var_name):
    W = create_weight(shape,var_name=(var_name +"_weight"))
    b = create_bias([shape[3]],var_name=(var_name +"_bias"))#keep in mind why it was initialise like this
    # bias was intiaslise because for one layer the bias remains same and since we are using 32 filer so shape address [3] will contain 32 value of filter or 32 bias
    return tf.nn.relu(create_convd(input_x, W) + b)

def normal_full_layer(input_layer, output_layer_size,var_name):
    input_size = int(input_layer.get_shape()[1])#this is used to get the column no from shape of fully connected layers
    W = create_weight([input_size, output_layer_size],var_name=(var_name +"_weight"))
    b = create_bias([output_layer_size],var_name=(var_name +"_bias"))#remember bias are always equal to the no of output layer in fully connected layer
    return tf.matmul(input_layer, W) + b

# ### Placeholders
x = tf.placeholder(tf.float32,shape=[None,784])
y_true = tf.placeholder(tf.float32,shape=[None,10])
# ### Layers
x_image = tf.reshape(x,[-1,28,28,1])

# create first layer
conv1=convolutional_layer(input_x=x_image,shape=(6,6,1,32),var_name="ConVar_1")
max_pl1=max_pool(conv1)#max pool will divide the layer into half

# second LAYER
conv2=convolutional_layer(input_x=max_pl1,shape=(6,6,32,64),var_name="Convar_2")
max_pl2=max_pool(conv2)


#fully connected laeyr
convo_2_flat = tf.reshape(max_pl2,[-1,7*7*64])# this will convert the ma pool into (1 row and 3136 column)
#in defining reshape(item,-1,4) means that convert the item into 4 column and tf is calculate itself amount of rows
full_layer_one = tf.nn.relu(normal_full_layer(convo_2_flat,1024,var_name="full_1"))#now the output will be 1 row and 1024 column



# NOTE THE PLACEHOLDER HERE!
hold_prob = tf.placeholder(tf.float32)#this made to pass the user defined value for dropout probabilty you could have also used contant value
full_one_dropout = tf.nn.dropout(full_layer_one,keep_prob=hold_prob)

# last layer for classifications
y_pred = normal_full_layer(full_one_dropout,10,var_name="full_last")#remember I have not used softmax or relu in the output of late layer
#but we will apply different approach for better learning

#  +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# now the layer construction is complete next steps is to find the loss , optimation for learning

# Two ways are there to define the cross entropy loss ,one is direct and another is by going through all the function we will be going through both
# ---------------------------------------------------------------------------
# first method
softmax_output=tf.nn.softmax(logits=y_pred)
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
train_matches = tf.equal(tf.argmax(y_pred, 1), tf.argmax(y_true, 1))
acc = tf.reduce_mean(tf.cast(train_matches, tf.float32))

    # _____


# intialiasing all variable

init=tf.global_variables_initializer()

epochs=20

sess=tf.Session()
sess.run(init)
#sess = tf_debug.TensorBoardDebugWrapperSession(sess, "localhost:7000")
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
        matches = tf.equal(tf.argmax(y_pred, 1), tf.argmax(y_true, 1))

        acc = tf.reduce_mean(tf.cast(matches, tf.float32))

        print(sess.run(acc, feed_dict={x: scaled_test, y_true: y_test, hold_prob: 1.0}))














'''for i in range(epochs):
    batch_x, batch_y = mnist.train.next_batch(batch)
    # actual_weight, actual_bias=sess.run([weight,bias])
    # print (actual_weight)
    layer_output,mycross_entropy, my_softmax_val, all_cross_ent, _ = sess.run([pred_value,loss, pred_value,total_cross_entropy, optimizer],
                                                                 feed_dict={input_matrix: batch_x,output_matrix: batch_y,drop_out_val:drop_out})

    # feed_dicy=({input_matrix:input_data1, bias: bias['bc1']})
    # conv1_val=sess.run(create_conv2d)
    my_prediction_train = tf.equal(tf.argmax(my_softmax_val, 1), tf.argmax(batch_y,1))  # comparing max index of softmax output and true y output matrix
    # cast convert true to 1 an false to 0
    accuray = tf.reduce_mean(tf.cast(my_prediction_train, tf.float32))
    prediction_val, final_acuracy = (sess.run([my_prediction_train, accuray]))
    print ('epoch no ', i)
    #print("predicted value are", prediction_val)
    print('accuracy ', final_acuracy * 100)'''
