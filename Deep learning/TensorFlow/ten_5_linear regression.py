import  tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def normalise_data(x):#nice way to normalise data
    mean_val=np.mean(x)
    std=np.std(x)
    new_x=(x-mean_val)/std
    return new_x

# data=tf.contrib.learn.datasets.load_datasets('boston')
boston = tf.contrib.learn.datasets.load_dataset('boston')#loading datasets
x_train,y_train=boston.data[:,5],boston.target #selecting only colum 5 as x train data
x_train=normalise_data(x_train)
n_sample=len(x_train)

#creating tensorflow graph()

x=tf.placeholder(dtype=tf.float32,name="X")#x and y
y=tf.placeholder(dtype=tf.float32,name="y")

#declaring variables
w=tf.Variable(initial_value=.47)#y=wx+b
b=tf.Variable(initial_value=0.0)

#node struction or computations

y_predict=w*x+b#linear regression model

#defining error/loss functions

loss=tf.square(y-y_predict,name="loss")

#now optimsation for error
# optimiser=tf.train.GradientDescentOptimizer(learning_rate=.003)
# optimiser.minimize(loss)
optimiser=tf.train.GradientDescentOptimizer(learning_rate=.003).minimize(loss)#this is optimiese

#declaring my variable used
init=tf.global_variables_initializer()

#begin graph session

sess=tf.Session()
sess.run(init)
total=[]
epoch=2#defining epochs
for i in range(epoch):
    total_loss=0
    for x_val,y_val in zip(x_train,y_train):
        _,los=sess.run([optimiser,loss],feed_dict={x:x_val,y:y_val})#feeding to placeholder
        total_loss+=los
        total.append(total_loss/n_sample)
        print('epoch{0}:loss{1}'.format(i, total_loss/n_sample))#this is good method to print with description check this out

        final_w,final_b=sess.run([w,b])

y_final_predict=final_w*x_train+final_b
plt.plot(x_train,y_train,'bo',label="real data")#bo is used to print it in scatterd format

plt.plot(x_train,y_final_predict,'r',label='[predicted data')
plt.legend()
plt.show()






