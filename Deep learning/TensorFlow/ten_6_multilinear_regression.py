import  tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def normalise_data(x):#nice way to normalise data
    mean_val=np.mean(x)
    std=np.std(x)
    new_x=(x-mean_val)/std
    return new_x


boston = tf.contrib.learn.datasets.load_dataset('boston')#loading datasets
x_train,y_train=boston.data,boston.target #selecting whole data as feature
x_train=normalise_data(x_train)
add_new_column=np.c_[np.ones(506),x_train]#new column added for adding bias(to add 1 in all row for first clumn )
y_train=np.reshape(y_train,[506,1])#reshape t train so as freely calculate together
n_sample=len(x_train)
m_coumn=13+1# bias#no of feature bias

x=tf.placeholder(dtype=tf.float32,name="X",shape=[None,14])#logicall i we don't know now of rows in placeholder
y=tf.placeholder(dtype=tf.float32,name="y",shape=[None,1])#(this will generae a placeholder of 506 rows and 1 column)

#declaring variables
w=tf.Variable(initial_value=tf.random_normal([m_coumn,1]))#y=wx+b(check rule for matrix multiplication before assigning values



#node struction or computations
#linear regression model with matrix multiplications
y_predict=tf.matmul(x,w)#all matrix calcualiton together
#defining error/loss functions

loss=tf.reduce_mean(tf.square(y-y_predict,name="loss"))#add loss of all 506 rows

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
total_loss=0
for i in range(epoch):
    #y_value=sess.run(y_predict,feed_dict=({x:x_train,y:y_train}))
    _,los=sess.run([optimiser,loss],feed_dict={x:add_new_column,y:y_train})#feeding to placeholder
    total_loss+=los
    total.append(total_loss/n_sample)
    print('epoch{0}:loss{1}'.format(i, total_loss/n_sample))#this is good method to print with description check this out
    final_w=sess.run(w)
    print(final_w)
    #los=sess.run(loss,feed_dict=({x:x_n_val,y:y_val}))

'''

    total_loss=0
    for x_val,y_val in zip(x_train,y_train):
        los=sess.run(loss,feed_dict=({x:x_val,y:y_val}))
        #_,los=sess.run([optimiser,loss],feed_dict={x:x_val,y:y_val})#feeding to placeholder
       # total_loss+=los
        #total.append(total_loss/n_sample)
        #print('epoch{0}:loss{1}'.format(i, total_loss/n_sample))#this is good method to print with description check this out

        final_w,final_b=sess.run([w,b])

y_final_predict=final_w*x_train+final_b
plt.plot(x_train,y_train,'bo',label="real data")#bo is used to print it in scatterd format

plt.plot(x_train,y_final_predict,'r',label='[predicted data')
plt.legend()
plt.show()'''






