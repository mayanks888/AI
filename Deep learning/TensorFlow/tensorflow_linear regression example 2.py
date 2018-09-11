import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
x_data=np.linspace(0,10,100000)#creting data for x
noise=np.random.randn(len(x_data))
y_data=(0.5+x_data)+5+noise#y=mx+c
#
# x_df=pd.DataFrame(data=x_data,columns=["X_data"])
# y_df=pd.DataFrame(data=y_data,columns=['Y_data'])
# mydata=pd.DataFrame(data=[x_data,y_data],columns=(["X_data",'Y_data']))
# mydata=pd.DataFrame(data=([x_data,y_data]))
mydata=pd.DataFrame({'X_data':x_data,'Y_data':y_data})#create dataframe from list of data
my_sample=mydata.sample(n=300)
# my_sample.plot(kind='scatter',x='X_data',y='Y_data')
# plt.scatter(my_sample['X_data'],my_sample['Y_data'])
# plt.show()

batch_size=8
m_slope=tf.Variable(initial_value=0.52)
bias=tf.Variable(initial_value=.21)

xph=tf.placeholder(tf.float32,[batch_size])
yph=tf.placeholder(tf.float32,[batch_size])

y_model=m_slope*xph+bias
error=tf.reduce_sum(tf.square(yph-y_model))

optimiser=tf.train.GradientDescentOptimizer(learning_rate=.003)
train=optimiser.minimize(error)
init=tf.global_variables_initializer()

sess=tf.Session()
sess.run(init)
batches_run=200

for i in range(batches_run):
    rand_int=np.random.randint(len(x_data),size=batch_size)
    feed={xph:x_data[rand_int],yph:y_data[rand_int]}
    sess.run(train,feed_dict=feed)
    
 
    
model_m,model_bias=sess.run([m_slope,bias])
 
        # this is for test datasets accuracy
xtest = np.linspace(-1, 11, 100)
#print(sess.run(error))
y_predict = model_m * xtest +model_bias
plt.plot(xtest, y_predict)
plt.scatter(my_sample['X_data'],my_sample['Y_data'])
plt.show()


