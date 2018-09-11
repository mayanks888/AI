import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.python import debug as tf_debug
xlabel=np.linspace(0,10,10)+np.random.uniform(-1.5,1.5,10)
ylabel=np.linspace(0,10,10)+np.random.uniform(-1.5,1.5,10)

m=tf.Variable(initial_value=.44)
b=tf.Variable(initial_value=.87)
error=0
note=tf.Variable(initial_value=35)
for x,y in zip(xlabel,ylabel):
    y_val=m*x+b
    error+=(y_val-y)**2
    


optimiser=tf.train.GradientDescentOptimizer(learning_rate=.001)
train_data=optimiser.minimize(error,note)
init=tf.global_variables_initializer()

sess=tf.Session()
sess.run(init)
sess = tf_debug.TensorBoardDebugWrapperSession(sess, "localhost:7000")
training_steps=5

for i in range(training_steps):
    sess.run(train_data)
    final_slope,final_intercept=sess.run([m,b])
    print (sess.run([m,b,note]))

    #this is for test datasets accuracy
    xtest=np.linspace(-1,11,10)
    y_predict=final_slope*xtest+final_intercept
    plt.plot(xtest,y_predict)
    plt.scatter(xlabel,ylabel)
    plt.show()
    print (sess.run(error))
    

