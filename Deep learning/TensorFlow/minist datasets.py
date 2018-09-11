import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

'''#-------------------------------------------------
data=pd.read_csv("mnist_train_100.csv")
print data.head()


# print data.iloc[2:3,:].values
part_data= data.iloc[[2]]
print part_data[:1].head()
image_array = np.asfarray( part_data [1:]).reshape((28,28))
plt.imshow(image_array,cmap='Greys', interpolation="None")
plt.show()
print part_data.shape
#-------------------------------------------------'''

data_file = open("mnist_train_100.csv", 'r')
data_list = data_file .readlines()
data_file .close()
print data_list[1]
all_values = data_list [1].split(',')

'''The first thing we need to do is to rescale the input colour values from the larger range 0 to 255to the much smaller range 0.01 1.0.We’ve deliberately chosen 0.01 as the lower end of the
range to avoid the problems we saw earlier with zero valued inputs because they can artificiallykill weight updates. We don’t have to choose 0.99 for the upper end of the input because wedon’t need
to avoid 1.0 for the inputs. It’s only for the outputs that we should avoid theimpossible to reach 1.0.

Dividing the raw inputs which are in the range 0255 by 255 will bring them into the range 01. We then need to multiply by 0.99 to bring them into the range 0.0 0.99.
We then add 0.01 to shift them up to the desired range 0.01 to 1.00. The following Python code shows this in action:'''


scaled_input = (np.asfarray(all_values[1:]) / 255.0 * 0.99) +0.01
image_array = np.asfarray(scaled_input.reshape(28,28))
print image_array
plt.imshow( image_array , cmap='Greys',interpolation='None')
plt.show()

