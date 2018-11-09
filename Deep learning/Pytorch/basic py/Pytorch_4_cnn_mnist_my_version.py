import cv2
import numpy as np
# import tensorflow as tf

import pandas as pd

from keras.utils import np_utils
from batchup import data_source

# from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable

# ___________________
data=pd.read_csv('/home/mayank-s/PycharmProjects/Datasets/MNIST_data/train_image.csv')
label=pd.read_csv('/home/mayank-s/PycharmProjects/Datasets/MNIST_data/train_label.csv')

test_feature=pd.read_csv('/home/mayank-s/PycharmProjects/Datasets/MNIST_data/test_image.csv')
test_label=pd.read_csv('/home/mayank-s/PycharmProjects/Datasets/MNIST_data/test_label.csv')



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
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=5,padding=2,stride=1)#this is nothing with(1=channel,layer=32,padding=1 =same size of input)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=5,padding=2)
        self.mp = nn.MaxPool2d(2)
        self.fc = nn.Linear(3136, 10)
        self.bn1 = nn.BatchNorm2d(32)

    def forward(self, x):
        in_size = x.size(0)
        # x = x.float()
        first_layer=self.conv1(x)
        x = F.relu(self.mp(first_layer))
        x = F.relu(self.mp(self.conv2(x)))
        x = x.view(in_size, -1)  # flatten the tensor
        x = self.fc(x)
        # prob=nn.Softmax(x)
        # print(prob.data[0])
        prob=F.log_softmax(x)
        # prob=prob.long()
        return prob


model = Net()
# model.cuda()
model = model.double()
criterea = torch.nn.CrossEntropyLoss()#this include softmax and cross entropy
criterea=torch.nn.BCELoss()#this is for binary cross entropy when you have only binary output (0/1 or True/False)
# criterea=torch.nn.MSELoss(size_average=False)#cross entropy loss
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.5)



run=0
epochs=10
for loop in range(epochs):
    # ______________________________________________________________________
    # my batch creater
    # Construct an array data source
    ds = data_source.ArrayDataSource([scaled_input, y])
    # Iterate over samples, drawing batches of 64 elements in random order
    for (data, target) in ds.batch_iterator(batch_size=1, shuffle=True):#shuffle true will randomise every batch
        new_data=data.reshape(-1,1,28,28)
        cool_data=new_data[0]
        # cv2.imshow("image",np.reshape(cool_data,newshape=(28,28,1)))
        # cv2.waitKey(500)
        # cv2.destroyAllWindows()
        # data, target=torch.from_numpy(new_data).double(),torch.from_numpy(target).double()
        # inputs, labels = Variable(inputs.cuda()), Variable(labels.cuda())#this is for cuda gpu
        data, target = torch.tensor(new_data), torch.tensor(target).long()
        optimizer.zero_grad()
        output = model(data)

        #this is to find accuracy
        out=torch.max(output, 1)[1]
        match=torch.eq(out, target.squeeze())
        all1=torch.sum(match)
        acc=all1.item()/len(match)

        #  _, predicted = torch.max(target.data, 1)
        # total += labels.size(0)
        # correct += (predicted == target).sum().item()
        # print('Accuracy of the network on the 10000 test images: %d %%' % (100 * correct / total))
        #
        loss = F.nll_loss(output, target.squeeze())                               
        #loss = criterea(output,torch.max(target, 1)[0])#max returns two values it means its return max value in each row and[0 ] means its take first return value
        #loss = criterea(output, target.squeeze())
        loss.backward()
        optimizer.step()
        run+=1 

        print('epoch is', run)
        print('train loss is',loss.item())
        print("train  accuracy is ",acc)

        if run % 10 == 0:
            ds = data_source.ArrayDataSource([scaled_test, label_test])
                    # Iterate over samples, drawing batches of 64 elements in random order
            for (test_data, test_target) in ds.batch_iterator(batch_size=100,shuffle=True):  # shuffle true will randomise every batch
                test_data=test_data.reshape(-1,1,28,28)
                # cv2.imshow("image",np.reshape(test_data[5],newshape=(28,28,1)))
                # cv2.waitKey(500)
                # cv2.destroyAllWindows()
                test_data1, target = torch.tensor(test_data), torch.tensor(test_target).long()

                optimizer.zero_grad()
                output = model(test_data1)

                out=torch.max(output, 1)[1]
                match=torch.eq(out, target.squeeze())
                all1=torch.sum(match)
                acc=all1.item()/len(match)
                print("test accuracy is ",acc)
                break
            # print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
            #     loop, batch_idx * len(data), len(train_loader.dataset),
            #            100. * batch_idx / len(train_loader), loss.data[0]))

        
         # ________________________________________________________________________

    # batch_x1, batch_y1 = tf.train.batch([feature_input, y_train], batch_size=50)
    # batch_x,batch_y=sess.run(batch_x1,batch_y1)
    # # sess.run(train, feed_dict={x: batch_x, y_true: batch_y, hold_prob: 0.5})
    #
    # if loop % 1 == 0:
    #     print('Currently on step {}'.format(loop))
    #     print('Accuracy is:')
    #     # Test the Train Modelsess.run(train, feed_dict={x: batch_x, y_true: batch_y, hold_prob: 0.5})
    #     matches = tf.equal(tf.argmax(output_layer, 1), tf.argmax(y_true, 1))
    #
    #     acc = tf.reduce_mean(tf.cast(matches, tf.float32))
    #
    #     print(sess.run(acc, feed_dict={x: scaled_test, y_true: y_test, hold_prob: 1.0}))
    #
    #

