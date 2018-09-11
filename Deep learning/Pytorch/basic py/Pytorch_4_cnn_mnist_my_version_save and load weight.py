'''Case # 1: Save the model to use it yourself for inference: You save the model, you restore it, and then you change the model to evaluation mode. This is done because you usually have BatchNorm and Dropout layers that by default are in train mode on construction:

torch.save(model.state_dict(), filepath)

#Later to restore:
model.load_state_dict(torch.load(filepath))
model.eval()

Case # 2: Save model to resume training later: If you need to keep training the model that you are about to save, you need to save more than just the model. You also need to save the state of the optimizer, epochs, score, etc. You would do it like this:

state = {
    'epoch': epoch,
    'state_dict': model.state_dict(),
    'optimizer': optimizer.state_dict(),
    ...
}
torch.save(state, filepath)

To resume training you would do things like: state = torch.load(filepath), and then, to restore the state of each individual object, something like this:

model.load_state_dict(state['state_dict'])
optimizer.load_state_dict(stata['optimizer'])

Since you are resuming training, DO NOT call model.eval() once you restore the states when loading.

Case # 3: Model to be used by someone else with no access to your code: In Tensorflow you can create a .pb file that defines both the architecture and the weights of the model. This is very handy, specially when using Tensorflow serve. The equivalent way to do this in Pytorch would be:

torch.save(model, filepath)

# Then later:
model = torch.load(filepath)

This way is still not bullet proof and since pytorch is still undergoing a lot of changes, I wouldn't recommend it.'''
import cv2
import numpy as np
import pandas as pd
from keras.utils import np_utils
from batchup import data_source

import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable

# ___________________
data=pd.read_csv('../../../Datasets/MNIST_data/train_image.csv')
label=pd.read_csv('../../../Datasets/MNIST_data/train_label.csv')

test_feature=pd.read_csv('../../../Datasets/MNIST_data/test_image.csv')
test_label=pd.read_csv('../../../Datasets/MNIST_data/test_label.csv')

PATH = '/home/mayank-s/PycharmProjects/Datasets/pytorch_model_save/mytraining.pt'

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
        self.conv1 = nn.Conv2d(1, 32, kernel_size=5,padding=2)#this is nothing with(1=channel,layer=32,padding=1 =same size of input)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=5,padding=2)
        self.mp = nn.MaxPool2d(2)
        self.fc = nn.Linear(3136, 10)

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
##############################################
# #this is for loading model saved parameter you dont need it while training or  you can resume you training with this
# #model = torch.load(PATH + str(7))
#2nd method
# state = torch.load(PATH + str(2))
# model.load_state_dict(state['state_dict'])
#
# print ("Epochs : {ep} -Training accuracy: {ta}".format(ep=state['epoch'],ta=state['best_accuracy']))
# # print ("Epochs : {ep} -Training accuracy: {ta} - optimiser :{so}".format(ep=state['epoch'],ta=state['best_accuracy'],so=state['optimizer']))
# ##############################################
# model.cuda()
model = model.double()
criterea = torch.nn.CrossEntropyLoss()#this include softmax and cross entropy
# criterea=torch.nn.MSELoss(size_average=False)#cross entropy loss
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.5)

# optimizer.load_state_dict(state['optimizer'])

run=0
epochs=10

def train(loop):
    run=0
    ds = data_source.ArrayDataSource([scaled_input, y])
    # Iterate over samples, drawing batches of 64 elements in random order
    for (data, target) in ds.batch_iterator(batch_size=1000, shuffle=True):  # shuffle true will randomise every batch
        new_data = data.reshape(-1, 1, 28, 28)
        cool_data = new_data[0]
        # cv2.imshow("image",np.reshape(cool_data,newshape=(28,28,1)))
        # cv2.waitKey(500)
        # cv2.destroyAllWindows()
        # data, target=torch.from_numpy(new_data).double(),torch.from_numpy(target).double()
        # inputs, labels = Variable(inputs.cuda()), Variable(labels.cuda())#this is for cuda gpu
        data, target = torch.tensor(new_data), torch.tensor(target).long()
        optimizer.zero_grad()
        output = model(data)

        # this is to find accuracy
        out = torch.max(output, 1)[1]
        match = torch.eq(out, target.squeeze())
        all1 = torch.sum(match)
        acc = all1.item() / len(match)

        #  _, predicted = torch.max(target.data, 1)
        # total += labels.size(0)
        # correct += (predicted == target).sum().item()
        # print('Accuracy of the network on the 10000 test images: %d %%' % (100 * correct / total))
        #
        loss = F.nll_loss(output, target.squeeze())
        # loss = criterea(output,torch.max(target, 1)[0])#max returns two values it means its return max value in each row and[0 ] means its take first return value
        # loss = criterea(output, target.squeeze())
        loss.backward()
        optimizer.step()
        run += 1
        print("Epochs : {ep} - Run Cycle : {rc} - Training Loss : {tl} - Training accuracy: {ta} ".format(ep=loop, rc=run,
                                                                                                        tl=loss.item(), ta=acc))
    # 1st method to save your complete model
    # PATH = '/home/mayank-s/PycharmProjects/Datasets/pytorch_model_save/mytraining.pt'+str(loop)
    # torch.save(model, PATH)
    # state = {'epoch': loop,'state_dict': model.state_dict(),'optimizer': optimizer.state_dict(),'best_accuracy': acc}
    # torch.save(state, PATH+str(loop))

def test():
    ds = data_source.ArrayDataSource([scaled_test, label_test])
    # Iterate over samples, drawing batches of 64 elements in random order
    for (test_data, test_target) in ds.batch_iterator(batch_size=1000,
                                                      shuffle=True):  # shuffle true will randomise every batch
        test_data = test_data.reshape(-1, 1, 28, 28)
        # cv2.imshow("image",np.reshape(test_data[5],newshape=(28,28,1)))
        # cv2.waitKey(500)
        # cv2.destroyAllWindows()
        test_data1, target = torch.tensor(test_data), torch.tensor(test_target).long()

        optimizer.zero_grad()
        output = model(test_data1)

        out = torch.max(output, 1)[1]
        match = torch.eq(out, target.squeeze())
        all1 = torch.sum(match)
        acc = all1.item() / len(match)
        print("test accuracy is ", acc)
        break


for loop in range(1, epochs):
    train(loop)
    test()

'''for loop in range(1,epochs):
    run = 0
    # ______________________________________________________________________
    # my batch creater
    # Construct an array data source
    ds = data_source.ArrayDataSource([scaled_input, y])
    # Iterate over samples, drawing batches of 64 elements in random order
    for (data, target) in ds.batch_iterator(batch_size=1000, shuffle=True):#shuffle true will randomise every batch
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
        print("Epochs : {ep} - Run Cycle : {rc} - Training Loss : {tl} - Training accuracy: {ta} ".format(ep=loop,rc=run,tl=loss.item(),ta=acc))

        # print('epoch is', epochs)
        # print('train loss is',loss.item())
        # print("train  accuracy is ",acc)

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
            PATH='/home/mayank-s/PycharmProjects/Datasets/pytorch_model_save/mytraining.pt'
            torch.save(model, PATH)

        # model.save_state_dict(PATH+run)'''