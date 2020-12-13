from __future__ import print_function

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torchvision import datasets, transforms

# Training settings
batch_size = 64

# MNIST Dataset
train_dataset = datasets.MNIST(root='./data/',
                               train=True,
                               transform=transforms.ToTensor(),
                               download=True)

test_dataset = datasets.MNIST(root='./data/',
                              train=False,
                              transform=transforms.ToTensor())

# Data Loader (Input Pipeline)
train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                           batch_size=batch_size,
                                           shuffle=True)

test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                          batch_size=batch_size,
                                          shuffle=False)


class RNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, layer_dim):
        super(RNN, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.layer_dim = layer_dim
        # creating a hidden layer
        # self.hidden_layer=nn.RNN(self.input_dim,self.hidden_dim,self.layer_dim,batch_first=True, nonlinearity='relu')
        self.hidden_layer = nn.LSTM(self.input_dim, self.hidden_dim, self.layer_dim, batch_first=True)
        # creating an output layer
        self.output_layer = nn.Linear(self.hidden_dim, self.output_dim)

    def forward(self, x, hidden_state):
        # remeber this is  how you have to defines input size
        # input ** of; shape; `(seq_len, batch, input_size)`: tensor; containing; the; features
        x = x.view(64, -1)  # here 64 is batch size do remember to
        # modiefes this also when changing batch
        x = x.unsqueeze(0)
        x = x.reshape(64, 1, -1)

        # ** h_n ** (num_layers * num_directions, batch, hidden_size)
        h0 = torch.randn(1, 64, 100)

        val, current_hidden_state = self.hidden_layer(x, hidden_state)
        val = self.output_layer(val)
        val = F.log_softmax(val)
        return val, current_hidden_state


input_dim = 784
output_dim = 10
hidden_dim = 100
no_of_hidden_layer = 1
# epoch=4


model = RNN(input_dim, hidden_dim, output_dim, no_of_hidden_layer)

optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.5)


def train(epoch):
    h_state = None
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = Variable(data), Variable(target)
        optimizer.zero_grad()
        output, h_state = model(data, h_state)
        h_state = h_state.data
        loss = F.nll_loss(output.squeeze(), target)
        loss.backward()
        optimizer.step()
        if batch_idx % 10 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                       100. * batch_idx / len(train_loader), loss.data[0]))


for epoch in range(1, 10):
    # train(epoch)
    h_state = None
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = Variable(data), Variable(target)
        optimizer.zero_grad()
        output, h_state = model(data, h_state)
        h_state = h_state.data
        loss = F.nll_loss(output.squeeze(), target)
        loss.backward()
        optimizer.step()
        if batch_idx % 10 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                       100. * batch_idx / len(train_loader), loss.data[0]))
    # test()
