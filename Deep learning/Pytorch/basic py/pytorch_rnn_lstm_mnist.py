
from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable


# Training settings
batch_size = 128

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
                                           shuffle=True,drop_last=True)

test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                          batch_size=batch_size,
                                          shuffle=False)




class RNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim,layer_dim):
        super(RNN, self).__init__()
        self.input_dim=input_dim
        self.hidden_dim=hidden_dim
        self.output_dim=output_dim
        self.layer_dim=layer_dim
        #creating a hidden layer
        # self.hidden_layer=nn.RNN(self.input_dim,self.hidden_dim,self.layer_dim,batch_first=True, nonlinearity='relu')
        self.hidden_layer=nn.LSTM(self.input_dim,self.hidden_dim,self.layer_dim,batch_first=True)
        #creating an output layer
        self.output_layer=nn.Linear(self.hidden_dim,self.output_dim)

    def forward(self, x,hidden_state,cell_stae):
        # remeber this is  how you have to defines input size
        # input ** of; shape; `(seq_len, batch, input_size)`: tensor; containing; the; features
        x=x.view(-1, 784)#here 64 is batch size do remember to
        # modiefes this also when changing batch
        x=x.unsqueeze(0)
        # x=x.reshape(64,1,-1)

        # ** h_n ** (num_layers * num_directions, batch, hidden_size)
        # h0 = torch.randn(1, 64, 100)

        # val,(current_hidden_state,current_cell_stae)=self.hidden_layer(x,(hidden_state,cell_stae))
        val,(current_hidden_state,current_cell_stae)=self.hidden_layer(x,None)
        val=self.output_layer(val)
        # val= F.log_softmax(val)
        return val,current_hidden_state,current_cell_stae

input_dim=784
output_dim=10
hidden_dim=100
no_of_hidden_layer=1
# epoch=4


model = RNN(input_dim,hidden_dim,output_dim,no_of_hidden_layer)

optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.5)
criterion = torch.nn.CrossEntropyLoss()

def train(epoch):
    h_state=torch.randn(1, 1, 100)
    cell_state=torch.randn(1, 1, 100)
    model.train()

    print(1)
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = Variable(data), Variable(target)
        optimizer.zero_grad()
        output,h_state,cell_state = model(data,h_state,cell_state)
        h_state = h_state.data
        cell_state = cell_state.data
        # output=output[:,-1,:]

        output=output.view(-1, 10)
        loss = criterion(output, target)
        # loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % 10 == 0:
            # test_output = rnn(test_x[:10].view(-1, 28, 28))
            # pred_y = torch.max(test_output, 1)[1].data.numpy().squeeze()
            # accuracy = float((pred_y == test_y).astype(int).sum()) / float(test_y.size)
            acc=test()
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}\tacc: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.data[0],acc))


def test():
    model.eval()
    for batch_idx, (data, target) in enumerate(train_loader):
    # Iterate over samples, drawing batches of 64 elements in random order

        # cv2.imshow("image",np.reshape(test_data[5],newshape=(28,28,1)))
        # cv2.waitKey(500)
        # cv2.destroyAllWindows()
        test_data1, target = torch.tensor(data), torch.tensor(target).long()
        output, h_state, cell_state = model(data, 1, 1)
        # optimizer.zero_grad()
        # output = model(test_data1)
        out = F.log_softmax(output.squeeze())
        out = torch.max(out, 1)[1]
        match = torch.eq(out, target.squeeze())
        all1 = torch.sum(match)
        acc = all1.item() / len(match)
        # print("test accuracy is ", acc)
        return acc
        break

for epoch in range(1, 10):
    train(epoch)
