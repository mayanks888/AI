import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms

train_datasets = datasets.MNIST(root="./data/", train=True, transform=transforms.ToTensor())
test_datasets = datasets.MNIST(root="./data/", train=False, transform=transforms.ToTensor())

train_loader = torch.utils.data.DataLoader(dataset=train_datasets, batch_size=16, shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset=test_datasets, batch_size=16, shuffle=True)


class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=10, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(in_channels=10, out_channels=10, kernel_size=3, padding=1)
        self.mxpool = nn.MaxPool2d(2)
        self.lin = nn.Linear(490, 10)

    def forward(self, x):
        in_size = x.size(0)
        first_Conv = self.conv1(x)
        x = F.relu(self.mxpool(first_Conv))
        sec_conv = self.conv2(x)
        x = F.relu(self.mxpool(sec_conv))
        x = x.view(in_size, -1)
        lin = self.lin(x)
        return F.log_softmax(lin)


Model = MyModel()
optimizer = optim.SGD(Model.parameters(), lr=0.01, momentum=0.5)
# optimizer = torch.optim.Adam(Model.parameters(), lr=0.1)
criterea = torch.nn.CrossEntropyLoss()  # this include softmax and cross entropy


def train(epoch):
    Model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        # data, target = Variable(data), Variable(target)
        data, target = torch.tensor(data), torch.tensor(target).long()
        optimizer.zero_grad()
        output = Model(data)
        loss = criterea(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % 10 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                       100. * batch_idx / len(train_loader), loss.data))


if __name__ == '__main__':
    for i in range(100):
        train(i)
