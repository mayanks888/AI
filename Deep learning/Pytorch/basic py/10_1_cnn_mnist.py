# from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torchvision import datasets, transforms

# Training settings
batch_size = 1000
# MNIST Dataset
# train_dataset = datasets.MNIST(root='./data/', train=True, download=True)
train_dataset = datasets.MNIST(root='./data/', train=True, transform=transforms.ToTensor(), download=True)
#
# # for img, label_id in train_dataset:
# for train_data,train_labels in train_dataset:
#     # print(label_id, train_dataset.classes[label_id])
#     # display(img)
#     # cv2.imshow('img', np.array(train_data))
#     # Image.save(trai)
#     # train_data.save('cool.jpg')
#     train_data.show()
#     # plt.imshow(np.asarray(train_data))
#     # break
#     # imshow(np.asarray(pil_im))

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


class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5, padding=1)
        self.conv2 = nn.Conv2d(10, 10, kernel_size=5)
        self.mp = nn.MaxPool2d(2)
        self.fc = nn.Linear(160, 10)

    def forward(self, x):
        in_size = x.size(0)
        first_conv = self.conv1(x)
        x = F.relu(self.mp(first_conv))
        # x = F.relu(self.mp(self.conv1(x)))
        # print (x.data)
        x = F.relu(self.mp(self.conv2(x)))
        # x = F.relu(self.mp(self.conv2(x)))
        # x = F.relu(self.mp(self.conv2(x)))
        # x = F.relu(self.mp(self.conv2(x)))
        # x = F.relu(self.mp(self.conv2(x)))
        # x = F.relu(self.mp(self.conv2(x)))
        x = x.view(in_size, -1)  # flatten the tensor
        x = self.fc(x)
        return F.log_softmax(x)


model = Net()

optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.5)


def train(epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = Variable(data), Variable(target)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % 10 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                       100. * batch_idx / len(train_loader), loss.data))


def test():
    model.eval()
    test_loss = 0
    correct = 0
    for data, target in test_loader:
        data, target = Variable(data, volatile=True), Variable(target)
        output = model(data)
        # sum up batch loss
        test_loss += F.nll_loss(output, target, size_average=False).data[0]
        # get the index of the max log-probability
        pred = output.data.max(1, keepdim=True)[1]
        correct += pred.eq(target.data.view_as(pred)).cpu().sum()

    test_loss /= len(test_loader.dataset)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))


for epoch in range(1, 10):
    train(epoch)
    break
    # test()

dummy_input = torch.randn(64, 1, 28, 28)

# torch.onnx.export(model, dummy_input, "onnx_model_name.onnx")
