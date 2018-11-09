import torch
from torchvision import datasets, transforms
# train_dataset = datasets.MNIST(root='./data/', train=True, download=True)
# train_dataset = datasets.MNIST(root='./data/', train=True, transform=transforms.ToTensor(), download=True)
# #
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

test_dataset = datasets.MNIST(root='./data/', train=False, transform=transforms.ToTensor())

IMG_SIZE = 15
_mean = [0.485, 0.456, 0.406]
_std = [0.229, 0.224, 0.225]




trans = transforms.Compose([
    transforms.RandomCrop(IMG_SIZE),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(.3, .3, .3),
    transforms.ToTensor(),
    transforms.Normalize(_mean, _std),
])

train_dataset = datasets.MNIST(root='./data/', train=True, transform=trans, download=True)
#now thing to understand is that transformaionof image will not happended now rather it will be for during downloader funtionis called

# cool=trans(train_dataset[13][0])#13 image and read it img values stored at 0
print(1)

train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=3, shuffle=True)
for batch_idx, (data, target) in enumerate(train_loader):
    print(data)