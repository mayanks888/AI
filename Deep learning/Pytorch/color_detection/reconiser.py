from __future__ import print_function
import torchvision.transforms as transforms
import os
import cv2
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(32*32*3, 50)
        self.fc1_drop = nn.Dropout(0.2)
        self.fc2 = nn.Linear(50, 50)
        self.fc2_drop = nn.Dropout(0.2)
        self.fc3 = nn.Linear(50, 10)

    def forward(self, x):
        x = x.view(-1, 32*32*3)
        x = F.relu(self.fc1(x))
        x = self.fc1_drop(x)
        x = F.relu(self.fc2(x))
        x = self.fc2_drop(x)
        return F.log_softmax(self.fc3(x), dim=1)


def main():

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    model = Net().to(device)
    model.load_state_dict(torch.load("snowball_traffic_color.pt"))
    model.eval()
    input_folder='/home/mayank_s/datasets/farminton/fileter color'
    for root, _, filenames in os.walk(input_folder):
        if (len(filenames) == 0):
            print("Input folder is empty")
            return 1
        # time_start = time.time()
        for filename in filenames:
            file_path = (os.path.join(root, filename));
            # ################################33
            img0 = Image.open(file_path)
            img0 = img0.convert("RGB")
            transform = transforms.Compose([transforms.Resize((32, 32)), transforms.ToTensor()])
            img = transform(img0)
            img=img.unsqueeze(0)
            img = img.to(device)
            output = model(img)
            data = torch.argmax(output, dim=1)
            print(output)
            traffic_light = ['black', 'green', 'red']
            light_color = traffic_light[int(data)]
            ############################################################################################
            image=cv2.imread(file_path)
            # transform = transforms.Compose([transforms.Resize((32, 32)), transforms.ToTensor()])
            # img=Image.fromarray(image)
            # image = transform(img)
            cv2.putText(image, light_color, (10, 10), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 0), 1)
            cv2.imshow('img', image)
            cv2.waitKey(1000)
            cv2.destroyAllWindows()

if __name__ == '__main__':
    main()