import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.nn.functional as F
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix
from batchup import data_source

from torch.autograd import Variable
# input matrix is (3x1)
#
# input_data = Variable(torch.Tensor([[1.0], [2.0], [3.0]]))
# output_data = Variable(torch.Tensor([[2.0], [4.0], [6.0]]))
data=pd.read_csv('/home/mayank-s/PycharmProjects/Datasets/Social_Network_Ads.csv')

print(data.head())

x = data.iloc[:, [2,3]].values
y = data.iloc[:, 4].values

X_train, X_test, y_train, y_test = train_test_split(x, y, test_size = .20, random_state = 0)

#Here I have standardise the features (set the value between mean 0 and sd=1))
scale_val=StandardScaler()
scale_x_train=scale_val.fit_transform(X_train)
scale_x_test = scale_val.transform(X_test)
# print("the scaler mean  is",scale_x[0].mean(),scale_x[1].mean())

# input_data=torch.tensor([[4.0],[5],[9]])
# output_data=torch.tensor([[5.0],[7],[11]])
# print(output_data.data)


# print(input_data.data,'\n',output_data.data)

class Mymodel(torch.nn.Module):#mymodel is a sub class of model nn.module
    def __init__(self):
        super(Mymodel, self).__init__()
        self.linear_function1=torch.nn.Linear(2,3)#(here (1,1) is no if input feature and output features
        self.linear_function2 =torch.nn.Linear(3, 1)

    def forward(self,x):
        hidden_layer1=F.relu(self.linear_function1(x))
        ypred=self.F.relu(self.linear_function2())
        print('y predictin',ypred.data)
        return ypred


model=Mymodel()

# Create loss function and optimise function
criterea=torch.nn.MSELoss(size_average=False)#this is like we pick a specific function from torch library and later we can give it a value
optimise=torch.optim.SGD(model.parameters(),lr=.01)#this a fucntion picked for optimisation

# ("predict (after training)", 4, model(hour_var).data[0][0])  #here 4 is upto 4 decimal digit

epochs=1
for loop in range(epochs):
    ds = data_source.ArrayDataSource([scale_x_train, y_train])
    # Iterate over samples, drawing batches of 64 elements in random order
    for (data, target) in ds.batch_iterator(batch_size=5, shuffle=True):
        new_data = data.reshape(-1,2,1)
        cool_data = new_data[0]
        data= torch.tensor(new_data).float()
        target= torch.tensor(target)
        y_predicted = model(data)
        loss = criterea(y_predicted, target)
        print(epochs, loss.data)
        print(epochs, loss.item())  # loss.data both means the same thing
        optimise.zero_grad()
        loss.backward()
        optimise.step()
