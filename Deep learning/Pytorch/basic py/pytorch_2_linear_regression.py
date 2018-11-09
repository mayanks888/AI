import torch
from torch.autograd import Variable
# input matrix is (3x1)
#
# input_data = Variable(torch.Tensor([[1.0], [2.0], [3.0]]))
# output_data = Variable(torch.Tensor([[2.0], [4.0], [6.0]]))

input_data=torch.tensor([[4.0],[5],[9]])
output_data=torch.tensor([[5.0],[7],[11]])
print(output_data.data)


# print(input_data.data,'\n',output_data.data)

class Mymodel(torch.nn.Module):#mymodel is a sub class of model nn.module
    def __init__(self):
        super(Mymodel, self).__init__()
        self.linear_function=torch.nn.Linear(1,1)#(here (1,1) is no if input feature and output features

    def forward(self,x):
        ypred=self.linear_function(x)
        print('y predictin',ypred.data)
        return ypred


model=Mymodel()

# Create loss function and optimise function
criterea=torch.nn.MSELoss(size_average=False)#this is like we pick a specific function from torch library and later we can give it a value
optimise=torch.optim.SGD(model.parameters(),lr=.01)#this a fucntion picked for optimisation

# now training and and forward pass
epochs=1
for loop in range(epochs):
    y_predicted=model(input_data)
    loss=criterea(y_predicted,output_data)
    print('lossdata',epochs, loss.data)
    print('lossitem',epochs, loss.item())#loss.data both means the same thing
    optimise.zero_grad()#this is to find fresh gradient every time in each epochs good practise
    loss.backward()#this commmand is responsible for backward propogation(it finds delta (loss vs each initialse weight)
    optimise.step()#with this command the gradient will added to the intial weights according to learning rate.(weight update)

#
# After training move to testing
hour_var =torch.Tensor([[4.0],[8]])
y_pred = model(hour_var)
print("predict (in training)", 4, model(hour_var).data[0])
# print("predict (after training)", 4, model(hour_var).data[0][0])  #here 4 is upto 4 decimal digit