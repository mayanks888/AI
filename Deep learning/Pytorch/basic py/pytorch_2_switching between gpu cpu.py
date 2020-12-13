import torch

tor = torch.ones(2, 2)
print(tor.data)
if torch.cuda.is_available():
    print('cuda availabel')
    tor.cuda()  # remember this convert cpu to cuda
    print(tor.data)
tor.cpu  # this will convert back to cpu
print(tor.data)

device = torch.device(
    'cuda:0' if torch.cuda.is_available() else 'cpu')  # this is to switch anything to gpu if available
print('the state of device is ', device)
tor1 = torch.ones(2, 2)
tor1.to(device)  # if gpu is presenet then it will be switch otherwise to cpu
print(tor1)
