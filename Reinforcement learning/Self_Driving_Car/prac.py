import torch
import pandas
# (torch.tensor([ 0.25, 0.25, 0.25, 0.25 ]))
'''probs=torch.tensor([0.90,0.05,0.03,.02])
dat=[]
for val in range(100):
    m = torch.distributions.Categorical(probs)
    action = m.sample()
    print(int(action))
    dat.append(int(action))

for val in range(4):
 print("occurane of {dt} is {vl}".format(dt=val,vl=dat.count(val)))'''


# m = torch.randn(4,2)
# ids = torch.Tensor([1,1,0,0]).long()
# id_ch=ids.view(-1,1)
# gather=(m.gather(1,id_ch ))
# print(gather)

global a_b
a_b=2

def init():
    a_b=5
    return a_b
# a_b=init()
print(a_b)

if init()==5:
    print(a_b)
    print('it worked')
