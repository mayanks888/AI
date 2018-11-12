import numpy  as np
import torch
import torch.nn as nn
import random
import os
import torch.optim as op
import torch.nn.functional as F

#lets make neural network

class Network(nn.Module):
    def __init__(self,input_size, np_action):
        #inout_size=no of sensor input
        # np_action=action taken by car
        super(Network,self).__init__()#this is for inheritance o
        self.input_size=input_size
        self.np_action=np_action
        self.fc1=nn.Linear(self.input_size,30)#30 is just a random values select ,Experiment with different values
        self.fc2=nn.Linear(in_features=30, out_features=self.np_action)

    def forward(self, state):
        fc1=F.relu(self.fc1(state))
        qvalues=self.fc2(fc1)
        return qvalues


#implement experiment replay
class Replay_memory(object):
    def __init__(self,capacity):
        self.capacity=capacity
        self.memory=[]#memory to be initialise for 100 transaction values

    def push(self, event):
        # event contain =[current_state, last_state, last_action, last_rewards]
        self.memory.append(event)#store event in memory
        if len(self.memory>self.capacity):
            del self.memory[0]#delete initial values if len is more then capacity of initialise storage
    def sample(self,batch_size):
        # self.sample=zip(*random.sample(self.memory, batch_size))
        samples = zip(*random.sample(self.memory, batch_size))#check how its work
        return map(lambda x: torch.tensor(torch.cat(x, 0)), samples)#convert sample in concat torch tensor

class Dqn():
    def __init__(self,input_size,np_action,gamma):
        self.gamma=gamma
        self.reward_window=[]#mean of previous rewards
        self.model=Network(input_size,np_action)
        self.memory=Replay_memory(100000)#total transition to go inside memory
        self.optimezer=torch.optim.Adam(self.model.parameters(),lr=0.001)#send all the parameter for model into optimiser
        self.last_state=torch.tensor(input_size).unsqueeze(0)
        self.last_action=0
        self.last_reward=0

    def select_action(self, state):
        probs = F.softmax(self.model(state) * 100)  # T=100 this just to increase the values before softmax
        action = probs.multinomial()#kind of argmax,lets run it and check
        return action.data[0, 0]#return value of action

    def learn(self, batch_state, batch_next_state, batch_reward, batch_action):
        outputs = self.model(batch_state).gather(1, batch_action.unsqueeze(1)).squeeze(1)
        next_outputs = self.model(batch_next_state).detach().max(1)[0]
        target = self.gamma * next_outputs + batch_reward
        td_loss = F.smooth_l1_loss(outputs, target)
        self.optimizer.zero_grad()
        td_loss.backward(retain_variables=True)
        self.optimizer.step()

