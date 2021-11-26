import os
import torch
import torch.nn as nn
import torch.nn.functional as F

# The critic model of DDPG
# input:
#  @ input_size   : int, the size of input
#  @ action_space : int, the size of the action space
#  @ hidden_size  : list[int] 3x1, the size of each hidden layer 
# 

class Critic(nn.Module):
    def __init__(self, critic_hidden, input_size, action_space):
        super(Critic, self).__init__()
        
        self.pre_states = nn.ModuleList([nn.Linear(input_size,critic_hidden[0])])
        self.pre_states.append(nn.ReLU())

        self.evaluate = nn.ModuleList([nn.Linear(critic_hidden[0]+action_space,critic_hidden[1])])
        for i in range(1,len(critic_hidden)-1):
            self.evaluate.append(nn.Linear(critic_hidden[i],critic_hidden[i+1]))    
            self.evaluate.append(nn.ReLU())
        self.evaluate.append(nn.Linear(critic_hidden[-1],1))


    def forward(self, inputs, actions):
        x = self.pre_states(inputs)
        x = torch.cat((x, actions), 1)
        V = self.evaluate(x)
        return V