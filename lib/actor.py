import os
import torch
import torch.nn as nn
import torch.nn.functional as F

# The actor model of DDPG
# input:
#  @ input_size   : int, the size of input
#  @ action_space : int, the size of the action space
#  @ hidden_size  : list<int> 3x1, the size of each hidden layer 
# 
class Actor(nn.Module):
    def __init__(self, actor_hidden, input_size, action_space):
        super(Actor, self).__init__()
        self.feature = nn.ModuleList([nn.Linear(input_size,actor_hidden[0])])
        for i in range(len(actor_hidden)-1):
            self.feature.append(nn.Linear(actor_hidden[i],actor_hidden[i+1]))
            self.feature.append(nn.ReLU())
        self.feature.append(nn.Linear(actor_hidden[-1],action_space))

    def forward(self, inputs):
        action = self.feature(inputs)
        return action
