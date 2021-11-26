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
    def __init__(self, obs_size, goal_size, action_space):
        super(Critic, self).__init__()

        self.f1 = nn.Linear(obs_size+goal_size+action_space,256)
        self.f2 = nn.Linear(256,256)
        self.f3 = nn.Linear(256,256)
        self.f4 = nn.Linear(256,1)


    def forward(self, state, actions):
        x = torch.cat((state, actions), dim=1)
        V = self.evaluate(x)
        return V