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
    def __init__(self, env_params):
        super(Critic, self).__init__()

        # self.bn = nn.BatchNorm1d(env_params['observation'] +env_params['goal'] + env_params['action'] )
        self.f1 = nn.Linear(env_params['observation'] +env_params['goal'] + env_params['action'] ,256)
        self.f2 = nn.Linear(256,256)
        self.f3 = nn.Linear(256,256)
        self.f4 = nn.Linear(256,1)


    def forward(self, x):
        # if len(list(x.shape)) != 1:
        #     x = self.bn(x)
        x = F.relu(self.f1(x))
        x = F.relu(self.f2(x))
        x = F.relu(self.f3(x))
        V = self.f4(x)
        return V