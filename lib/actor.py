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
    def __init__(self, env_params):
        super(Actor, self).__init__()
        self.f1 = nn.Linear(env_params['observation'] +env_params['goal'],256)
        self.f2 = nn.Linear(256,256)
        self.f3 = nn.Linear(256,256)
        self.f4 = nn.Linear(256,env_params['action'] )
        self.action_max = env_params['action_max']

    def forward(self, x):
        x = F.relu(self.f1(x))
        x = F.relu(self.f2(x))
        x = F.relu(self.f3(x))
        actions = self.action_max*torch.tanh(self.f4(x))
        return actions
