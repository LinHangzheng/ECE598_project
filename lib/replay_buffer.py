from collections import  namedtuple, deque
from torch.utils.data import Dataset
import numpy as np
import random

class ReplayBuffer(object):
    def __init__(self, size, env_params):
        self.size = size
        self.buffer = {
            'obs': np.empty((size, env_params['max_episode_steps'], env_params['observation'])),
            'action': np.empty((size, env_params['max_episode_steps'], env_params['action'])),
            'next_obs': np.empty((size, env_params['max_episode_steps'], env_params['observation'])),
            'reward': np.empty((size, env_params['max_episode_steps'], 1)),
            'achieved_goal': np.empty((size, env_params['max_episode_steps'], env_params['goal'])),
            'desired_goal': np.empty((size, env_params['max_episode_steps'], env_params['goal'])),
            'next_achieved_goal': np.empty((size, env_params['max_episode_steps'], env_params['goal']))
        }
        
        self.idx = 0
        self.buffer_full = False


    # insert new data input the replay buffer
    # @ rollouts : list<trainsition> nx1  
    def insert(self, rollouts):
        self.buffer['obs'][self.idx] = rollouts['obs']
        self.buffer['action'][self.idx] = rollouts['action']
        self.buffer['reward'][self.idx] = rollouts['reward']
        self.buffer['next_obs'][self.idx] = rollouts['next_obs'] 
        self.buffer['achieved_goal'][self.idx] = rollouts['achieved_goal']
        self.buffer['desired_goal'][self.idx] = rollouts['desired_goal']
        self.buffer['next_achieved_goal'][self.idx] = rollouts['next_achieved_goal']
 
        self.update_idx()
    
    
    # sample a batch of data
    # @ batch_size : batch size of the output
    def sample_batch(self, batch_size):
        if self.buffer_full:
            samples_idx = np.random.choice(self.size,batch_size,replace=False)
        else:
            samples_idx = np.random.choice(self.idx,batch_size,replace=True)

        transition = {
            'obs':self.buffer['obs'][samples_idx],
            'action':self.buffer['action'][samples_idx],
            'reward':self.buffer['reward'][samples_idx],
            'next_obs':self.buffer['next_obs'][samples_idx],
            'achieved_goal':self.buffer['achieved_goal'][samples_idx],
            'desired_goal':self.buffer['desired_goal'][samples_idx],
            'next_achieved_goal':self.buffer['next_achieved_goal'][samples_idx]
        }
        return transition
    

    def update_idx(self):
        if self.idx == self.size-1:
            self.buffer_full = True
        self.idx = (self.idx+1)%self.size
    

    # get the size of replay buffer
    def __len__(self):
        if self.buffer_full:
            len = self.size
        else:
            len = self.idx
        
        return len 


