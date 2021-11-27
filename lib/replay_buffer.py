from collections import  namedtuple, deque
from torch.utils.data import Dataset
import numpy as np
import random

class ReplayBuffer(object):
    def __init__(self, size, env_params):
        self.size = size
        self.state = np.empty((size,env_params['observation']+env_params['goal']))
        self.act = np.empty((size,env_params['action']))
        self.next_state = np.empty((size,env_params['observation']+env_params['goal']))
        self.reward = np.empty((size,1))
        self.idx = 0
        self.buffer_full = False


    # insert new data input the replay buffer
    # @ rollouts : list<trainsition> nx1  
    def insert(self, rollouts):
        for rollout in rollouts:
            obs = rollout[0]['observation']
            next_obs = rollout[3]['observation']
            goal = rollout[0]['desired_goal']

            state = np.concatenate((obs,goal))
            next_state = np.concatenate((next_obs,goal))

            self.state[self.idx] = state
            self.act[self.idx] = rollout[1]
            self.reward[self.idx] = rollout[2]
            self.next_state[self.idx] = next_state 
            self.update_idx()
    
    
    # sample a batch of data
    # @ batch_size : batch size of the output
    def sample_batch(self, batch_size):
        if self.buffer_full:
            samples_idx = np.random.choice(self.size,batch_size,replace=False)
        else:
            samples_idx = np.random.choice(self.idx,batch_size,replace=True)
        states = self.state[samples_idx]
        acts = self.act[samples_idx]
        rewards = self.reward[samples_idx]
        next_states = self.next_state[samples_idx]
        return states, acts, rewards, next_states
    

    def update_idx(self):
        if self.idx == self.size-1:
            self.buffer_full == True
        self.idx = (self.idx+1)%self.size
    

    # get the size of replay buffer
    def __len__(self):
        if self.buffer_full:
            len = self.size
        else:
            len = self.idx
        
        return len 


