from collections import  namedtuple, deque
from torch.utils.data import Dataset
import random

class ReplayBuffer(object):
    def __init__(self, size):
        
        self.ReBuffer = deque([],maxlen=size)

    # insert new data input the replay buffer
    # @ rollouts : list<trainsition> nx1  
    def insert(self, rollouts):
        for rollout in rollouts:
            self.ReBuffer.append(rollout)
        
    # sample a batch of data
    # @ batch_size : batch size of the output
    def sample_batch(self, batch_size):
        if len(self.ReBuffer) < batch_size:
            samples = random.choices(self.ReBuffer,k=batch_size)
        else:
            samples = random.sample(self.ReBuffer,batch_size)
        return samples
    
    # get the size of replay buffer
    def __len__(self):
        return len(self.ReBuffer)


