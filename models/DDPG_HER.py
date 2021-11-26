from stable_baselines3.common import buffers
from lib import *
from collections import  namedtuple
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class DDPG(object):
    def __init__(self, args, input_size, action_space, env):
        # super(DDPG, self).__init__()
        # Define the actor
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.actor = Actor(args.actor_hidden, input_size, action_space).to(self.device)
        self.actor_target = Actor(args.actor_hidden, input_size, action_space).to(self.device)

        # Define the critic
        self.critic = Critic(args.critic_hidden, input_size, action_space).to(self.device)
        self.critic_target = Critic(args.critic_hidden, input_size, action_space).to(self.device)

        self.args = args
        self.input_size = input_size
        self.action_space = action_space

        self.HER_buffer = ReplayBuffer(args.buffer_size) 
        self.transition = namedtuple('Transition',('state','action','reward','next_state','target'))

        self.env = env
        self.gamma = args.gamma
        self.T = args.T
        self.episode = args.episode
        self.epoch_num = args.epoch_num
        self.batch_size = args.batch_size


    def learn(self):
        #log info  

        for episode_num in range(self.episode):
        
            #sample a goal and starting position from env

            #collect rollouts and store in HER buffer
            self.collect_rollouts()

            #update models using HER buffer
            self.update_model()

            #validation
            #log info

        #save model
        # self.save()

        return

    def collect_rollouts(self):
        rollouts = []
        state = self.env.reset()
        done = False

        # loop to collect the rough rollouts
        while not done:
            action = self.actor(torch.tensor(np.cat((state.observation,state.achieved_goal)),device=self.device).float())
            action = [action.item()]
            # state = {observation: array<float>[25,1],
            #            achieved_goal: array<float>[3,1],
            #            desired_goal: array<float>[3,1]}
            # reward = float, -1 or 0
            # done = bool               
            next_state, reward, done, _ = self.env.step(action)
            # add the new rollout
            rollouts.append(self.transition(state,action,reward,next_state))
            state = next_state

        # recalculate the goal    
        rollouts, new_rollouts = self.recal_reward(rollouts)
        
        # store the rollouts
        self.HER_buffer.insert(rollouts)
        self.HER_buffer.insert(new_rollouts)
            
    # The function that recalculates the reward
    # input:
    #   @ rollouts : list<transition>[n,1]
    # output:
    #   @ rollouts: the rollouts that recalculates the rewards
    #   @ new_rollouts: the rollouts that switches the goal
    def recal_reward(self,rollouts):
        new_rollouts = rollouts.copy()
        for idx in range(len(new_rollouts)):
            # all the rewards will be the same as the final reward
            rollouts[idx].reward = rollouts[-1].reward
            new_rollouts[idx].reward = 0
            # switch the target 
            new_rollouts[idx].state.desired_goal = rollouts[-1].next_state.achieved_goal

        return rollouts, new_rollouts

    def update_model(self):
        for epoch in range(self.epoch_num):

            #sample a minibatch from HER buffer

            #update model with the minibatch


        pass

    def predict(self):
        pass

    def save(self):
        pass