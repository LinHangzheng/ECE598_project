from stable_baselines3.common import buffers
from lib import *
from collections import  namedtuple
import os
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
            self.collect_rollouts(batch=10)

            #update models using HER buffer
            self.update_model()

            #validation
            #log info

        #save model
        # self.save()

        return

    def collect_rollouts(self, batch):
        rollouts = []
        
        for _ in range(batch):
            obs = self.env.reset()
        
            actions = self.actor(torch.tensor(obs,device=self.device).float())
            while (1):
                action = [actions[env_idx].item()]
                new_state, reward, done, _ = envs[env_idx].step(action)
                # add the new rollout
                rollouts.append(transition(states[env_idx],action,reward,new_state))
                states[env_idx] = new_state
                if done:
                    break

    def update_model(self):
        for epoch in range(self.epoch_num):

            #sample a minibatch from HER buffer

            #update model with the minibatch


        pass

    def predict(self):
        pass

    def save(self):
        pass