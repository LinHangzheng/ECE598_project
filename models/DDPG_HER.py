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
            self.collect_rollouts()

            #update models using HER buffer
            self.update_model()

            #validation
            #log info

        #save model
        # self.save()

        return

    def collect_rollouts(self):
        moved = False
        while not moved:
            rollouts = []
            obs = self.env.reset()
            action = self.actor(torch.tensor(obs,device=self.device).float())
            done = False

            # loop to collect the rough rollouts
            while not done:
                action = [action.item()]
                # new_obs = {observation: array<float>[25,1],
                #            achieved_goal: array<float>[3,1],
                #            desired_goal: array<float>[3,1]}
                # reward = float, -1 or 0
                # done = bool               
                new_obs, reward, done, _ = self.env.step(action)
                # add the new rollout
                rollouts.append(self.transition(torch.cat((obs.observation,obs.achieved_goal)),
                                                action,reward,
                                                torch.cat((new_obs.observation,new_obs.achieved_goal)))
                                                ,new_obs.desired_goal)
                obs = new_obs

            # recalculate the goal    
            moved, rollouts = self.recal_reward(rollouts)
            if not moved:
                continue
            
            # switch the target 
            new_rollouts = rollouts.copy()
            for idx in range(len(new_rollouts)):
                new_rollouts[idx].desired_goal = rollouts[-1].achieved_goal
            
            # store the rollouts
            self.HER_buffer.insert(rollouts)
            self.HER_buffer.insert(new_rollouts)
            
    # The function that recalculates the reward
    # input:
    #   @ rollouts : list<transition>[n,1]
    # output:
    #   @ moved: bool, if the slide block moved, it would be 1, otherwise it would be 0
    #   @ rollouts: the rollouts that recalculates the rewards
    def recal_reward(self,rollouts):
        # check if the block moved.
        if rollouts[-1].achieved_goal == rollouts[0].achieved_goal:
            return False, rollouts
        else:    
            # all the reward will be the same as the final reward
            for idx in range(len(rollouts)):
                rollouts.reward[idx] = rollouts.reward[-1]
            return True, rollouts

    def update_model(self):
        for epoch in range(self.epoch_num):

            #sample a minibatch from HER buffer

            #update model with the minibatch


        pass

    def predict(self):
        pass

    def save(self):
        pass