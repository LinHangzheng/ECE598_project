from stable_baselines3.common import buffers
from lib import *
from collections import  namedtuple
import os
import numpy as np
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F

class DDPG_HER(object):
    def __init__(self, args, env):
        self.args = args
        self.env = env

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        env_params = get_env_parameters(self.env)

        self._set_net(env_params)
        self._set_criterion()
        self._set_opt()

        self.HER_buffer = ReplayBuffer(args.buffer_size, env_params) 

    def learn(self):
        #log info  

        for episode_num in range(self.args.episode):
        
            #sample a goal and starting position from env

            #collect rollouts and store in HER buffer
            self._collect_rollouts()

            #update models using HER buffer
            self._update_model()

            #validation
            #log info

        #save model
        # self.save()

        return


    def predict(self, obs, goal):
        action = self.actor(torch.tensor(np.concatenate((obs,goal)),device=self.device).float())   
        return action


    def save(self, PATH):
        torch.save(self.actor, PATH)
        torch.save(self.critic, PATH)
        return
    

    def _set_net(self,env_params):
        # Define the actor
        self.actor = Actor(env_params).to(self.device)
        self.actor_target = Actor(env_params).to(self.device)

        # Define the critic
        self.critic = Critic(env_params).to(self.device)
        self.critic_target = Critic(env_params).to(self.device)

        # load the weights to the target nets
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.critic_target.load_state_dict(self.critic.state_dict())
    

    def _set_criterion(self):
        self.criterion = torch.nn.MSELoss()


    def _set_opt(self):
        self.optim_actor = torch.optim.Adam(self.actor.parameters(), lr = self.args.lr)
        self.optim_critic = torch.optim.Adam(self.critic.parameters(), lr = self.args.lr)

    # The function that recalculates the reward
    # input:
    #   @ rollouts : list<transition>[n,1]
    # output:
    #   @ rollouts: the rollouts that recalculates the rewards
    #   @ new_rollouts: the rollouts that switches the goal
    def _recal_reward(self,rollouts):
        new_rollouts = copy.deepcopy(rollouts)
        for idx in range(len(new_rollouts)):
            # all the rewards will be the same as the final reward
            rollouts[idx][2] = rollouts[-1][2]
            new_rollouts[idx][2] = 0
            # switch the target 
            new_rollouts[idx][0]['desired_goal'] = rollouts[-1][3]['achieved_goal']
            new_rollouts[idx][3]['desired_goal'] = rollouts[-1][3]['achieved_goal']

        return rollouts, new_rollouts


    def _collect_rollouts(self):
        rollouts = []
        state = self.env.reset()
        done = False

        # loop to collect the rough rollouts
        while not done:
            action = self.actor(torch.tensor(np.concatenate((state['observation'],state['achieved_goal'])),device=self.device).float())
            action = action.tolist()
            # state = {observation: array<float>[25,1],
            #            achieved_goal: array<float>[3,1],
            #            desired_goal: array<float>[3,1]}
            # reward = float, -1 or 0
            # done = bool               
            next_state, reward, done, _ = self.env.step(action)
            # add the new rollout
            rollouts.append([state,action,reward,next_state])
            state = next_state

        # recalculate the goal    
        rollouts, new_rollouts = self._recal_reward(rollouts)
        
        # store the rollouts
        self.HER_buffer.insert(rollouts)
        self.HER_buffer.insert(new_rollouts)
    

    def _update_model(self):
        for epoch in range(self.args.epoch_num):
            actor_loss, critic_loss = 0., 0.

            states, acts, rewards, next_states  = self.HER_buffer.sample_batch(self.args.batch_size)

            states = torch.from_numpy(states).float().to(self.device)
            acts = torch.from_numpy(acts).float().to(self.device)
            rewards = torch.from_numpy(rewards).float().to(self.device)
            next_states = torch.from_numpy(next_states).float().to(self.device)

            cur_val = self.critic(torch.cat((states, acts),1))

            next_act = self.actor(next_states)
            next_val = self.critic(torch.cat((next_states, next_act),1))

            V_target = rewards + self.args.gamma * next_val

            actor_loss = -torch.mean(V_target)
            critic_loss = self.criterion(V_target, cur_val)
            # total_loss = critic_loss + actor_loss

            
            self.optim_actor.zero_grad()
            actor_loss.backward(retain_graph=True)
            self.optim_critic.zero_grad()
            critic_loss.backward()

            self.optim_actor.step()
            self.optim_critic.step()
            # actor_loss.backward(retain_graph=True)
            # critic_loss.backward()
            
            #log info

        return