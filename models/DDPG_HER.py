from stable_baselines3.common import buffers
from lib import *
from collections import  namedtuple
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class DDPG_HER(object):
    def __init__(self, args, env):
        # super(DDPG, self).__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.env = env
        obs = self.env.reset()
        action_space = env.action_space.shape[0]
        obs_size = obs['observation'].shape[0]
        goal_size = obs['desired_goal'].shape[0]

        # Define the actor
        self.actor = Actor(obs_size, goal_size, action_space).to(self.device)
        self.actor_target = Actor(obs_size, goal_size, action_space).to(self.device)

        # Define the critic
        self.critic = Critic(obs_size, goal_size, action_space).to(self.device)
        self.critic_target = Critic(obs_size, goal_size, action_space).to(self.device)

        self.args = args

        self.HER_buffer = ReplayBuffer(args.buffer_size) 
        # self.transition = namedtuple('Transition',('state','action','reward','next_state'))

        
        self.gamma = args.gamma
        self.T = args.T
        self.episode = args.episode
        self.epoch_num = args.epoch_num
        self.batch_size = args.batch_size
        self.lr = args.lr
    
    def transition(self, state, action, reward, next_state):
        return [state, action, reward, next_state]


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
            action = self.actor(torch.tensor(np.concatenate((state['observation'],state['achieved_goal'])),device=self.device).float())
            action = action.tolist()
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
            rollouts[idx][2] = rollouts[-1][2]
            new_rollouts[idx][2] = 0
            # switch the target 
            new_rollouts[idx][0]['desired_goal'] = rollouts[-1][3]['achieved_goal']

        return rollouts, new_rollouts

    def update_model(self):
        #modified to use two optimizers
        loss = torch.nn.MSELoss()
        optim_actor = torch.optim.Adam(self.actor.parameters(), lr = self.lr)
        optim_critic = torch.optim.Adam(self.critic.parameters(), lr = self.lr)

        for epoch in range(self.epoch_num):
            actor_loss, critic_loss = 0., 0.

            sample_batch = self.HER_buffer.sample_batch(self.batch_size)

            state_batch = []
            act_batch = []
            next_state_batch = []
            reward_batch = []

            for i in sample_batch:
                state_batch.append(np.concatenate((i[0]['observation'],i[0]['achieved_goal'])))
                act_batch.append(i[1])
                next_state_batch.append(np.concatenate((i[3]['observation'],i[3]['achieved_goal'])))
                reward_batch.append(i[2])

            state_batch = torch.FloatTensor(state_batch,device=self.device)
            act_batch = torch.FloatTensor(act_batch, device=self.device)
            next_state_batch = torch.FloatTensor(next_state_batch, device=self.device)
            reward_batch = torch.FloatTensor(reward_batch, device=self.device)
            # act_batch = act_batch.reshape(-1,1)

            cur_act = self.actor(state_batch)
            cur_val = self.critic(torch.cat((state_batch, cur_act),1))

            next_act = self.actor(next_state_batch)
            next_val = self.critic(torch.cat((next_state_batch, next_act),1))

            V_target = reward_batch.reshape(-1,1) + self.gamma * next_val

            delta = V_target - cur_val
            
            #needs modification
            log_act_distribution = self.actor.actor_to_distribution(cur_act).log_prob(act_batch)

            actor_loss = -torch.mean(delta.detach() * log_act_distribution)
            critic_loss = loss(V_target.detach(), cur_val)
            # total_loss = critic_loss + actor_loss

            optim_actor.zero_grad()
            actor_loss.backward()
            optim_actor.step()

            optim_critic.zero_grad()
            critic_loss.backward()
            optim_critic.step()
            # actor_loss.backward(retain_graph=True)
            # critic_loss.backward()
            
            #log info

        return

    def predict(self, obs, goal):
        action = self.actor(torch.tensor(np.concatenate((obs,goal)),device=self.device).float())   
        return action

    def save(self, PATH):
        torch.save(self.actor, PATH)
        torch.save(self.critic, PATH)
        return