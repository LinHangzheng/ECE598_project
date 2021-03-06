from stable_baselines3.common import buffers, noise
from lib import *
from collections import  namedtuple
import os
import numpy as np
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
import logging as log
from tqdm import tqdm

class DDPG_HER(object):
    def __init__(self, args, env):
        self.args = args
        self.env = env

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.total_actor_loss, self.total_critic_loss =[],[]
        self.env_params = get_env_parameters(self.env)

        self._set_net(self.env_params)
        self._set_criterion()
        self._set_opt()
        self._set_random_act()

        self.HER_buffer = ReplayBuffer(args.buffer_size, self.env_params) 

    def learn(self):
        #log info  

        for episode_num in range(self.args.episode):
        
            #sample a goal and starting position from env

            #collect rollouts and store in HER buffer

            self._collect_rollouts(4,episode_num)

            #update models using HER buffer
            self._update_model()

            self._update_target()

            # self._random_act_decay()

            # if episode_num !=0 and np.mod(episode_num, self.args.log_per_episode) == 0:
            self._log(episode_num)

            #evaluate
            # if episode_num !=0 and np.mod(episode_num, self.args.evaluate_per_episode) == 0:
            success_rate = self._evaluate(episode_num)

            #log info

        #save model
        # self.save()

        return


    def predict(self, obs, goal):
        with torch.no_grad():
            action = self.actor(torch.tensor(np.concatenate((obs,goal)),device=self.device).float()).cpu().numpy().squeeze()
        return action


    def save(self, PATH):
        torch.save(self.actor, PATH)
        torch.save(self.critic, PATH)
        return
    
    def _evaluate(self, episode_num):
        total_success = 0.0
        max_episode_step = self.env_params['max_episode_steps']
        for _ in range(self.args.eval_episode_num):
            state = self.env.reset()
            obs = state['observation']
            goal = state['desired_goal']
            for _ in range(max_episode_step):
                act = self.predict(obs,goal)
                state_next, _, _, info = self.env.step(act)
                obs = state_next['observation']
                goal = state_next['desired_goal']
                is_success = info['is_success']
            total_success += is_success
        success_rate = total_success/self.args.eval_episode_num
        log.info(f'Evaluation at episode #{episode_num}, eval success rate = {success_rate}')

        #log info
        return success_rate


    def _update_target(self):
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.critic_target.load_state_dict(self.critic.state_dict())

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

    def _set_random_act(self):
        self.noise_eps = self.args.noise_eps
        self.random_eps = self.args.random_eps
        self.random_decay = self.args.random_decay

    def _random_act_decay(self):
        self.random_eps *= self.random_decay
        self.noise_eps *= self.random_decay

    # The function that recalculates the reward
    # input:
    #   @ rollouts : list<transition>[n,1]
    # output:
    #   @ rollouts: the rollouts that recalculates the rewards
    #   @ new_rollouts: the rollouts that switches the goal
    def _recal_reward(self,transition):
        HER_transition = copy.deepcopy(transition)
        # for idx in range(len(transition['action'])):
            # all the rewards will be the same as the final reward
            # rollouts[idx][2] = rollouts[-1][2]
            # new_rollouts[idx][2] = 0
            # switch the target 
        HER_transition['desired_goal'][:,:,:] = np.expand_dims(HER_transition['achieved_goal'][:,-1,:],axis=1).repeat(self.env_params['max_episode_steps'],axis=1)
        HER_transition['reward'][:,:,0] = self.env.compute_reward(HER_transition['achieved_goal'],HER_transition['desired_goal'],None)
        
        return HER_transition


    def _collect_rollouts(self, rollouts_num,episode_num):
        for _ in range(rollouts_num):
            moved = False
            # loop to collect the rough rollouts
            # while not moved:
            state = self.env.reset()
            rollouts = {'obs':[], 'action':[], 'reward':[], 'next_obs':[], 'achieved_goal':[], 'desired_goal':[]}
            done = False
            while not done:
                action = self.actor(torch.tensor(np.concatenate((state['observation'],state['achieved_goal'])),device=self.device).float())
                action = self._actions_noise(action,episode_num)
                action = action.tolist()
                # state = {observation: array<float>[25,1],
                #            achieved_goal: array<float>[3,1],
                #            desired_goal: array<float>[3,1]}
                # reward = float, -1 or 0
                # done = bool               
                next_state, reward, done, _ = self.env.step(action)
                # add the new rollout
                rollouts['obs'].append(state['observation'])
                rollouts['action'].append(action)
                rollouts['reward'].append([reward])
                rollouts['next_obs'].append(next_state['observation'])
                rollouts['achieved_goal'].append(next_state['achieved_goal'])
                rollouts['desired_goal'].append(next_state['desired_goal'])
                state = next_state
                # moved = self.env.compute_reward(rollouts[0][0]['achieved_goal'],rollouts[-1][3]['achieved_goal'],{})


            # recalculate the goal    
            # rollouts, new_rollouts = self._recal_reward(rollouts)
            
            # store the rollouts
            self.HER_buffer.insert(rollouts)
            # self.HER_buffer.insert(new_rollouts)
    

    def _update_model(self):
        self.actor.train()
        self.critic.train()
        for epoch in tqdm(range(self.args.epoch_num)):
            actor_loss, critic_loss = 0., 0.

            transition = self.HER_buffer.sample_batch(self.args.batch_size)
            HER_transition = self._recal_reward(transition)

            states = np.concatenate((HER_transition['obs'],HER_transition['desired_goal']),axis=2)
            next_states = np.concatenate((HER_transition['next_obs'],HER_transition['desired_goal']),axis=2)
            states = torch.from_numpy(states).float().to(self.device)
            acts = torch.from_numpy(HER_transition['action']).float().to(self.device)
            rewards = torch.from_numpy(HER_transition['reward']).float().to(self.device)
            next_states = torch.from_numpy(next_states).float().to(self.device)

            for path_idx in range(self.args.batch_size):
                cur_act = self.actor(states[path_idx])
                cur_val = self.critic(torch.cat((states[path_idx], acts[path_idx]),1))

                with torch.no_grad():
                    next_act = self.actor_target(next_states[path_idx]).detach()
                    next_val = self.critic_target(torch.cat((next_states[path_idx], next_act),1)).detach()

                    V_target = rewards[path_idx] + self.args.gamma * next_val
                    V_target = torch.clamp(V_target, -50, 0)

                actor_loss += -torch.mean(self.critic(torch.cat((states[path_idx], cur_act),1))) 
                actor_loss += torch.mean(cur_act.pow(2))
                critic_loss += self.criterion(V_target.detach(), cur_val)
            self.total_actor_loss.append(actor_loss.item())
            self.total_critic_loss.append(critic_loss.item())

                
            self.optim_actor.zero_grad()
            actor_loss.backward()
            self.optim_actor.step()

            self.optim_critic.zero_grad()
            critic_loss.backward()
            self.optim_critic.step()
        


        

    def _log(self,episode_num):
        #log info
        mean_actor_loss = np.mean(self.total_actor_loss)
        mean_critic_loss = np.mean(self.total_critic_loss)
        
        log.info(f'epoch: {episode_num}/{self.args.episode} actor loss: {mean_actor_loss} | critic loss: {mean_critic_loss}')
        log.info(f'buffer size: {len(self.HER_buffer)} noise epsilon: {self.noise_eps}')
        self.total_actor_loss = []
        self.total_cirtic_loss = []
    

    def _actions_noise(self, action, epoch):
        # add the gaussian
        action += self.noise_eps * self.env_params['action_max'] * torch.randn(*action.shape,device = self.device)
        action = torch.clamp(action, -self.env_params['action_max'], self.env_params['action_max'])
        # random actions...
        random_actions = np.random.uniform(low=-self.env_params['action_max'], high=self.env_params['action_max'], \
                                            size=self.env_params['action'])
        # choose if use the random actions
        action = random_actions if torch.rand(1) < self.random_eps else action
        return action