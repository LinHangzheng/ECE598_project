from stable_baselines3.common import buffers
from lib import *
from collections import  namedtuple
import os
import numpy as np
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F

from lib.normalizer import normalizer

class DDPG_HER(object):
    def __init__(self, args, env):
        self.args = args
        self.env = env
        self.normalizer_obs = normalizer(size=25, default_clip_range=200)
        self.normalizer_goal = normalizer(size=3, default_clip_range=200)

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
            a, b = self._update_model()
            if (episode_num%10==0):
                print("Episode:", episode_num)
                print("Actor_Loss: ", a, "Critic_Loss: ", b)

            #validation
            if (episode_num%50==0 and episode_num!=0):
                self.evaluate()
            #log info

        #save model
        # self.save()

        return


    def predict(self, state):
        action = self.actor(torch.tensor(np.concatenate((state['observation'],state['desired_goal'])),device=self.device).float())   
        return action


    def save(self, PATH):
        torch.save(self.actor, PATH)
        torch.save(self.critic, PATH)
        return
    

    def _update_target(self):
        self.actor_target.load_state_dict(self.actor.state_dict())
        for param in self.actor_target.parameters():
            param.requires_grad = False
        self.critic_target.load_state_dict(self.critic.state_dict())
        for param in self.critic_target.parameters():
            param.requires_grad = False

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
            # rollouts[idx][2] = rollouts[-1][2]
            # new_rollouts[idx][2] = 0
            # switch the target 
            new_rollouts[idx][0]['desired_goal'] = rollouts[-1][3]['achieved_goal']
            new_rollouts[idx][3]['desired_goal'] = rollouts[-1][3]['achieved_goal']
            new_rollouts[idx][2]=self.env.compute_reward(new_rollouts[idx][3]['achieved_goal'],new_rollouts[idx][3]['desired_goal'],{})

        return rollouts, new_rollouts

    def _passing_reward(self,rollouts):
        new_rollouts = copy.deepcopy(rollouts)
        for idx in range(len(new_rollouts)//2, len(new_rollouts)):
            # all the rewards will be the same as the final reward
            # rollouts[idx][2] = rollouts[-1][2]
            # new_rollouts[idx][2] = 0
            # switch the target 
            new_rollouts[idx][0]['desired_goal'] = rollouts[idx][3]['achieved_goal']
            new_rollouts[idx][3]['desired_goal'] = rollouts[idx][3]['achieved_goal']
            new_rollouts[idx][2]=self.env.compute_reward(new_rollouts[idx][3]['achieved_goal'],new_rollouts[idx][3]['desired_goal'],{})

        return new_rollouts
    
    def choose_act(self, action):
        action = action.detach().cpu().numpy().squeeze()
        # add the gaussian
        action += 0.2 * self.env.action_space.high[0] * np.random.randn(*action.shape)
        action = np.clip(action, -self.env.action_space.high[0], self.env.action_space.high[0])
        # random actions...
        random_actions = np.random.uniform(low=-self.env.action_space.high[0], high=self.env.action_space.high[0], \
                                            size=4)
        # choose if use the random actions
        action += np.random.binomial(1, 0.3, 1)[0] * (random_actions - action)
        return action

    def _collect_rollouts(self):
        
        rollouts = []
        state = self.env.reset()
        done = False

        obs_list = []
        goal_list = []

        # loop to collect the rough rollouts
        while not done:
            action = self.actor(torch.tensor(np.concatenate((state['observation'],state['achieved_goal'])),device=self.device).float())
            
            #choose action with noice
            action = self.choose_act(action)
            
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

            obs_list.append(np.clip(state['observation'],-200,200))
            goal_list.append(np.clip(state['achieved_goal'],-200,200))

        # recalculate the goal    
        rollouts, new_rollouts = self._recal_reward(rollouts)

        new_passing_rollouts = self._passing_reward(rollouts)

        
        # store the rollouts
        self.HER_buffer.insert(rollouts)
        self.HER_buffer.insert(new_rollouts)
        self.HER_buffer.insert(new_passing_rollouts)

        #update normalizers
        self.normalizer_obs.update(np.array(obs_list))
        self.normalizer_goal.update(np.array(goal_list))

        self.normalizer_obs.recompute_stats()
        self.normalizer_goal.recompute_stats()
    
    

    def _update_model(self):
        actor_loss, critic_loss = 0., 0.

        for epoch in range(self.args.epoch_num):
            actor_loss, critic_loss = 0., 0.                    # input_tensor = self._preproc_inputs(obs, g)s = 0., 0.

            obss, acts, rewards, next_obss, goals  = self.HER_buffer.sample_batch(self.args.batch_size)

            #normalize obs, goal and next_obs
            obss = torch.from_numpy(obss).float().to(self.device)
            obss = np.clip(obss, -200, 200)
            obss = self.normalizer_obs.normalize(obss)

            next_obss = torch.from_numpy(next_obss).float().to(self.device)
            next_obss = np.clip(next_obss, -200, 200)
            next_obss = self.normalizer_obs.normalize(next_obss)

            goals = torch.from_numpy(goals).float().to(self.device)
            goals = np.clip(goals, -200, 200)
            goals = self.normalizer_goal.normalize(goals)


            acts = torch.from_numpy(acts).float().to(self.device)
            rewards = torch.from_numpy(rewards).float().to(self.device)

            states = torch.cat((obss,goals),1)
            next_states = torch.cat((next_obss,goals),1)

            cur_act = self.actor(states)
            cur_val = self.critic(torch.cat((states, acts/self.env.action_space.high[0]),1))

            next_act = self.actor_target(next_states)
            next_val = self.critic_target(torch.cat((next_states, next_act/self.env.action_space.high[0]),1))

            V_target = rewards + self.args.gamma * next_val

            #clip the V
            clip_return = 1 / (1 - self.args.gamma)
            V_target = torch.clamp(V_target, -clip_return, 0)

            actor_loss = -torch.mean(self.critic(torch.cat((states, cur_act),1)))
            # actor_loss = -self.critic_network(inputs_norm_tensor, actions_real).mean()
            actor_loss += 1 * (cur_act / self.env.action_space.high[0]).pow(2).mean()
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
            
            if np.mod(epoch, self.args.target_update_per_epoch) == 0:
                self._update_target()

            #log info

        return actor_loss, critic_loss

    def evaluate(self):
        total_success_rate = []
        for _ in range(128):
            state = self.env.reset()
            
            done = False
            flag_success = False

            while (not done) and (not flag_success):
                with torch.no_grad():
                    state_o = np.clip(state['observation'],-200,200)
                    state_o = self.normalizer_obs.normalize(state_o)
                    state_g = np.clip(state['desired_goal'],-200,200)
                    state_g = self.normalizer_goal.normalize(state_g)

                    state = np.concatenate((state_o,state_g))
                    # state = np.clip(state, -200, 200)
                    state = torch.from_numpy(state).float().to(self.device)
                    pi = self.actor(state)
                    # convert the actions
                    actions = pi.detach().cpu().numpy().squeeze()
                _state, _, done, info = self.env.step(actions)
                state = _state
                flag_success = info['is_success']
            total_success_rate.append(flag_success)
        total_success_rate = np.array(total_success_rate)
        print("Success Rate: ", float(np.sum(total_success_rate))/float(len(total_success_rate)))

        return None
