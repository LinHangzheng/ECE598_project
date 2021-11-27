import torch
import numpy as np
import gym
import os
import logging as log
import random
from stable_baselines3 import PPO, HerReplayBuffer, DDPG
from torch._C import device
from models.DDPG_HER import DDPG_HER

class Trainer(object):
    def __init__(self, args, args_str):
        self.args = args 
        self.args_str = args_str
        
        self.init_device()
        self.init_seed()
        self.build_path()
        self.set_env()
        self.set_network()

    def init_device(self):
        self.use_cuda = torch.cuda.is_available()   
        self.device = torch.device('cuda' if self.use_cuda else 'cpu')
        if self.use_cuda:
            device_name = torch.cuda.get_device_name(device=self.device)
            log.info(f'Using {device_name} with CUDA v{torch.version.cuda}')

    def init_seed(self):
         ## fix random seed
        random.seed(self.args.seed)
        np.random.seed(self.args.seed)
        torch.manual_seed(self.args.seed)

    def build_path(self):
        # create the output folder
        if not os.path.exists(self.args.output_path):
            os.makedirs(self.args.output_path)    
        if not os.path.exists(os.path.join(self.args.output_path,self.args.model)):
            os.makedirs(os.path.join(self.args.output_path,self.args.model))    

    def set_env(self):
        self.env = gym.make(self.args.environment_name)
        self.obs = self.env.reset()
    
    
    def set_network(self):
        if self.args.model == 'PPO':
            self.model = PPO("MultiInputPolicy", self.env, verbose=1,device=self.device )

        if self.args.model == 'DEFAULT_HER':
            self.model = DDPG(
                "MultiInputPolicy",
                self.env,
                replay_buffer_class=HerReplayBuffer,
                # Parameters for HER
                replay_buffer_kwargs=dict(
                    n_sampled_goal=4,
                    goal_selection_strategy='future',
                    online_sampling=True,
                    max_episode_length=100000,
                ),
                verbose=1,
                device = self.device
            )

        if self.args.model == 'DDPG_HER':
            self.model = DDPG_HER(self.args, self.env)
    
    def render(self):
        for i in range(1000):
            action, _states = self.model.predict(obs, deterministic=True)
            obs, reward, done, info = self.env.step(action)
            self.env.render()
            if done:
                obs = self.env.reset()
        self.env.close()
    
    
    def train(self):
        if self.args.model in ['PPO', 'DEFAULT_HER']:
            print(self.env.action_space)
            obs = self.env.reset()

            # print(obs['observation'].shape[0])
            # print(obs['desired_goal'].shape[0])
            # print(self.env.initial_state.qpos)
            # print(self.env.step(self.env.action_space.sample()))
            # print(len(self.env.action_space.sample()))
            self.model.learn(total_timesteps=self.args.epoch_num)

        elif self.args.model == 'DDPG_HER':
            print("Start testing DDPG_HER")
            self.model.learn()
            
    
    def test(self):
        rewards = []
        for _ in range(200):
            obs = self.env.reset()
            eps_reward = 0
            for i in range(1000):
                action, _states = self.model.predict(obs, deterministic=True)
                obs, reward, done, info = self.env.step(action)
                eps_reward += reward               
                if done:
                    break
            rewards.append(reward)
        total_reward = np.mean(rewards)
        print("Total reward: {}".format(total_reward))
        self.env.close()
        return total_reward

       

    def save_model(self):
        if self.args.model in ['PPO', 'DEFAULT_HER']:
            self.model.save(os.path.join(self.args.output_path,self.args.model,self.args.model+'_trash'))
