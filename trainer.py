import torch
import numpy as np
import gym
import os
import logging as log
from stable_baselines3 import PPO

class Trainer(object):
    def __init__(self, args, args_str):
        self.args = args 
        self.args_str = args_str
        self.use_cuda = torch.cuda.is_available()
        self.device = torch.device('cuda' if self.use_cuda else 'cpu')
        device_name = torch.cuda.get_device_name(device=self.device)
        log.info(f'Using {device_name} with CUDA v{torch.version.cuda}')

        self.build_path()
        self.set_env()
        self.set_network()


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
    
    
    def render(self):
        for i in range(1000):
            action, _states = self.model.predict(obs, deterministic=True)
            obs, reward, done, info = self.env.step(action)
            self.env.render()
            if done:
                obs = self.env.reset()
        self.env.close()
    
    
    def train(self):
        if self.args.model == 'PPO':
            self.model.learn(total_timesteps=self.args.epochs)
    

    def save_model(self):
        if self.args.model == 'PPO':
            self.model.save(os.path.join(self.args.output_path,'PPO','PPO_trash'))
