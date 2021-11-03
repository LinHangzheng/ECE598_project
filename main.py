import numpy as np
import gym
from models.policy import random_policy
from options import parse_options
import logging as log
import torch
from trainer import Trainer
import os
from stable_baselines3 import PPO

# Set logger display format
log.basicConfig(format='[%(asctime)s] [INFO] %(message)s', 
                datefmt='%d/%m %H:%M:%S',
                level=log.INFO)

if __name__ == "__main__":
    """Main program."""

    args, args_str = parse_options()
    log.info(f'Parameters: \n{args_str}')
    log.info(f'Training on model {args.model}')
    trainer = Trainer(args, args_str)
    trainer.train()
    trainer.save_model()
    trainer.render()


    # If we want, we can substitute a goal here and re-compute
    # the reward. For instance, we can just pretend that the desired
    # goal was what we achieved all along.
    # substitute_goal = obs['achieved_goal'].copy()
    # substitute_reward = env.compute_reward( obs['achieved_goal'], substitute_goal, info)
    # print('reward is {}, substitute_reward is {}'.format(reward, substitute_reward))