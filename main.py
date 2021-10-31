import numpy as np
import gym
from policy import random_policy
from options import parse_options
import logging as log

# Set logger display format
log.basicConfig(format='[%(asctime)s] [INFO] %(message)s', 
                datefmt='%d/%m %H:%M:%S',
                level=log.INFO)

if __name__ == "__main__":
    """Main program."""

    args, args_str = parse_options()
    env = gym.make(args.environment_name)
    obs = env.reset()
   
    action = random_policy(obs['observation'], obs['desired_goal'],env)
    obs, reward, done, info = env.step(action)

    # If we want, we can substitute a goal here and re-compute
    # the reward. For instance, we can just pretend that the desired
    # goal was what we achieved all along.
    substitute_goal = obs['achieved_goal'].copy()
    substitute_reward = env.compute_reward( obs['achieved_goal'], substitute_goal, info)
    print('reward is {}, substitute_reward is {}'.format(reward, substitute_reward))