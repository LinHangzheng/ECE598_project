import numpy as np
import gym

def random_policy(observation, desired_goal, env):
    # Here you would implement your smarter policy. In this case,
    # we just sample random actions.
    return env.action_space.sample()