def get_env_parameters(env):
    obs = env.reset()
    action_space = env.action_space.shape[0]
    obs_size = obs['observation'].shape[0]
    goal_size = obs['desired_goal'].shape[0]
    env_params = {'action': action_space,
                  'observation': obs_size,
                  'goal': goal_size,
                  'action_max': env.action_space.high[0]}
    return env_params