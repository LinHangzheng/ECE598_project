def get_env_parameters(env):
    obs = env.reset()
    action_space = env.action_space.shape[0]
    obs_size = obs['observation'].shape[0]
    goal_size = obs['desired_goal'].shape[0]
    return action_space, obs_size, goal_size