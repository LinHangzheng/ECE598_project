import argparse
import pprint


def parse_options(return_parser=False):
    # New CLI parser
    parser = argparse.ArgumentParser(description='Train Reinforcemnt Learning for ShadowHand robot.')
    
    # Global arguments
    global_group = parser.add_argument_group('global')
    global_group.add_argument('--environment_name', type=str, default='FetchSlide-v1',
                              help='Experiment name.')
    global_group.add_argument('--output_path', type=str, default='_result/',
                              help='output path.')
    global_group.add_argument('--model', type=str, default='DDPG_HER', choices=['PPO', 'DEFAULT_HER','DDPG_HER'], 
                             help='model to be used.')
    global_group.add_argument('--episode', type=int, default=10000, 
                             help='Number of episodes to run the training.')
    global_group.add_argument('--evaluate_per_episode', type=int, default=200, 
                             help='number of episodes between two evaluation.')
    global_group.add_argument('--log_per_episode', type=int, default=50, 
                             help='number of episodes between two loss log.')
    global_group.add_argument('--eval_episode_num', type=int, default=100, 
                             help='number of episodes used for each evaluation.')
    global_group.add_argument('--epoch_num', type=int, default=50, 
                             help='Number of epochs for model update.')
    global_group.add_argument('--target_update_per_epoch', type=int, default=5, 
                             help='number of epochs between two target updates.')
    global_group.add_argument('--batch_size', type=int, default=1024, 
                             help='batch size for model update.')
    # global_group.add_argument('--T', type=int, default=500, 
    #                          help='Total time steps for each rollout in the environment.')
    global_group.add_argument('--buffer_size', type=int, default=1000000, 
                             help='the size of the replay buffer.')
    global_group.add_argument('--seed', type=int, default =0,
                              help='NumPy random seed.')
    global_group.add_argument('--gamma', type=float, default=0.99, 
                                help='discount factor for DDPG.')
    global_group.add_argument('--fail-rate', type=float, default=0.25, 
                                help='discount factor for DDPG.')
    global_group.add_argument('--lr', type=float, default=0.001, 
                                help='learning rate for model update.')

    # Parse and run
    if return_parser:
        return parser
    else:
        return argparse_to_str(parser)


def argparse_to_str(parser):
    """Convert parser to string representation for Tensorboard logging.

    Args:
        parser (argparse.parser): CLI parser
    """

    args = parser.parse_args()

    args_dict = {}
    for group in parser._action_groups:
        group_dict = {a.dest:getattr(args, a.dest, None) for a in group._group_actions}
        args_dict[group.title] = vars(argparse.Namespace(**group_dict))

    pp = pprint.PrettyPrinter(indent=2)
    args_str = pp.pformat(args_dict)
    args_str = f'```{args_str}```'

    return args, args_str
