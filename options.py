import argparse
import pprint


def parse_options(return_parser=False):
    # New CLI parser
    parser = argparse.ArgumentParser(description='Train Reinforcemnt Learning for ShadowHand robot.')
    
    # Global arguments
    global_group = parser.add_argument_group('global')
    global_group.add_argument('--environment-name', type=str, default='FetchSlide-v1',
                              help='Experiment name.')
    global_group.add_argument('--output-path', type=str, default='_result/',
                              help='output path.')
    global_group.add_argument('--model', type=str, default='DEFAULT_HER', choices=['PPO', 'DEFAULT_HER'], 
                             help='model to be used.')
    global_group.add_argument('--episode', type=int, default=10000, 
                             help='Number of episodes to run the training.')
    global_group.add_argument('--epoch_num', type=int, default=50, 
                             help='Number of epochs for model update.')
    global_group.add_argument('--batch_size', type=int, default=512, 
                             help='batch size for model update.')
    global_group.add_argument('--T', type=int, default=500, 
                             help='Total time steps for each rollout in the environment.')
    global_group.add_argument('--buffer-size', type=int, default=10000, 
                             help='the size of the replay buffer.')
    global_group.add_argument('--seed', type=int, default =0,
                              help='NumPy random seed.')
    global_group.add_argument('--actor-hidden', type=int, nargs=3, default=[128, 64, 32], 
                                help='hidden layers size of actor.')
    global_group.add_argument('--critic-hidden', type=int, nargs=3, default=[128, 64, 32], 
                                help='hidden layers size of actor.')
    global_group.add_argument('--gamma', type=float, default=0.99, 
                                help='discount factor for DDPG.')


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
