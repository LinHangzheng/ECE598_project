import argparse
import pprint


def parse_options(return_parser=False):
    # New CLI parser
    parser = argparse.ArgumentParser(description='Train deep implicit 3D geometry representations.')
    
    # Global arguments
    global_group = parser.add_argument_group('global')
    global_group.add_argument('--environment_name', type=str, default='FetchSlide-v1',
                              help='Experiment name.')
    global_group.add_argument('--seed', type=int,
                              help='NumPy random seed.')

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
