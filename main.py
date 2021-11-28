from options import parse_options
import logging as log
from trainer import Trainer
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
    # trainer.render()
    trainer.test()
    
