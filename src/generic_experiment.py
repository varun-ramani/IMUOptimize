"""
This file doesn't actually do that much. It just returns the optimizer,
criterion, and network given one of the following experiment configs:

full/transformer: train/evaluate a transformer on the full dataset.
optim/transformer: train/evaluate a transformer on minimized IMU dataset.
full/birnn: train/evaluate a biRNN on the full dataset
optim/birnn: train/evaluate a biRNN on minimized IMU dataset.
"""

from torch import nn
from torch.optim import Optimizer, AdamW, SGD
from model import StepOneTransformer, StepOneRNN, StepTwoRNN, StepTwoTransformer
import utils

def net_optim_crit(args):
    net: nn.Module = None
    criterion: nn.Module = None
    optimizer: Optimizer = None
    if args.model == 'transformer': 
        if args.stage == 'full':
            net = StepOneTransformer()
        elif args.stage == 'optim': 
            net = StepTwoTransformer()
        else:
            utils.log_error(f"Invalid stage {args.stage}")
            exit(-1)
        # optimizer = AdamW(net.parameters())
        optimizer = SGD(net.parameters(), lr=0.005)
        criterion = nn.MSELoss()

    elif args.model == 'birnn': 
        if args.stage == 'full':
            net = StepOneRNN(num_hidden=1024)
        elif args.stage == 'optim': 
            net = StepTwoRNN(num_hidden=1024)
        else:
            utils.log_error(f"Invalid stage {args.stage}")
            exit(-1)
        optimizer = SGD(net.parameters(), lr=0.005)
        criterion = nn.MSELoss()
    else:
        utils.log_error(f"Invalid model type {args.model}")
        exit(-1)

    return net, optimizer, criterion
