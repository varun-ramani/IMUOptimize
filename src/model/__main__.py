from sys import argv
import utils
from rich import print
from dataclasses import dataclass
from typing import List
from torch import nn
from torch.optim import SGD, Optimizer
from model.full_ds.birnn import StepOneRNN
from model.optim_ds.birnn import StepTwoRNN
from model.workflow import train_model

# start by parsing the arguments
def acquire_argument(arg_name, is_boolean=False):
    if is_boolean:
        return arg_name in argv
    else:
        try:
            i = argv.index(arg_name)
            return argv[i + 1]
        except ValueError as e:
            utils.log_error(f"Missing mandatory parameter {arg_name}")
            exit(-1)
        except IndexError as e:
            utils.log_error(f"No actual argument passed to {arg_name}")
            exit(-1)

@dataclass
class Args:
    input_ds: str
    checkpoints_dir: str
    model: str
    stage: str
    epochs: int
    scratch: bool
    num_sensors: int

args = Args(
    input_ds=acquire_argument('--data'),
    checkpoints_dir=acquire_argument('--checkpoints'),
    model=acquire_argument('--model'), # should be transformer or birnn
    stage=acquire_argument('--stage'), # should be full or optim
    epochs=int(acquire_argument('--epochs')),
    scratch=acquire_argument('--scratch', is_boolean=True),
    num_sensors=int(acquire_argument('--num-sensors')),
)

net: nn.Module = None
criterion: nn.Module = None
optimizer: Optimizer = None
if args.model == 'transformer': 
    if args.stage == 'full':
        pass
    elif args.stage == 'optim': 
        pass
    else:
        utils.log_error(f"Invalid stage {args.stage}")
        exit(-1)
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

train_model(
    net, 
    optimizer,
    criterion,
    args.epochs,
    args.input_ds,
    args.checkpoints_dir,
    args.scratch,
    args.num_sensors
)