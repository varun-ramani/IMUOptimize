from sys import argv
import utils
from rich import print
from dataclasses import dataclass
from typing import List
from model.workflow import train_model
from generic_experiment import net_optim_crit

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

net, optimizer, criterion = net_optim_crit(args)

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