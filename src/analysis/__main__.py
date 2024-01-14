from sys import argv
import utils
from rich import print
from dataclasses import dataclass
from typing import List
from torch import nn
from torch.optim import SGD, Optimizer
from model.full_ds.birnn import StepOneRNN
from model.workflow import load_train_context, find_latest_checkpoint
from .error_evaluation import evaluate_mean_per_joint_error
from rich.table import Table

# start by parsing the arguments
def acquire_argument(arg_name, is_boolean=False, default=None):
    if is_boolean:
        return arg_name in argv
    else:
        try:
            i = argv.index(arg_name)
            return argv[i + 1]
        except ValueError as e:
            if default is not None:
                return default
            utils.log_error(f"Missing mandatory parameter {arg_name}")
            exit(-1)
        except IndexError as e:
            utils.log_error(f"No actual argument passed to {arg_name}")
            exit(-1)

@dataclass
class Args:
    input_ds: str
    output_dir: str
    checkpoints_dir: str
    smpl_model: str
    model: str
    stage: str
    subset_size: int

args = Args(
    input_ds=acquire_argument('--data'),
    checkpoints_dir=acquire_argument('--checkpoints'),
    output_dir=acquire_argument('--output'),
    model=acquire_argument('--model'), # should be transformer or birnn
    smpl_model=acquire_argument('--smpl-model'), # don't get this confused with the --model
    stage=acquire_argument('--stage'), # should be full or optim
    subset_size=int(acquire_argument('--subset', default=50))
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
        pass
    else:
        utils.log_error(f"Invalid stage {args.stage}")
        exit(-1)
    optimizer = SGD(net.parameters(), lr=0.005)
    criterion = nn.MSELoss()
else:
    utils.log_error(f"Invalid model type {args.model}")
    exit(-1)

if find_latest_checkpoint(args.checkpoints_dir) is None:
    utils.log_error(f"No checkpoint at '{args.checkpoints_dir}'.")
    exit(-1)

load_train_context(args.checkpoints_dir, net, optimizer)

crit_score, mpje_score = evaluate_mean_per_joint_error(net, criterion, args.smpl_model, args.input_ds, subset_size=args.subset_size)
pos_err, loc_rot_err, global_rot_err = mpje_score

results_table = Table(title=f"{net.__class__.__name__} Evaluation")

results_table.add_column("Metric")
results_table.add_column("Score")

results_table.add_row(str(criterion), str(crit_score.detach().numpy()))
results_table.add_row("Positional Error", str(pos_err.detach().numpy()))
results_table.add_row("Local Rotation Error", str(loc_rot_err.detach().numpy()))
results_table.add_row("Global Rotation Error", str(global_rot_err.detach().numpy()))

utils.console.print(results_table)