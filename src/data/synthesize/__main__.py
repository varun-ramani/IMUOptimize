from .parallelize import produce_synthetic_dataset
from sys import argv
import utils
from rich import print
from dataclasses import dataclass
from typing import List

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
    output_ds: str
    model: str
    joints: List[int]
    purge_existing: bool
    keep_subdirectories: bool


args = {
    '--input': acquire_argument('--input'),
    '--output': acquire_argument('--output'),
    '--model': acquire_argument('--model'),
    '--joints': acquire_argument('--joints').split(),
    '--purge-existing': acquire_argument('--purge-existing', is_boolean=True),
    '--keep-subdirectories': acquire_argument('--keep-subdirectories', is_boolean=True)
}

args = Args(
    input_ds=acquire_argument('--input'),
    output_ds=acquire_argument('--output'),
    model=acquire_argument('--model'),
    joints=[int(value) for value in acquire_argument('--joints').split()],
    purge_existing=acquire_argument('--purge-existing', is_boolean=True),
    keep_subdirectories=acquire_argument('--keep-subdirectories', is_boolean=True)
)

produce_synthetic_dataset(
    args.input_ds,
    args.output_ds,
    args.model,
    args.joints,
    args.purge_existing,
    args.keep_subdirectories
)