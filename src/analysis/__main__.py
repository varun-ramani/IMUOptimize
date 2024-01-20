from sys import argv
import utils
from rich import print
from dataclasses import dataclass
from typing import List
from torch import nn
from model.workflow import load_train_context, find_latest_checkpoint
from .error_evaluation import evaluate_mean_per_joint_error
from .feature_ablation import run_ablation
import pandas as pd
from pathlib import Path
from datetime import datetime
import json
import matplotlib.pyplot as plt
from generic_experiment import net_optim_crit

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
    num_sensors: int
    no_eval: bool
    no_analyze: bool
    recurse: bool

args = Args(
    input_ds=acquire_argument('--data'),
    checkpoints_dir=acquire_argument('--checkpoints'),
    output_dir=acquire_argument('--output'),
    model=acquire_argument('--model'), # should be transformer or birnn
    smpl_model=acquire_argument('--smpl-model'), # don't get this confused with the --model
    stage=acquire_argument('--stage'), # should be full or optim
    subset_size=int(acquire_argument('--subset', default=50)),
    num_sensors=int(acquire_argument('--num-sensors', default=24)),
    no_analyze=acquire_argument('--no-analyze', is_boolean=True),
    no_eval=acquire_argument('--no-eval', is_boolean=True),
    recurse=acquire_argument('--recurse', is_boolean=True)
)

net, optimizer, criterion = net_optim_crit(args)

if find_latest_checkpoint(args.checkpoints_dir) is None:
    utils.log_error(f"No checkpoint at '{args.checkpoints_dir}'.")
    exit(-1)

load_train_context(args.checkpoints_dir, net, optimizer)

def importance_plot(input_df, output_path):
    joint_names = [
    'Pelvis', 'L Hip', 'R Hip', 'Spine1', 'L Knee', 'R Knee', 'Spine2',
    'L Ankle', 'R Ankle', 'Spine3', 'L Foot', 'R Foot', 'Neck', 'L Collar',
    'R Collar', 'Head', 'L Shoulder', 'R Shoulder', 'L Elbow', 'R Elbow',
    'L Wrist', 'R Wrist', 'L Hand', 'R Hand'
    ]
    indices_to_body_parts = dict(enumerate(joint_names))
    attr_mean = input_df.abs().mean()

    attr_mean = attr_mean.sort_values(ascending=False)

    # Create a DataFrame with 'joint' and 'fa_importance' columns
    result_df = pd.DataFrame({'joint':attr_mean.index, 'fa_importance': attr_mean.values})
    result_df['joint_names'] = [indices_to_body_parts[int(i)] for i in result_df.joint]
    # Save the DataFrame to CSV
    outfile = output_path / 'joint_importance_sorted_fa.csv'
    result_df.to_csv(outfile, index=False)
    utils.log_info(f"Saved sorted feature analysis of joints to {outfile}")
    # Plot the data
    plt.figure(figsize=(10, 6))
    plt.bar(result_df['joint_names'], result_df['fa_importance'])
    # plt.xlabel('Joint Index')
    plt.ylabel('Feature Ablation Importance',fontsize = 13)
    plt.title(f'Feature Ablation Importance Sorted by Joint - Dataset', fontsize = 14)
    # Set x-axis tick labels using the dictionary
    # plt.xticks(range(len(y_values)), [indices_to_body_parts[index] for index in range(len(y_values))], rotation=45, ha='right')

    plt.xticks(rotation=45, ha='right', fontsize = 13)  # Rotate x-axis labels for better readability
    plt.tight_layout()

    # Save the plot as an image
    outfile = output_path / 'joint_importance_sorted_fa.png'
    plt.savefig(outfile)
    utils.log_info(f"Saved plot of sorted feature analysis of joints to {outfile}")



def run_analysis(input_ds, output_path):
    if not output_path.exists():
        output_path.mkdir(parents=True)

    utils.log_info(f"Analysis for {input_ds}")

    if not args.no_eval:
        utils.log_info("Now running basic model evaluation")

        crit_score, mpje_score = evaluate_mean_per_joint_error(net, criterion, args.smpl_model, input_ds, subset_size=args.subset_size, num_sensors=args.num_sensors)
        pos_err, loc_rot_err, global_rot_err = mpje_score

        data = {
            'crit_type': str(criterion),
            'crit_score': crit_score.detach().item(),
            'pos_err': pos_err.detach().item(),
            'loc_rot_err': loc_rot_err.detach().item(),
            'global_rot_err': global_rot_err.detach().item()
        }

        outfile = output_path / 'evaluation.json'
        with open(outfile, 'w') as fp:
            json.dump(data, fp, indent=4)
            utils.log_info(f"Wrote evaluation to {outfile}")
        # now also write the data to a LaTeX table by converting it to a Pandas DF and using to_latex
        eval_df = pd.DataFrame(list(data.items()), columns=['Attribute', 'Value'])
        eval_df['Value'] = eval_df['Value'].apply(lambda x: format(x, '.3f') if isinstance(x, (float, int)) else x)
        # Get the last directory name (= database name)
        ds_name = output_path.name
        caption = f"Evaluation of {args.stage} {args.model} model on {ds_name} dataset"
        label = f"eval_{args.stage}_{args.model}_{ds_name}"
        latex_table = eval_df.to_latex(index=False, caption = caption, label = label)
        outfile = output_path / 'evaluation.tex'
        with open(outfile, 'w') as f:
            f.write(latex_table)
            utils.log_info(f"Wrote evaluation to a LaTeX table {outfile}")
        # now also write the data to a png table
        outfile = output_path / 'evaluation.png'
        fig, ax = plt.subplots(figsize=(6, 3))
        ax.axis('off')  # Hide axes
        ax.table(cellText=eval_df.values, colLabels=eval_df.columns, cellLoc='center', loc='center')
        ax.set_title(caption) 
        plt.savefig(outfile, bbox_inches='tight', pad_inches=0.1)
        utils.log_info(f"Saved evaluation to a figure {outfile}")

        


    if not args.no_analyze:
        utils.log_info("Now running feature ablation to find most important IMUs")
        result = run_ablation(net, input_ds, args.subset_size, args.num_sensors)
        df = pd.DataFrame(result)
        outfile = output_path / 'feature_analysis.csv'
        df.to_csv(outfile, index=None)
        utils.log_info(f"Wrote feature analysis to {outfile}")
        
        # now produce a boxplot from the data 
        # produce a bargraph from the data
        # produce a boxplot that's sorted 
        # produce a bargraph that's sorted
        importance_plot(df, output_path)
        

outputs_dir = Path(args.output_dir)
timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
output_path = outputs_dir / f'analysis_{timestamp}'

input_path = Path(args.input_ds)
if args.recurse:
    for subdirectory in input_path.glob('*/'):
        local_output_path = output_path / subdirectory.name
        run_analysis(subdirectory, local_output_path)

run_analysis(input_path, output_path)
