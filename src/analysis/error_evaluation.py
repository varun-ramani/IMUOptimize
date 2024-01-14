from articulate.evaluator import *
from torch import nn
import torch
from torch.utils.data import DataLoader, SubsetRandomSampler
from data.load import AMASSDataset
import utils
from rich.progress import Progress, track
from rich import print

device = utils.torch_device

def evaluate_mean_per_joint_error(
    net: nn.Module,
    crit: nn.Module,
    smpl_model: str,
    eval_dataset: str,
    subset_size=50
):
    """
    Evaluates the mean per joint error on a dataset basis.

    - net: Model to evaluate. Needs to have checkpoint loaded.
    - smpl_model: Path to SMPL model (not SMPL-H, but SMPL)
    - eval_dataset: Path to AMASS dataset or subdirectory
    """

    net.to(device)
    evaluator = MeanPerJointErrorEvaluator(smpl_model, device=device)
    dataset = AMASSDataset(eval_dataset)
    loader = DataLoader(dataset, batch_size=1, pin_memory=True, shuffle=True)

    # start off by building a subset of the requested size. this way, we can
    # throw it all onto the GPU at once.
    X_vals = []
    Y_vals = []
    Y_preds = []
    for _, (x, y) in track(
        zip(range(subset_size), loader), 
        console=utils.console, 
        description="Running inference", 
        total=subset_size
    ):
        X_vals.append(x)
        Y_vals.append(y)
        Y_preds.append(net(x))

    Y = torch.hstack(Y_vals)
    Y_pred = torch.hstack(Y_preds)
    crit_score = crit(Y, Y_pred)

    Y_rep = Y.view(-1, 24, 3, 3)
    Y_pred_rep = Y_pred.view(-1, 24, 3, 3)
    evaluator_score = evaluator(Y_rep, Y_pred_rep)

    return crit_score, evaluator_score