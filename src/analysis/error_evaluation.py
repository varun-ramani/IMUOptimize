from articulate.evaluator import *
from torch import nn
import torch
from torch.utils.data import DataLoader, SubsetRandomSampler
from data.load import AMASSDataset
import utils
from rich.progress import Progress, track
from rich import print

torch_device = utils.torch_device

def evaluate_mean_per_joint_error(
    net: nn.Module,
    crit: nn.Module,
    smpl_model: str,
    eval_dataset: str,
    subset_size=50,
    num_sensors=24
):
    """
    Evaluates the mean per joint error on a dataset basis.

    - net: Model to evaluate. Needs to have checkpoint loaded.
    - smpl_model: Path to SMPL model (not SMPL-H, but SMPL)
    - eval_dataset: Path to AMASS dataset or subdirectory
    - criterion: loss function
    - subset_size: size of subset of data to evaluate on.
    """

    net.to(torch_device)
    evaluator = MeanPerJointErrorEvaluator(smpl_model, device=torch_device)
    dataset = AMASSDataset(eval_dataset, num_sensors, ds_type='test')
    loader = DataLoader(dataset, batch_size=1, shuffle=True)

    net.eval()

    # run inference. we could definitely do this entire thing in one fell swoop
    # on the GPU, but we don't since that would mess with the LSTM's hidden
    # state.
    crit_scores = []
    eval_scores = []

    for _, (x, y) in track(
        zip(range(subset_size), loader), 
        console=utils.console, 
        description="Scoring model", 
        total=subset_size,
    ):
        x, y = x.to(torch_device), y.to(torch_device)
        y_pred = net(x)

        crit_score = crit(y, y_pred).detach()
        evaluator_score = evaluator(y, y_pred).detach()

        crit_scores.append(crit_score)
        eval_scores.append(evaluator_score)
    
    return torch.tensor(crit_scores).mean().cpu(), torch.tensor(eval_scores).mean().cpu()