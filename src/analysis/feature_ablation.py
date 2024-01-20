"""
Contains code to run feature ablation - the process of figuring out the most
important (groups) of input features. In this case, IMUs.
"""

import torch
from torch.utils.data import DataLoader
import utils
from data.load import AMASSDataset
from rich.progress import track
from rich import print
from captum.attr import FeatureAblation
from torch import nn
from torch.utils.data import Subset

torch_device = utils.torch_device

class StepOneRNN_SO(nn.Module):
    def __init__(self, original_model):
        super(StepOneRNN_SO, self).__init__()
        self.original_model = original_model

    def forward(self, x):
        # Call the forward method of the original model
        output = self.original_model(x)

        # Extract the first element of the tuple (output_tuple[0]) and squeez it to a 2D (instead of 3D) Tensor
        return output.squeeze(0)

def gen_feature_mask(num_examples, num_groups):
    input_size = 12 * num_groups
    feature_mask = torch.zeros(num_examples, input_size, dtype=torch.long).to(torch_device)
    
    for i in range(num_examples):
        for j in range(num_groups):
            acc_start_ind = j * 3
            acc_end_ind = (j + 1) * 3
            rot_start_ind = (num_groups*3) + j * 9
            rot_end_ind = (num_groups*3) + (j+1) * 9
            feature_mask[i, acc_start_ind:acc_end_ind] = j
            feature_mask[i, rot_start_ind:rot_end_ind] = j
    
    return feature_mask

def calculate_group_means(f_mask, x):
    unique_groups = torch.unique(f_mask)
    num_groups = len(unique_groups)
    group_means = torch.zeros(num_groups, dtype=x.dtype)  # Use dtype=x.dtype to match the input tensor's dtype
    
    for i, group in enumerate(unique_groups):
        binary_mask = torch.where(f_mask == group, torch.tensor(1, dtype=x.dtype), torch.tensor(0, dtype=x.dtype))
        sum_result = torch.sum(x * binary_mask)
        group_means[i] = sum_result / torch.sum(binary_mask)
    
    return group_means

def run_ablation(net, eval_ds, subset_size=50, num_sensors=24):
    """
    Runs feature ablation to determine the most important input IMUs.
    
    - net: input neural network. needs to have checkpoint loaded.
    - eval_ds: path to AMASS dataset or subdirectory.
    - subset_size: size of subset of data to evaluate on.
    - num_sensors: the number of IMUs that the dataset was generated on.
    """

    dataset = AMASSDataset(eval_ds, num_sensors, ds_type='test')
    loader = DataLoader(dataset, batch_size=1, shuffle=True)
    net.to(torch_device)
    net = StepOneRNN_SO(net).to(torch_device)

    all_attr = []

    for index, (x, y) in track(
        zip(range(subset_size), loader), 
        console=utils.console,
        description="Running feature ablation",
        total=min(subset_size, len(loader))
    ):
        x, y = x.to(torch_device), y.to(torch_device)
        x.requires_grad_()
        feature_mask = gen_feature_mask(x.shape[0], num_sensors)
        fa = FeatureAblation(net)
        try:
            res = fa.attribute(
                x,
                target=0,
                feature_mask=feature_mask,
                return_convergence_delta=True
            )
            mean_attr = torch.mean(res, dim=0)
            group_means = calculate_group_means(feature_mask, mean_attr)
            all_attr.append(group_means.detach())
        except Exception as e:
            utils.log_warning(f"Error (probably OOM) on sample of shape {x.shape}")

    return torch.stack(all_attr, dim=0)