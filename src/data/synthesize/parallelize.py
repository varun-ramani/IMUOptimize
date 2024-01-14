"""
Computes a synthesized version of the provided dataset in a massively parallel way.
"""
from data.load.raw_amass import RawAMASSDataset
from torch.utils.data import DataLoader
from pathlib import Path
import shutil
import utils
import articulate
from rich import print
from rich.progress import track
import torch
from .pipeline import generate_synthesized_sample
from multiprocessing.pool import ThreadPool

def sample_map_target(payload):
    """
    When we're mapping over the dataset using our cores, we'll leverage this.
    """
    samp_id, sample, model, requested_joints, keep_subdirectories, input_root, output_root, smooth_n = payload
    seq_path, seq_data = sample
    seq_path = seq_path[0] # because for some reason, it's a tuple here
    (poses, trans, betas, 
        joint, imu_acc, imu_rot) = generate_synthesized_sample(
        model, seq_data, requested_joints, smooth_n
    )
    seq_path_obj = Path(seq_path)
    parent_directory = Path(str(seq_path).replace(str(input_root), '')).parts[1]
    if keep_subdirectories:
        output_dir = Path(output_root) / parent_directory
    else:
        output_dir = Path(output_root)

    if not output_dir.exists():
        output_dir.mkdir(parents=True)

    seq_out_path = output_dir / f'seq{samp_id}'
    seq_out_path.mkdir()

    torch.save(poses, seq_out_path / 'poses.pt')
    torch.save(trans, seq_out_path / 'trans.pt')
    torch.save(betas, seq_out_path / 'betas.pt')
    torch.save(joint, seq_out_path / 'joint.pt')
    torch.save(imu_acc, seq_out_path / 'accelerations.pt')
    torch.save(imu_rot, seq_out_path / 'rotations.pt')

    return seq_out_path

def produce_synthetic_dataset(input_ds, output_ds, smpl_model_path, desired_joints=None, keep_subdirectories=True, purge_existing=False):
    """
    Creates a synthetic dataset.

    input_ds: the input AMASS dataset
    output_ds: the folder to write to (will be created if doesn't exist already)
    desired_joints: array of joints that we want
    keep_subdirectories: whether to keep structure of AMASS dataset or just flatten hierarchy
    """
    input_data = RawAMASSDataset(input_ds)
    # input_data = Subset(input_data, range(10))
    input_data_loader = DataLoader(input_data, shuffle=True)

    # establish a fresh copy of the output directory 
    output_folder = Path(output_ds)
    if output_folder.exists():
        if not purge_existing:
            utils.log_error(f"output {output_ds} already exists and purge_existing was not specified.")
            return -1
        else:
            shutil.rmtree(output_folder)
            utils.log_info(f"Purging directory {output_ds} so it can be recreated.")
    
    utils.log_info(f"Created {output_ds}")
    output_folder.mkdir()

    smpl_model = articulate.ParametricModel(smpl_model_path)

    with ThreadPool(8) as pool:
        tasks = ((i, sample, smpl_model, desired_joints, keep_subdirectories, input_ds, output_ds, 4) for i, sample in enumerate(input_data_loader))
        for seq in track(pool.imap_unordered(sample_map_target, tasks), total=len(input_data_loader), description="Synthesizing dataset..."):
            print(seq)