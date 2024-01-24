"""
Computes a synthesized version of the provided dataset in a massively parallel way.
"""
from data.load import RawAMASSDataset
from torch.utils.data import DataLoader
from pathlib import Path
import shutil
import utils
import articulate
from rich import print
from rich.progress import track
import torch
from .pipeline import transform_poses_trans, run_kinematics, synthesize_imu_data
from multiprocessing.pool import ThreadPool
from rich.progress import Progress

torch_device = utils.torch_device

def sample_map_target(payload):
    """
    When we're mapping over the dataset using our cores, we'll leverage this.
    """
    samp_id, sample, model, requested_joints, keep_subdirectories, input_root, output_root, smooth_n, progress = payload
    seq_path, seq_data = sample

    seq_path = seq_path[0] # because for some reason, it's a tuple here

    task = progress.add_task(f"Transforming {seq_path}", total=4)

    poses, trans, betas = seq_data

    poses = poses.to(torch_device)
    trans = trans.to(torch_device)
    betas = betas.to(torch_device)

    poses, trans = transform_poses_trans(poses, trans)
    progress.update(task, advance=1)
    grot, joint, vert = run_kinematics(model, poses, trans, betas)
    progress.update(task, advance=1)
    imu_acc, imu_rot = synthesize_imu_data(model, vert, grot, requested_joints, smooth_n)
    progress.update(task, advance=1)

    # sanity check - can we transform everything?
    try:
        poses.view(-1, 24 * 3 * 3).shape
        imu_acc.view(-1, len(requested_joints) * 3).shape
        imu_rot.view(-1, len(requested_joints) * 3 * 3).shape
    except Exception as e:
        utils.log_error("Tried running transformation, but the result had the wrong shape. Below is the error:")
        utils.log_error(e)
        exit(-1)

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

    progress.update(task, visible=False)

    return seq_path

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
    torch.multiprocessing.set_sharing_strategy('file_system')
    input_data_loader = DataLoader(input_data, shuffle=True, num_workers=8)

    # establish a fresh copy of the output directory 
    output_folder = Path(output_ds)
    if output_folder.exists():
        if not purge_existing:
            utils.log_error(f"output {output_ds} already exists and purge_existing was not specified.")
            return -1
        else:
            utils.log_info(f"Purging directory {output_ds} so it can be recreated.")
            shutil.rmtree(output_folder)
    
    utils.log_info(f"Created {output_ds}")
    output_folder.mkdir()

    smpl_model = articulate.ParametricModel(smpl_model_path, device=torch_device)

    if torch_device == torch.device('cpu'):
        num_threads = 8
    else:
        num_threads = 1

    with ThreadPool(num_threads) as pool, Progress(console=utils.console) as progress:
        tasks = ((i, sample, smpl_model, desired_joints, keep_subdirectories, input_ds, output_ds, 4, progress) for i, sample in enumerate(input_data_loader))
        root_task = progress.add_task("Running all transformations", total=len(input_data_loader))

        for sid, seq in enumerate(pool.imap_unordered(sample_map_target, tasks)):
            utils.log_info(f"{sid} / {len(input_data_loader)}: '{seq}'")
            progress.update(root_task, advance=1)

    utils.log_info("Done with all transformations.")