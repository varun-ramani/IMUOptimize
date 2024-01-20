import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from pathlib import Path
import utils
from rich.progress import track, Progress
from multiprocessing.pool import ThreadPool

def read_amass_npz(npz_path):
    try:
        cdata = np.load(npz_path)
    except:
        return None
    
    if 'mocap_framerate' not in cdata:
        utils.log_warning(f"'{npz_path}' does not contain mocap_framerate")
        return None

    framerate = int(cdata['mocap_framerate'])
    if framerate == 120: 
        step = 2
    elif framerate == 60 or framerate == 59: 
        step = 1
    else: 
        utils.log_warning(f"'{npz_path}' has a bad framerate of {framerate}")
        return None

    poses = torch.tensor(cdata['poses'][::step].astype(np.float32))
    trans = torch.tensor(cdata['trans'][::step].astype(np.float32))

    data_length = poses.shape[0]   
    if data_length <= 12:
        utils.log_warning(f"'{npz_path}' contains less than 12 frames.")
        return None

    betas = torch.tensor(cdata['betas'][:10].astype(np.float32))
    
    return poses, trans, betas


class RawAMASSDataset(Dataset):
    def __init__(self, amass_directory):
        self.amass_directory = amass_directory

        base_npzs = list(Path(self.amass_directory).glob("**/*.npz"))
        self.npz_files = []

        with ThreadPool(8) as pool, Progress(console=utils.console) as progress:
            root_task = progress.add_task("Filtering bad elements from dataset", total=len(base_npzs))

            for npz, res in pool.imap_unordered(lambda x: (x, read_amass_npz(x)), base_npzs):
                progress.update(root_task, advance=1)
                if res is not None:
                    self.npz_files.append(npz)

    def __len__(self):
        return len(self.npz_files)

    def __getitem__(self, idx):
        if idx >= len(self):
            raise StopIteration

        npz_path = self.npz_files[idx]
        data = read_amass_npz(npz_path)
        poses, trans, betas = data

        return str(npz_path), (poses, trans, betas)