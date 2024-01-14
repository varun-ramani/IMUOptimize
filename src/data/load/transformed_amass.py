"""
This file provides data loading capabilities
"""

import torch
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
import random

class AMASSDataset(Dataset):
    def __init__(self, amass_dir: str, shuffle=True, num_sensors=24):
        self.amass_dir = Path(amass_dir)
        self.sequences = list(self.amass_dir.glob('**/seq*/'))
        if shuffle:
            random.shuffle(self.sequences)
        self.num_sensors = num_sensors

    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, index):
        poses, vacc, vrot = (
            torch.load(self.sequences[index] / file) for file in ('poses.pt', 'accelerations.pt', 'rotations.pt')
        )

        poses = poses.view(-1, 24 * 3 * 3)
        vacc = vacc.view(-1, self.num_sensors * 3)
        vrot = vrot.view(-1, self.num_sensors * 3 * 3)

        inputs = torch.hstack((vacc, vrot))
        outputs = poses

        return inputs, outputs