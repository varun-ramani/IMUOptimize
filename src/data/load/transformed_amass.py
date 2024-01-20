"""
This file provides data loading capabilities
"""

import torch
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
import random
import utils

class AMASSDataset(Dataset):
    def __init__(self, amass_dir: str, num_sensors=24, ds_type='train'):
        self.amass_dir = Path(amass_dir)
        
        # first, acquire all the sequences. shuffle the result with the same
        # seed.
        self.all_sequences = list(self.amass_dir.glob('**/seq*/'))
        random.Random(11).shuffle(self.all_sequences)

        # then, partition the dataset into val, test, and train.
        val_cutoff = int(len(self.all_sequences) * 0.02)
        test_cutoff = int(len(self.all_sequences) * 0.1) + val_cutoff
        self.partitioned = {
            'validation': self.all_sequences[:val_cutoff],
            'test': self.all_sequences[val_cutoff:test_cutoff],
            'train': self.all_sequences[test_cutoff:]
        }

        # set the sequence to the correct one for backwards compat
        self.sequence = self.partitioned[ds_type]

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