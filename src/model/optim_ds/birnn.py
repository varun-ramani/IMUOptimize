from torch import nn
from pathlib import Path
from torch.utils.data import DataLoader
from torch.nn import MSELoss

class StepTwoRNN(nn.Module):
    def __init__(self, num_hidden=1024):
        super(StepTwoRNN, self).__init__()
        self.linear = nn.Linear(6 * (3 + 9), num_hidden)
        self.lstm = nn.LSTM(num_hidden, num_hidden, 2, bidirectional=True, batch_first=True)
        self.linear2 = nn.Linear(num_hidden * 2, (24 * 9))

    def forward(self, x):
        x = self.linear(x)
        x, _ = self.lstm(x)
        x = self.linear2(x)
        return x
    

