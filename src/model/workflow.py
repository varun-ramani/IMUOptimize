"""
Implements routines for saving, loading, and training models.
"""
from pathlib import Path
import utils
import torch
from datetime import datetime
from data.load import AMASSDataset
from rich.progress import track
from torch.utils.data import DataLoader
from rich.progress import Progress
from torch import nn
import itertools

device = utils.torch_device

def find_latest_checkpoint(checkpoints_dir: str):
    """
    Goes to the provided directory and returns the most recent file.

    - checkpoints_dir: directory to load from
    """
    output_path = Path(checkpoints_dir)
    
    # List all files in the directory
    files = [f for f in output_path.glob('*.pth') if f.is_file()]
    
    # Check if there are any files
    if not files:
        return None

    # Sort files by modification time in descending order
    files.sort(key=lambda f: f.stat().st_mtime, reverse=True)

    # Return the most recent file
    return files[0]

def load_train_context(
    checkpoints_dir: str, 
    net: torch.nn.Module, 
    optimizer: torch.optim.Optimizer
):
    """
    Loads the most recent checkpoint.

    - checkpoints_dir: to read checkpoints from
    - net: network to load network state into from checkpoint
    - optimizer: optimizer to load optimizer state into from checkpoint
    """
    checkpoints_path = Path(checkpoints_dir)
    if not checkpoints_path.exists():
        utils.log_error(f"Checkpoints directory {checkpoints_dir} does not exist. Create it and run the script again.")
        exit(-1)
    
    latest_checkpoint = find_latest_checkpoint(checkpoints_path)
    if latest_checkpoint:
        checkpoint = torch.load(latest_checkpoint, map_location=device)
        net.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        utils.log_info(f"Loaded checkpoint '{latest_checkpoint}'")
    else:
        utils.log_warning(f"Couldn't find a checkpoint. We'll have to train from scratch.")

    return net, optimizer

def save_train_context(
    checkpoints_dir: str, 
    net: torch.nn.Module, 
    optimizer: torch.optim.Optimizer, 
    loss: torch.nn.Module
):
    """
    Writes the current network, optimizer, and loss to a timestamped checkpoint.

    - checkpoints_dir: directory to write to
    - net: model
    - optimizer: current optimizer
    - loss: current loss
    """
    checkpoints_dir = Path(checkpoints_dir)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    checkpoint_filename = f'checkpoint_step1_{timestamp}.pth'
    checkpoint_path = checkpoints_dir / checkpoint_filename
    torch.save({
        'model_state_dict': net.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
    }, checkpoint_path)
    utils.log_info(f"Saved '{checkpoint_path}' with loss {loss}")

def train_model(
    net: torch.nn.Module, 
    optimizer: torch.optim.Optimizer, 
    criterion: torch.nn.Module, 
    epochs: int, 
    dataset_dir: str, 
    checkpoints_dir: str, 
    scratch=False,
    num_sensors=24
):
    """
    Supplied with an empty network, optimizer and criterion, will optionally
    load the most recent checkpoint and start training.

    - net: model
    - optimizer: optimizer
    - criterion: criterion
    - epochs: number of epochs to train for
    - dataset_dir: contains transformed AMASS dataset
    - checkpoints_dir: contains the most recent checkpoint, will also write checkpoints here.
    - scratch: if True, does not load the most recent checkpoint. If False, will load the most recent checkpoint if it exists.
    """
    if not scratch:
        load_train_context(checkpoints_dir, net, optimizer)
    
    train_ds = AMASSDataset(dataset_dir, num_sensors=num_sensors, ds_type='train')
    val_ds = AMASSDataset(dataset_dir, num_sensors=num_sensors, ds_type='validation')

    output = Path(checkpoints_dir)

    # we need to ensure that the network has been put into training mode
    net.to(device)
    net.train()

    with Progress(console=utils.console) as progress:
        training_task = progress.add_task("Training model...", total=epochs)

        for epoch in range(epochs):
            torch.multiprocessing.set_sharing_strategy('file_system')
            loader = DataLoader(train_ds, batch_size=1, shuffle=True)
            val_loader = DataLoader(val_ds, batch_size=1, shuffle=True)

            training_losses = torch.zeros(100)
            val_losses = torch.zeros(100)
            loss_index = 0

            epoch_task = progress.add_task(f"Epoch {epoch} / {epochs}", total=len(loader))
            for (index, (x, y)), (val_x, val_y) in zip(enumerate(loader), itertools.cycle(val_loader)):
                progress.update(epoch_task, advance=1)
                x, y = x.to(device), y.to(device)
                val_x, val_y = val_x.to(device), val_y.to(device)

                optimizer.zero_grad()
                y_pred = net(x)
                loss = criterion(y_pred, y)
                loss.backward()
                optimizer.step()

                val_loss = criterion(net(val_x), val_y)

                training_losses[loss_index] = loss.detach()
                val_losses[loss_index] = val_loss.detach()

                loss_index = (loss_index + 1) % 100

                if index % 100 == 0:
                    message = f'e{epoch}/s{index}: train loss {training_losses.mean()}, val loss {val_losses.mean()}'
                    utils.log_info(message)

                if index % 1000 == 0:
                    save_train_context(output, net, optimizer, loss)
                    

            progress.update(epoch_task, visible=False)
            progress.update(training_task, advance=1)

        save_train_context(output, net, optimizer, loss)