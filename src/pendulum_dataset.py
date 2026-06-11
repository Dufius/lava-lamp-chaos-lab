"""
PyTorch Dataset for double-pendulum trajectory data.

Each sample is a (context, target) pair of Cartesian joint positions.
The model receives a window of `seq_len` past positions and must predict
the next `horizon` positions auto-regressively during evaluation.
"""

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

from src.envs.double_pendulum import to_cartesian


class PendulumDataset(Dataset):
    def __init__(self, trajectories, seq_len=30, horizon=1):
        """
        Parameters
        ----------
        trajectories : list of np.ndarray [T, 4]  (angle-space states)
        seq_len      : number of context timesteps fed to the RNN
        horizon      : number of future steps to predict (default 1 for training)
        """
        self.seq_len = seq_len
        self.horizon = horizon
        self.samples = []

        for traj in trajectories:
            cart = to_cartesian(traj).astype(np.float32)  # [T, 4]
            T = len(cart)
            for i in range(T - seq_len - horizon + 1):
                context = cart[i : i + seq_len]  # [seq_len, 4]
                target = cart[i + seq_len : i + seq_len + horizon]  # [horizon, 4]
                self.samples.append((context, target))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        context, target = self.samples[idx]
        return torch.from_numpy(context), torch.from_numpy(target)


def make_dataloaders(trajectories, seq_len=30, horizon=1, batch_size=256, val_frac=0.1):
    """Split trajectories into train/val and return DataLoaders."""
    split = max(1, int(len(trajectories) * val_frac))
    train_trajs = trajectories[split:]
    val_trajs = trajectories[:split]

    train_ds = PendulumDataset(train_trajs, seq_len, horizon)
    val_ds = PendulumDataset(val_trajs, seq_len, horizon)

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, num_workers=0
    )
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0)
    return train_loader, val_loader
