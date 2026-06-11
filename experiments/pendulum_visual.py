"""
Phase 1: visual encoder + pretrained GRU weights.

Trains LightEncoder + LightDecoder around a frozen (or fine-tunable) GRU
that was already trained on Cartesian coordinates in Phase 0.

Loss = alpha * recon_loss + (1 - alpha) * pred_loss

Usage
-----
python -m experiments.pendulum_visual \\
    --rnn-checkpoint runs/gru_best.pt \\
    --epochs 40 --alpha 0.3
"""

import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import Dataset, DataLoader

from src.envs.double_pendulum import generate_trajectories, to_cartesian
from src.envs.render import render_trajectory
from src.models.light_encoder import PendulumVideoPredictor
from src.models.rnn_predictor import GRUPredictor

# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------


class PendulumVideoDataset(Dataset):
    """Eager dataset: pre-renders all frames at init, getitem is O(1) tensor slice."""

    def __init__(self, trajectories, seq_len=30, img_size=64):
        self.seq_len = seq_len
        self.samples = []

        for traj in trajectories:
            cart = to_cartesian(traj)
            frames = render_trajectory(cart, size=img_size)  # [T, 1, H, W]
            T = len(frames)
            for i in range(T - seq_len - 1):
                ctx = frames[i : i + seq_len]   # view [seq_len, 1, H, W]
                tgt = frames[i + seq_len]        # view [1, H, W]
                self.samples.append((ctx, tgt))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


# ---------------------------------------------------------------------------
# Train / validate
# ---------------------------------------------------------------------------


def train_epoch(model, loader, optimizer, device, alpha):
    model.train()
    total = 0.0
    for ctx, tgt in loader:
        ctx = ctx.to(device)
        tgt = tgt.to(device)
        optimizer.zero_grad()
        _, recon_loss, pred_loss = model(ctx, tgt)
        loss = alpha * recon_loss + (1 - alpha) * pred_loss
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total += loss.item()
    return total / len(loader)


@torch.no_grad()
def validate(model, loader, device, alpha):
    model.eval()
    total = 0.0
    for ctx, tgt in loader:
        ctx = ctx.to(device)
        tgt = tgt.to(device)
        _, recon_loss, pred_loss = model(ctx, tgt)
        total += (alpha * recon_loss + (1 - alpha) * pred_loss).item()
    return total / len(loader)


# ---------------------------------------------------------------------------
# Horizon decay (pixel MSE)
# ---------------------------------------------------------------------------


@torch.no_grad()
def eval_horizon_decay(model, test_trajs, seq_len, max_horizon, img_size, device):
    model.eval()
    mse_by_h = np.zeros(max_horizon)
    count = 0

    for traj in test_trajs:
        cart = to_cartesian(traj)
        frames = render_trajectory(cart, size=img_size).to(device)  # [T, 1, H, W]
        T = len(frames)
        if T < seq_len + max_horizon:
            continue

        ctx = frames[:seq_len].unsqueeze(0)  # [1, seq_len, 1, H, W]
        gt = frames[seq_len : seq_len + max_horizon].cpu().numpy()  # [H, 1, H, W]

        preds = model.rollout(ctx, max_horizon).squeeze(0).cpu().numpy()  # [H, 1, H, W]
        mse_by_h += ((preds - gt) ** 2).mean(axis=(1, 2, 3))
        count += 1

    return np.arange(1, max_horizon + 1), mse_by_h / max(count, 1)


# ---------------------------------------------------------------------------
# Visualise a few predictions
# ---------------------------------------------------------------------------


@torch.no_grad()
def save_sample(model, traj, seq_len, img_size, path, device, n_future=8):
    model.eval()
    cart = to_cartesian(traj)
    frames = render_trajectory(cart, size=img_size).to(device)

    ctx = frames[:seq_len].unsqueeze(0)
    preds = model.rollout(ctx, n_future).squeeze(0).cpu()  # [n_future, 1, H, W]
    gt = frames[seq_len : seq_len + n_future].cpu()

    fig, axes = plt.subplots(2, n_future, figsize=(n_future * 1.5, 3))
    for i in range(n_future):
        axes[0, i].imshow(gt[i, 0], cmap="gray", vmin=0, vmax=1)
        axes[0, i].axis("off")
        axes[0, i].set_title(f"+{i+1}", fontsize=7)
        axes[1, i].imshow(preds[i, 0], cmap="gray", vmin=0, vmax=1)
        axes[1, i].axis("off")

    axes[0, 0].set_ylabel("GT", fontsize=8)
    axes[1, 0].set_ylabel("Pred", fontsize=8)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--rnn-checkpoint", default="runs/gru_best.pt")
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--seq-len", type=int, default=30)
    p.add_argument("--img-size", type=int, default=64)
    p.add_argument("--n-train", type=int, default=150)
    p.add_argument("--n-test", type=int, default=20)
    p.add_argument("--t-end", type=float, default=20.0)
    p.add_argument("--dt", type=float, default=0.02)
    p.add_argument("--alpha", type=float, default=0.3, help="recon loss weight")
    p.add_argument("--freeze-rnn", action="store_true", help="freeze RNN weights")
    p.add_argument("--max-horizon", type=int, default=100)
    p.add_argument("--runs-dir", default="runs")
    p.add_argument("--device", default="auto")
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.runs_dir, exist_ok=True)

    device = (
        torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if args.device == "auto"
        else torch.device(args.device)
    )
    print(f"Device: {device}")

    # ------------------------------------------------------------------
    # Data
    # ------------------------------------------------------------------
    print("Simulating + rendering trajectories...")
    all_trajs = generate_trajectories(
        n=args.n_train + args.n_test, t_end=args.t_end, dt=args.dt, seed=1
    )
    train_trajs = all_trajs[: args.n_train]
    test_trajs = all_trajs[args.n_train :]

    print("  Building video datasets (rendering all frames)...")
    train_ds = PendulumVideoDataset(train_trajs, args.seq_len, args.img_size)
    val_ds = PendulumVideoDataset(
        test_trajs[: max(1, len(test_trajs) // 2)], args.seq_len, args.img_size
    )

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False, num_workers=0
    )
    print(f"  train samples: {len(train_ds)} | val samples: {len(val_ds)}")

    # ------------------------------------------------------------------
    # Model — load pretrained GRU
    # ------------------------------------------------------------------
    rnn = GRUPredictor(state_dim=4, hidden_dim=64, num_layers=2, dropout=0.1)
    rnn.load_state_dict(torch.load(args.rnn_checkpoint, map_location="cpu"))
    print(f"Loaded RNN from {args.rnn_checkpoint}")

    model = PendulumVideoPredictor(
        rnn, img_size=args.img_size, freeze_rnn=args.freeze_rnn
    )
    model.to(device)

    enc_params = sum(p.numel() for p in model.encoder.parameters())
    dec_params = sum(p.numel() for p in model.decoder.parameters())
    rnn_params = sum(p.numel() for p in model.rnn.parameters())
    print(
        f"  Encoder: {enc_params:,}  Decoder: {dec_params:,}  RNN: {rnn_params:,} ({'frozen' if args.freeze_rnn else 'trainable'})"
    )

    # ------------------------------------------------------------------
    # Train
    # ------------------------------------------------------------------
    optimizer = Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-5)
    best_val = float("inf")
    ckpt = os.path.join(args.runs_dir, "visual_best.pt")

    for epoch in range(1, args.epochs + 1):
        tr = train_epoch(model, train_loader, optimizer, device, args.alpha)
        val = validate(model, val_loader, device, args.alpha)
        scheduler.step()

        if val < best_val:
            best_val = val
            torch.save(model.state_dict(), ckpt)

        if epoch % 10 == 0 or epoch == 1:
            print(f"[{epoch:3d}/{args.epochs}] train={tr:.5f}  val={val:.5f}")

    model.load_state_dict(torch.load(ckpt, map_location="cpu"))
    model.to(device)
    print(f"Best val loss: {best_val:.5f}")

    # ------------------------------------------------------------------
    # Sample visualisation
    # ------------------------------------------------------------------
    save_sample(
        model,
        test_trajs[0],
        args.seq_len,
        args.img_size,
        os.path.join(args.runs_dir, "visual_sample.png"),
        device,
    )
    print("Saved: runs/visual_sample.png")

    # ------------------------------------------------------------------
    # Horizon decay
    # ------------------------------------------------------------------
    print(f"Evaluating horizon decay over {args.max_horizon} steps...")
    horizons, mse = eval_horizon_decay(
        model, test_trajs, args.seq_len, args.max_horizon, args.img_size, device
    )
    times = horizons * args.dt

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].plot(times, mse)
    axes[0].set_xlabel("Prediction horizon (s)")
    axes[0].set_ylabel("Pixel MSE")
    axes[0].set_title("Visual model — MSE vs horizon")
    axes[0].grid(True)
    axes[1].semilogy(times, mse + 1e-8)
    axes[1].set_xlabel("Prediction horizon (s)")
    axes[1].set_ylabel("Pixel MSE (log)")
    axes[1].set_title("Visual model — exponential divergence?")
    axes[1].grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(args.runs_dir, "visual_horizon_decay.png"), dpi=150)
    print("Saved: runs/visual_horizon_decay.png")


if __name__ == "__main__":
    main()
