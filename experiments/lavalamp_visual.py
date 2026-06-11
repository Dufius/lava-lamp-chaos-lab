"""
Phase 2: train a video predictor on real lava lamp footage.

Unlike Phase 1 there are no Cartesian coordinates — the model learns
dynamics purely from pixel sequences.  The architecture is the same
(LightEncoder → GRU → LightDecoder) but trained end-to-end from scratch.

Loss = alpha * recon_loss + (1 - alpha) * pred_loss

Usage
-----
python -m experiments.lavalamp_visual \\
    --frames data/frames/lavalamp_frames.npy \\
    --epochs 30 --seq-len 20
"""

import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, Dataset

from src.models.light_encoder import LightDecoder, LightEncoder
from src.models.rnn_predictor import GRUPredictor

# ---------------------------------------------------------------------------
# Model — end-to-end video predictor (no pretrained RNN)
# ---------------------------------------------------------------------------


class VideoPredictor(nn.Module):
    """LightEncoder → GRU → LightDecoder trained end-to-end from pixels."""

    def __init__(self, latent_dim=16, hidden_dim=128, img_size=64):
        super().__init__()
        self.encoder = LightEncoder(latent_dim=latent_dim, img_size=img_size)
        self.decoder = LightDecoder(latent_dim=latent_dim, img_size=img_size)
        self.gru = nn.GRU(latent_dim, hidden_dim, num_layers=2, batch_first=True)
        self.head = nn.Linear(hidden_dim, latent_dim)

    def encode_sequence(self, frames):
        """frames: [B, T, 1, H, W] → [B, T, latent_dim]"""
        B, T, C, H, W = frames.shape
        z = self.encoder(frames.view(B * T, C, H, W))
        return z.view(B, T, -1)

    def forward(self, context_frames, target_frame):
        """
        Parameters
        ----------
        context_frames : [B, T, 1, H, W]
        target_frame   : [B, 1, H, W]

        Returns
        -------
        pred_frame  : [B, 1, H, W]
        recon_loss  : pixel MSE on context reconstruction
        pred_loss   : pixel MSE on next-frame prediction
        """
        z_ctx = self.encode_sequence(context_frames)  # [B, T, latent_dim]

        # Reconstruct context frames
        B, T, D = z_ctx.shape
        recon = self.decoder(z_ctx.reshape(B * T, D)).view(
            B, T, 1, *context_frames.shape[3:]
        )
        recon_loss = nn.functional.mse_loss(recon, context_frames)

        # Predict next latent
        out, _ = self.gru(z_ctx)  # [B, T, hidden_dim]
        z_pred = self.head(out[:, -1])  # [B, latent_dim]

        pred_frame = self.decoder(z_pred)  # [B, 1, H, W]
        pred_loss = nn.functional.mse_loss(pred_frame, target_frame)

        return pred_frame, recon_loss, pred_loss

    @torch.no_grad()
    def rollout(self, context_frames, horizon):
        """
        Auto-regressive rollout.

        Parameters
        ----------
        context_frames : [B, T, 1, H, W]
        horizon        : int

        Returns
        -------
        [B, horizon, 1, H, W]
        """
        self.eval()
        B, T, C, H, W = context_frames.shape
        z_ctx = self.encode_sequence(context_frames)  # [B, T, latent_dim]

        preds = []
        _, h = self.gru(z_ctx)  # prime hidden state

        z_last = self.head(self.gru(z_ctx)[0][:, -1])  # [B, latent_dim]
        z_window = z_ctx  # growing context

        for _ in range(horizon):
            out, h = self.gru(z_last.unsqueeze(1), h)  # [B, 1, hidden_dim]
            z_last = self.head(out[:, 0])  # [B, latent_dim]
            frame = self.decoder(z_last)  # [B, 1, H, W]
            preds.append(frame.unsqueeze(1))

        return torch.cat(preds, dim=1)  # [B, horizon, 1, H, W]


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------


class LavaLampDataset(Dataset):
    """Sliding windows over a single long lava lamp sequence."""

    def __init__(self, frames, seq_len=20):
        """
        Parameters
        ----------
        frames  : np.ndarray or torch.Tensor [T, 1, H, W]
        seq_len : int  — context length
        """
        if isinstance(frames, np.ndarray):
            frames = torch.from_numpy(frames)
        self.frames = frames
        self.seq_len = seq_len
        self.n = len(frames) - seq_len - 1

    def __len__(self):
        return self.n

    def __getitem__(self, idx):
        ctx = self.frames[idx : idx + self.seq_len]  # [seq_len, 1, H, W]
        tgt = self.frames[idx + self.seq_len]  # [1, H, W]
        return ctx, tgt


# ---------------------------------------------------------------------------
# Train / validate
# ---------------------------------------------------------------------------


def train_epoch(model, loader, optimizer, alpha, device):
    model.train()
    total = 0.0
    for ctx, tgt in loader:
        ctx, tgt = ctx.to(device), tgt.to(device)
        optimizer.zero_grad()
        _, recon_loss, pred_loss = model(ctx, tgt)
        loss = alpha * recon_loss + (1 - alpha) * pred_loss
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total += loss.item()
    return total / len(loader)


@torch.no_grad()
def validate(model, loader, alpha, device):
    model.eval()
    total = 0.0
    for ctx, tgt in loader:
        ctx, tgt = ctx.to(device), tgt.to(device)
        _, recon_loss, pred_loss = model(ctx, tgt)
        total += (alpha * recon_loss + (1 - alpha) * pred_loss).item()
    return total / len(loader)


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------


@torch.no_grad()
def eval_horizon_decay(model, frames, seq_len, max_horizon, device):
    """Pixel MSE vs prediction horizon on the test sequence."""
    model.eval()
    frames_t = (
        torch.from_numpy(frames).to(device)
        if isinstance(frames, np.ndarray)
        else frames.to(device)
    )
    T = len(frames_t)
    mse = np.zeros(max_horizon)
    count = 0

    for start in range(0, T - seq_len - max_horizon, seq_len):
        ctx = frames_t[start : start + seq_len].unsqueeze(0)  # [1, T, 1, H, W]
        gt = frames_t[start + seq_len : start + seq_len + max_horizon].cpu().numpy()
        preds = model.rollout(ctx, max_horizon).squeeze(0).cpu().numpy()
        mse += ((preds - gt) ** 2).mean(axis=(1, 2, 3))
        count += 1

    return mse / max(count, 1)


@torch.no_grad()
def save_sample(model, frames, seq_len, device, path, n_future=8):
    model.eval()
    frames_t = (
        torch.from_numpy(frames).to(device)
        if isinstance(frames, np.ndarray)
        else frames.to(device)
    )
    ctx = frames_t[:seq_len].unsqueeze(0)
    preds = model.rollout(ctx, n_future).squeeze(0).cpu()
    gt = frames_t[seq_len : seq_len + n_future].cpu()

    fig, axes = plt.subplots(2, n_future, figsize=(n_future * 1.5, 3))
    for i in range(n_future):
        axes[0, i].imshow(gt[i, 0], cmap="gray", vmin=0, vmax=1)
        axes[0, i].axis("off")
        axes[0, i].set_title(f"+{i+1}s" if True else f"+{i+1}", fontsize=7)
        axes[1, i].imshow(preds[i, 0], cmap="gray", vmin=0, vmax=1)
        axes[1, i].axis("off")
    axes[0, 0].set_ylabel("GT", fontsize=8)
    axes[1, 0].set_ylabel("Pred", fontsize=8)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Saved: {path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--frames", default="data/frames/lavalamp_frames.npy")
    p.add_argument("--runs-dir", default="runs")
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--seq-len", type=int, default=20)
    p.add_argument("--latent-dim", type=int, default=16)
    p.add_argument("--hidden-dim", type=int, default=128)
    p.add_argument("--alpha", type=float, default=0.3)
    p.add_argument("--train-split", type=float, default=0.8)
    p.add_argument("--max-horizon", type=int, default=50)
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
    print(f"Loading frames from {args.frames} ...")
    all_frames = np.load(args.frames)  # [T, 1, H, W]
    T_total = len(all_frames)
    img_size = all_frames.shape[-1]
    T_train = int(T_total * args.train_split)

    train_frames = all_frames[:T_train]
    test_frames = all_frames[T_train:]
    print(
        f"  total={T_total}  train={T_train}  test={T_total-T_train}  size={img_size}×{img_size}"
    )

    train_ds = LavaLampDataset(train_frames, args.seq_len)
    val_ds = LavaLampDataset(test_frames, args.seq_len)
    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False, num_workers=0
    )
    print(f"  train windows={len(train_ds)}  val windows={len(val_ds)}")

    # ------------------------------------------------------------------
    # Model
    # ------------------------------------------------------------------
    model = VideoPredictor(
        latent_dim=args.latent_dim,
        hidden_dim=args.hidden_dim,
        img_size=img_size,
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"VideoPredictor: {n_params:,} parameters")

    optimizer = Adam(model.parameters(), lr=args.lr)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-5)
    best_val = float("inf")
    ckpt = os.path.join(args.runs_dir, "lavalamp_best.pt")

    # ------------------------------------------------------------------
    # Train
    # ------------------------------------------------------------------
    for epoch in range(1, args.epochs + 1):
        tr = train_epoch(model, train_loader, optimizer, args.alpha, device)
        val = validate(model, val_loader, args.alpha, device)
        scheduler.step()

        if val < best_val:
            best_val = val
            torch.save(model.state_dict(), ckpt)

        if epoch % 5 == 0 or epoch == 1:
            print(f"[{epoch:3d}/{args.epochs}] train={tr:.5f}  val={val:.5f}")

    print(f"Best val loss: {best_val:.5f}")
    model.load_state_dict(torch.load(ckpt, map_location="cpu"))
    model.to(device)

    # ------------------------------------------------------------------
    # Results
    # ------------------------------------------------------------------
    save_sample(
        model,
        test_frames,
        args.seq_len,
        device,
        os.path.join(args.runs_dir, "lavalamp_sample.png"),
    )

    print(f"Evaluating horizon decay over {args.max_horizon} steps ...")
    mse = eval_horizon_decay(model, test_frames, args.seq_len, args.max_horizon, device)
    dt = 1.0 / 5.0  # 5 fps → 0.2s per step
    times = np.arange(1, args.max_horizon + 1) * dt

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].plot(times, mse)
    axes[0].set_xlabel("Prediction horizon (s)")
    axes[0].set_ylabel("Pixel MSE")
    axes[0].set_title("Lava lamp — MSE vs horizon")
    axes[0].grid(True)
    axes[1].semilogy(times, mse + 1e-8)
    axes[1].set_xlabel("Prediction horizon (s)")
    axes[1].set_ylabel("Pixel MSE (log)")
    axes[1].set_title("Lava lamp — exponential divergence?")
    axes[1].grid(True)
    plt.tight_layout()
    out = os.path.join(args.runs_dir, "lavalamp_horizon_decay.png")
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"Saved: {out}")


if __name__ == "__main__":
    main()
