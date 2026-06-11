"""
Phase 0 baseline: predict double-pendulum Cartesian joint positions with
GRU / LSTM / VRNN — no pixel encoder, pure state-space dynamics.

Usage
-----
python -m experiments.pendulum_baseline --model gru --epochs 50
python -m experiments.pendulum_baseline --model vrnn --epochs 50 --beta 0.5
python -m experiments.pendulum_baseline --eval-only --checkpoint runs/gru_best.pt --model gru

The key output is the prediction-horizon decay plot, which shows how quickly
MSE grows beyond the Lyapunov time horizon (~2-4 s for a double pendulum).
"""
import argparse
import json
import os

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm

from src.envs.double_pendulum import PendulumParams, generate_trajectories, to_cartesian
from src.models.rnn_predictor import GRUPredictor, LSTMPredictor, VRNNPredictor
from src.pendulum_dataset import make_dataloaders

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

MODELS = {
    "gru": GRUPredictor,
    "lstm": LSTMPredictor,
    "vrnn": VRNNPredictor,
}


def build_model(name, state_dim=4, hidden_dim=128):
    if name == "vrnn":
        return VRNNPredictor(state_dim=state_dim, hidden_dim=hidden_dim, latent_dim=32)
    return MODELS[name](state_dim=state_dim, hidden_dim=hidden_dim, num_layers=2, dropout=0.1)


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------


def train_epoch(model, loader, optimizer, device, is_vrnn):
    model.train()
    total_loss = 0.0
    for context, target in loader:
        context = context.to(device)  # [B, T, 4]
        target = target.to(device)  # [B, 1, 4]

        optimizer.zero_grad()

        if is_vrnn:
            # Feed full context; model predicts 1..T, we care about last step
            preds, kl = model(context)  # [B, T, 4]
            mse = nn.functional.mse_loss(preds[:, -1], target[:, 0])
            loss = mse + model.beta * kl
        else:
            pred, _ = model(context)  # [B, 4]
            loss = nn.functional.mse_loss(pred, target[:, 0])

        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total_loss += loss.item()

    return total_loss / len(loader)


@torch.no_grad()
def validate(model, loader, device, is_vrnn):
    model.eval()
    total_mse = 0.0
    for context, target in loader:
        context = context.to(device)
        target = target.to(device)
        if is_vrnn:
            preds, _ = model(context)
            mse = nn.functional.mse_loss(preds[:, -1], target[:, 0])
        else:
            pred, _ = model(context)
            mse = nn.functional.mse_loss(pred, target[:, 0])
        total_mse += mse.item()
    return total_mse / len(loader)


# ---------------------------------------------------------------------------
# Horizon decay evaluation
# ---------------------------------------------------------------------------


@torch.no_grad()
def eval_horizon_decay(model, test_trajs, seq_len, max_horizon, device):
    """
    For each test trajectory, burn in `seq_len` steps, then roll out
    `max_horizon` steps and compare to ground truth.

    Returns
    -------
    horizons : np.ndarray  [max_horizon]
    mse      : np.ndarray  [max_horizon]
    """
    model.eval()
    mse_by_h = np.zeros(max_horizon)
    count = 0

    for traj in test_trajs:
        cart = to_cartesian(traj).astype(np.float32)
        T = len(cart)
        if T < seq_len + max_horizon:
            continue

        context = torch.from_numpy(cart[:seq_len]).unsqueeze(0).to(device)  # [1, T, 4]
        gt = cart[seq_len : seq_len + max_horizon]  # [H, 4]

        preds = model.rollout(context, max_horizon).squeeze(0).cpu().numpy()  # [H, 4]
        mse_by_h += ((preds - gt) ** 2).mean(axis=1)
        count += 1

    return np.arange(1, max_horizon + 1), mse_by_h / max(count, 1)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", choices=["gru", "lstm", "vrnn"], default="gru")
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--hidden-dim", type=int, default=128)
    p.add_argument("--seq-len", type=int, default=30)
    p.add_argument("--n-train", type=int, default=500, help="training trajectories")
    p.add_argument("--n-test", type=int, default=50, help="held-out test trajectories")
    p.add_argument("--t-end", type=float, default=20.0, help="simulation seconds per traj")
    p.add_argument("--dt", type=float, default=0.02, help="timestep (s)")
    p.add_argument("--beta", type=float, default=1.0, help="KL weight (VRNN only)")
    p.add_argument("--max-horizon", type=int, default=200, help="steps for decay plot")
    p.add_argument("--runs-dir", default="runs", help="where to save checkpoints / plots")
    p.add_argument("--eval-only", action="store_true")
    p.add_argument("--checkpoint", default=None)
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
    print("Simulating trajectories...")
    params = PendulumParams()
    all_trajs = generate_trajectories(
        n=args.n_train + args.n_test,
        t_end=args.t_end,
        dt=args.dt,
        params=params,
        seed=0,
    )
    train_trajs = all_trajs[: args.n_train]
    test_trajs = all_trajs[args.n_train :]
    print(f"  train: {len(train_trajs)} | test: {len(test_trajs)}")

    train_loader, val_loader = make_dataloaders(
        train_trajs,
        seq_len=args.seq_len,
        horizon=1,
        batch_size=args.batch_size,
    )

    # ------------------------------------------------------------------
    # Model
    # ------------------------------------------------------------------
    model = build_model(args.model, state_dim=4, hidden_dim=args.hidden_dim)
    if args.checkpoint:
        model.load_state_dict(torch.load(args.checkpoint, map_location="cpu"))
        print(f"Loaded checkpoint: {args.checkpoint}")
    model.to(device)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model: {args.model}  |  params: {n_params:,}")

    is_vrnn = args.model == "vrnn"
    if is_vrnn:
        model.beta = args.beta

    # ------------------------------------------------------------------
    # Train
    # ------------------------------------------------------------------
    if not args.eval_only:
        optimizer = Adam(model.parameters(), lr=args.lr)
        scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-5)

        best_val = float("inf")
        history = {"train_loss": [], "val_mse": []}
        ckpt_best = os.path.join(args.runs_dir, f"{args.model}_best.pt")

        for epoch in range(1, args.epochs + 1):
            tr_loss = train_epoch(model, train_loader, optimizer, device, is_vrnn)
            val_mse = validate(model, val_loader, device, is_vrnn)
            scheduler.step()

            history["train_loss"].append(tr_loss)
            history["val_mse"].append(val_mse)

            if val_mse < best_val:
                best_val = val_mse
                torch.save(model.state_dict(), ckpt_best)

            if epoch % 10 == 0 or epoch == 1:
                print(f"[{epoch:3d}/{args.epochs}] train={tr_loss:.5f}  val_mse={val_mse:.5f}")

        with open(os.path.join(args.runs_dir, f"{args.model}_history.json"), "w") as f:
            json.dump(history, f)

        # Load best for evaluation
        model.load_state_dict(torch.load(ckpt_best, map_location="cpu"))
        model.to(device)
        print(f"Best val MSE: {best_val:.5f}")

    # ------------------------------------------------------------------
    # Horizon decay
    # ------------------------------------------------------------------
    print(f"Evaluating horizon decay over {args.max_horizon} steps...")
    horizons, mse = eval_horizon_decay(
        model, test_trajs, args.seq_len, args.max_horizon, device
    )
    dt = args.dt
    times = horizons * dt  # seconds

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))

    axes[0].plot(times, mse)
    axes[0].set_xlabel("Prediction horizon (s)")
    axes[0].set_ylabel("MSE")
    axes[0].set_title(f"{args.model.upper()} — MSE vs horizon")
    axes[0].grid(True)

    axes[1].semilogy(times, mse + 1e-8)
    axes[1].set_xlabel("Prediction horizon (s)")
    axes[1].set_ylabel("MSE (log scale)")
    axes[1].set_title(f"{args.model.upper()} — exponential divergence?")
    axes[1].grid(True)

    plt.tight_layout()
    plot_path = os.path.join(args.runs_dir, f"{args.model}_horizon_decay.png")
    plt.savefig(plot_path, dpi=150)
    print(f"Saved: {plot_path}")

    # Save decay data
    np.save(os.path.join(args.runs_dir, f"{args.model}_horizons.npy"), horizons)
    np.save(os.path.join(args.runs_dir, f"{args.model}_mse.npy"), mse)


if __name__ == "__main__":
    main()
