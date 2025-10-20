"""Visualization tools for lava lamp predictions."""

import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import List, Optional, Dict
import json


def visualize_predictions(
    inputs: torch.Tensor,
    predictions: torch.Tensor,
    targets: torch.Tensor,
    save_path: Optional[str] = None,
    num_samples: int = 4,
):
    """
    Visualize prediction results.

    Args:
        inputs: Input frames [B, C, H, W]
        predictions: Predicted frames [B, C, H, W]
        targets: Target frames [B, C, H, W]
        save_path: Optional path to save figure
        num_samples: Number of samples to show
    """
    num_samples = min(num_samples, inputs.shape[0])

    fig, axes = plt.subplots(num_samples, 3, figsize=(12, 4 * num_samples))

    if num_samples == 1:
        axes = axes.reshape(1, -1)

    for i in range(num_samples):
        # Convert tensors to numpy and transpose to [H, W, C]
        input_img = inputs[i].cpu().numpy().transpose(1, 2, 0)
        pred_img = predictions[i].cpu().numpy().transpose(1, 2, 0)
        target_img = targets[i].cpu().numpy().transpose(1, 2, 0)

        # Clip values to [0, 1] range
        input_img = np.clip(input_img, 0, 1)
        pred_img = np.clip(pred_img, 0, 1)
        target_img = np.clip(target_img, 0, 1)

        # Plot input
        axes[i, 0].imshow(input_img)
        axes[i, 0].set_title(f"Input Frame {i+1}")
        axes[i, 0].axis("off")

        # Plot prediction
        axes[i, 1].imshow(pred_img)
        axes[i, 1].set_title(f"Prediction {i+1}")
        axes[i, 1].axis("off")

        # Plot target
        axes[i, 2].imshow(target_img)
        axes[i, 2].set_title(f"Ground Truth {i+1}")
        axes[i, 2].axis("off")

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"📊 Saved visualization to {save_path}")
    else:
        plt.show()

    plt.close()


def visualize_prediction_sequence(
    initial_frame: torch.Tensor,
    predicted_sequence: List[torch.Tensor],
    target_sequence: Optional[List[torch.Tensor]] = None,
    save_path: Optional[str] = None,
):
    """
    Visualize a sequence of predictions (for autoregressive prediction).

    Args:
        initial_frame: Starting frame [C, H, W]
        predicted_sequence: List of predicted frames
        target_sequence: Optional list of ground truth frames
        save_path: Optional path to save figure
    """
    seq_len = len(predicted_sequence)
    has_targets = target_sequence is not None

    if has_targets:
        fig, axes = plt.subplots(2, seq_len + 1, figsize=(3 * (seq_len + 1), 6))
    else:
        fig, axes = plt.subplots(1, seq_len + 1, figsize=(3 * (seq_len + 1), 3))
        axes = axes.reshape(1, -1)

    # Plot initial frame
    initial_img = initial_frame.cpu().numpy().transpose(1, 2, 0)
    initial_img = np.clip(initial_img, 0, 1)

    axes[0, 0].imshow(initial_img)
    axes[0, 0].set_title("Initial Frame")
    axes[0, 0].axis("off")

    if has_targets:
        axes[1, 0].imshow(initial_img)
        axes[1, 0].set_title("Initial Frame")
        axes[1, 0].axis("off")

    # Plot predictions
    for i, pred in enumerate(predicted_sequence):
        pred_img = pred.cpu().numpy().transpose(1, 2, 0)
        pred_img = np.clip(pred_img, 0, 1)

        axes[0, i + 1].imshow(pred_img)
        axes[0, i + 1].set_title(f"Pred t+{i+1}")
        axes[0, i + 1].axis("off")

    # Plot targets if available
    if has_targets:
        for i, target in enumerate(target_sequence):
            target_img = target.cpu().numpy().transpose(1, 2, 0)
            target_img = np.clip(target_img, 0, 1)

            axes[1, i + 1].imshow(target_img)
            axes[1, i + 1].set_title(f"True t+{i+1}")
            axes[1, i + 1].axis("off")

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"📊 Saved sequence visualization to {save_path}")
    else:
        plt.show()

    plt.close()


def plot_training_history(
    history_path: str,
    save_path: Optional[str] = None,
):
    """
    Plot training history from JSON file.

    Args:
        history_path: Path to training_history.json
        save_path: Optional path to save figure
    """
    with open(history_path, "r") as f:
        history = json.load(f)

    train_loss = history.get("train_loss", [])
    val_loss = history.get("val_loss", [])

    epochs = range(1, len(train_loss) + 1)

    plt.figure(figsize=(10, 6))

    plt.plot(epochs, train_loss, "b-", label="Training Loss", linewidth=2)
    plt.plot(epochs, val_loss, "r-", label="Validation Loss", linewidth=2)

    plt.xlabel("Epoch", fontsize=12)
    plt.ylabel("Loss (MSE)", fontsize=12)
    plt.title("Training History", fontsize=14, fontweight="bold")
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)

    # Add minimum validation loss annotation
    if val_loss:
        min_val_loss = min(val_loss)
        min_epoch = val_loss.index(min_val_loss) + 1
        plt.axhline(y=min_val_loss, color="r", linestyle="--", alpha=0.5)
        plt.text(
            len(epochs) * 0.7,
            min_val_loss * 1.1,
            f"Best: {min_val_loss:.4f} (Epoch {min_epoch})",
            fontsize=10,
        )

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"📊 Saved training history to {save_path}")
    else:
        plt.show()

    plt.close()


def plot_prediction_horizon_decay(
    horizon_metrics: Dict[int, dict],
    save_path: Optional[str] = None,
):
    """
    Plot how prediction quality degrades with prediction horizon.

    This is crucial for understanding chaos - we expect exponential decay.

    Args:
        horizon_metrics: Dict mapping horizon to metrics dict
        save_path: Optional path to save figure
    """
    horizons = sorted(horizon_metrics.keys())

    # Extract metrics
    mse_values = [horizon_metrics[h]["mse"] for h in horizons]
    ssim_values = [horizon_metrics[h]["ssim"] for h in horizons]
    psnr_values = [horizon_metrics[h]["psnr"] for h in horizons]

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    # MSE plot (expect exponential growth)
    axes[0].plot(horizons, mse_values, "o-", linewidth=2, markersize=8)
    axes[0].set_xlabel("Prediction Horizon (frames)")
    axes[0].set_ylabel("MSE")
    axes[0].set_title("MSE vs Prediction Horizon")
    axes[0].grid(True, alpha=0.3)
    axes[0].set_yscale("log")  # Log scale to see exponential decay

    # SSIM plot (expect decay to 0)
    axes[1].plot(horizons, ssim_values, "o-", linewidth=2, markersize=8, color="green")
    axes[1].set_xlabel("Prediction Horizon (frames)")
    axes[1].set_ylabel("SSIM")
    axes[1].set_title("SSIM vs Prediction Horizon")
    axes[1].grid(True, alpha=0.3)
    axes[1].set_ylim([0, 1])

    # PSNR plot (expect decay)
    axes[2].plot(horizons, psnr_values, "o-", linewidth=2, markersize=8, color="red")
    axes[2].set_xlabel("Prediction Horizon (frames)")
    axes[2].set_ylabel("PSNR (dB)")
    axes[2].set_title("PSNR vs Prediction Horizon")
    axes[2].grid(True, alpha=0.3)

    plt.suptitle(
        "Prediction Quality Decay (Testing Chaos Hypothesis)",
        fontsize=14,
        fontweight="bold",
    )
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"📊 Saved horizon decay plot to {save_path}")
    else:
        plt.show()

    plt.close()


if __name__ == "__main__":
    # Test visualization
    print("Testing visualization functions...")

    # Create dummy data
    inputs = torch.rand(4, 3, 128, 128)
    predictions = torch.rand(4, 3, 128, 128)
    targets = torch.rand(4, 3, 128, 128)

    # Test prediction visualization
    visualize_predictions(inputs, predictions, targets, num_samples=2)

    print("✅ Visualization test passed!")
