"""Evaluation metrics for lava lamp prediction models."""

import torch
import numpy as np
from dataclasses import dataclass
from typing import Dict
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr


@dataclass
class PredictionMetrics:
    """Container for prediction metrics."""

    mse: float
    mae: float
    psnr: float
    ssim: float
    rmse: float

    def to_dict(self) -> Dict[str, float]:
        """Convert to dictionary."""
        return {
            "mse": self.mse,
            "mae": self.mae,
            "psnr": self.psnr,
            "ssim": self.ssim,
            "rmse": self.rmse,
        }

    def __str__(self) -> str:
        return (
            f"MSE: {self.mse:.6f} | MAE: {self.mae:.6f} | "
            f"PSNR: {self.psnr:.2f} dB | SSIM: {self.ssim:.4f} | "
            f"RMSE: {self.rmse:.6f}"
        )


def compute_metrics(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    data_range: float = 1.0,
) -> PredictionMetrics:
    """
    Compute evaluation metrics for predictions.

    Args:
        predictions: Predicted frames [B, C, H, W]
        targets: Target frames [B, C, H, W]
        data_range: Data range for PSNR/SSIM (typically 1.0 or 255.0)

    Returns:
        PredictionMetrics object
    """
    # Convert to numpy for some metrics
    pred_np = predictions.detach().cpu().numpy()
    target_np = targets.detach().cpu().numpy()

    # MSE (Mean Squared Error)
    mse = torch.mean((predictions - targets) ** 2).item()

    # MAE (Mean Absolute Error)
    mae = torch.mean(torch.abs(predictions - targets)).item()

    # RMSE (Root Mean Squared Error)
    rmse = np.sqrt(mse)

    # PSNR (Peak Signal-to-Noise Ratio) - higher is better
    psnr_values = []
    for i in range(pred_np.shape[0]):
        # Convert from [C, H, W] to [H, W, C] for skimage
        pred_img = np.transpose(pred_np[i], (1, 2, 0))
        target_img = np.transpose(target_np[i], (1, 2, 0))

        psnr_val = psnr(target_img, pred_img, data_range=data_range)
        psnr_values.append(psnr_val)

    avg_psnr = np.mean(psnr_values)

    # SSIM (Structural Similarity Index) - higher is better
    ssim_values = []
    for i in range(pred_np.shape[0]):
        pred_img = np.transpose(pred_np[i], (1, 2, 0))
        target_img = np.transpose(target_np[i], (1, 2, 0))

        # SSIM expects channel_axis parameter
        ssim_val = ssim(
            target_img,
            pred_img,
            data_range=data_range,
            channel_axis=2,  # RGB channels
        )
        ssim_values.append(ssim_val)

    avg_ssim = np.mean(ssim_values)

    return PredictionMetrics(
        mse=mse,
        mae=mae,
        psnr=avg_psnr,
        ssim=avg_ssim,
        rmse=rmse,
    )


def compute_temporal_consistency(
    sequence: torch.Tensor,
) -> float:
    """
    Compute temporal consistency across a sequence.

    Measures how much frames differ from their neighbors.

    Args:
        sequence: Sequence of frames [T, C, H, W]

    Returns:
        Average frame-to-frame difference
    """
    if len(sequence) < 2:
        return 0.0

    differences = []
    for i in range(len(sequence) - 1):
        diff = torch.mean(torch.abs(sequence[i + 1] - sequence[i])).item()
        differences.append(diff)

    return np.mean(differences)


def compute_prediction_horizon_metrics(
    model: torch.nn.Module,
    initial_frame: torch.Tensor,
    target_frames: torch.Tensor,
    device: str = "cpu",
) -> Dict[int, PredictionMetrics]:
    """
    Evaluate model at different prediction horizons.

    This is critical for chaos research - we expect metrics to degrade
    with prediction horizon due to sensitivity to initial conditions.

    Args:
        model: Trained model
        initial_frame: Starting frame [1, C, H, W]
        target_frames: Ground truth future frames [T, C, H, W]
        device: Device to use

    Returns:
        Dictionary mapping horizon to metrics
    """
    model.eval()
    horizon_metrics = {}

    with torch.no_grad():
        current_frame = initial_frame.to(device)

        for horizon in range(1, len(target_frames) + 1):
            # Predict next frame
            prediction = model(current_frame)

            # Compute metrics against ground truth
            target = target_frames[horizon - 1:horizon].to(device)
            metrics = compute_metrics(prediction, target)

            horizon_metrics[horizon] = metrics

            # Use prediction as input for next step (autoregressive)
            current_frame = prediction

    return horizon_metrics


if __name__ == "__main__":
    # Test metrics
    print("Testing evaluation metrics...")

    # Create dummy data
    pred = torch.rand(4, 3, 128, 128)
    target = torch.rand(4, 3, 128, 128)

    # Compute metrics
    metrics = compute_metrics(pred, target)
    print(f"\nMetrics: {metrics}")

    # Test temporal consistency
    sequence = torch.rand(10, 3, 128, 128)
    consistency = compute_temporal_consistency(sequence)
    print(f"\nTemporal consistency: {consistency:.6f}")

    print("\n✅ Metrics test passed!")
