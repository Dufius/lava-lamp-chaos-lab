# Usage Examples

This document provides detailed examples for using the Lava Lamp Chaos Lab tools.

## Table of Contents

1. [Data Collection](#data-collection)
2. [Training Models](#training-models)
3. [Making Predictions](#making-predictions)
4. [Evaluation and Visualization](#evaluation-and-visualization)
5. [Advanced Usage](#advanced-usage)

---

## Data Collection

### Generate Synthetic Test Data

For quick testing without a physical lava lamp:

```python
from scripts.generate_test_data import generate_moving_blobs, save_frames_as_images

# Generate 500 frames
frames = generate_moving_blobs(num_frames=500, frame_size=(128, 128), num_blobs=5)

# Save to disk
save_frames_as_images(frames, "data/samples")
```

### Capture from Webcam

```python
from src.data_collection import VideoCapture

# Initialize capturer
capturer = VideoCapture(
    source=0,  # 0 for default webcam, or 1, 2, etc.
    output_dir="data/raw",
    frame_size=(640, 480),
    fps=30
)

# Capture 5 minutes of footage
video_path = capturer.start_capture(duration_seconds=300)
capturer.release()

print(f"Saved to: {video_path}")
```

### Extract Frames from Video

```python
from src.data_collection import extract_frames_from_video

# Extract every frame
num_frames = extract_frames_from_video(
    video_path="data/raw/lavalamp_20250101_120000.mp4",
    output_dir="data/frames",
    frame_interval=1  # Extract every frame (1), or every 5th frame (5), etc.
)

print(f"Extracted {num_frames} frames")
```

---

## Training Models

### Basic Training

```python
from src.train import train

train(
    data_path="data/samples",
    epochs=10,
    batch_size=4,
    learning_rate=1e-3,
    model_name="simple_cnn"
)
```

### Advanced Training with Custom Configuration

```python
from src.train import Trainer
from src.dataset import create_dataloaders
from src.models import UNet
import torch.nn as nn
import torch.optim as optim

# Create dataloaders
train_loader, val_loader = create_dataloaders(
    data_dir="data/samples",
    batch_size=8,
    train_split=0.8,
    sequence_length=1,
    prediction_horizon=1,
    frame_size=(128, 128),
)

# Create model
model = UNet(base_channels=64)

# Create trainer
trainer = Trainer(
    model=model,
    device="cuda",  # or "cpu"
    checkpoint_dir="checkpoints/unet_experiment_1"
)

# Custom loss and optimizer
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)

# Train
history = trainer.train(
    train_loader=train_loader,
    val_loader=val_loader,
    epochs=50,
    criterion=criterion,
    optimizer=optimizer,
    save_every=10
)
```

### Training All Three Models for Comparison

```bash
# SimpleCNN
python -m src.train --data-path data/samples --epochs 30 --model simple_cnn --batch-size 8

# UNet
python -m src.train --data-path data/samples --epochs 30 --model unet --batch-size 8

# ConvLSTM
python -m src.train --data-path data/samples --epochs 30 --model conv_lstm --batch-size 4
```

---

## Making Predictions

### Single Frame Prediction

```python
from src.predict import Predictor
import cv2

# Load predictor
predictor = Predictor(
    model_path="checkpoints/best_model.pt",
    model_type="unet",
    device="cuda"
)

# Load frame
frame = cv2.imread("data/samples/frame_000050.png")

# Predict next frame
prediction = predictor.predict_next_frame(frame)

# Save prediction
cv2.imwrite("prediction.png", prediction)
```

### Autoregressive Sequence Prediction

This is crucial for testing prediction horizon limits:

```python
from src.predict import Predictor
import cv2

predictor = Predictor(
    model_path="checkpoints/best_model.pt",
    model_type="unet"
)

# Load initial frame
initial_frame = cv2.imread("data/samples/frame_000000.png")

# Predict 100 steps into the future
predictions = predictor.predict_sequence(
    initial_frame=initial_frame,
    num_steps=100
)

# Save as video
import cv2
height, width = predictions[0].shape[:2]
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter('prediction_sequence.mp4', fourcc, 10, (width, height))

for pred in predictions:
    out.write(pred)

out.release()
```

### Video Processing

```python
from src.predict import Predictor

predictor = Predictor(
    model_path="checkpoints/best_model.pt",
    model_type="simple_cnn"
)

# Process video (creates side-by-side comparison)
predictor.predict_from_video(
    video_path="data/raw/test_video.mp4",
    output_path="predictions/comparison.mp4",
    max_frames=300  # Optional: limit frames
)
```

---

## Evaluation and Visualization

### Compute Metrics

```python
from src.evaluation import compute_metrics
import torch

# Your predictions and targets
predictions = torch.rand(10, 3, 128, 128)  # Example
targets = torch.rand(10, 3, 128, 128)

# Compute metrics
metrics = compute_metrics(predictions, targets)

print(metrics)
# Output: MSE: 0.123456 | MAE: 0.234567 | PSNR: 25.67 dB | SSIM: 0.8234 | RMSE: 0.351234
```

### Visualize Predictions

```python
from src.evaluation import visualize_predictions
import torch

# Load or create sample data
inputs = torch.rand(4, 3, 128, 128)
predictions = torch.rand(4, 3, 128, 128)
targets = torch.rand(4, 3, 128, 128)

# Visualize
visualize_predictions(
    inputs=inputs,
    predictions=predictions,
    targets=targets,
    save_path="results/predictions_visual.png",
    num_samples=4
)
```

### Plot Training History

```python
from src.evaluation import plot_training_history

plot_training_history(
    history_path="checkpoints/training_history.json",
    save_path="results/training_curve.png"
)
```

### Test Prediction Horizon Decay

This is the core chaos experiment:

```python
from src.evaluation.metrics import compute_prediction_horizon_metrics
from src.evaluation.visualize import plot_prediction_horizon_decay
from src.models import UNet
import torch

# Load model
model = UNet()
checkpoint = torch.load("checkpoints/best_model.pt")
model.load_state_dict(checkpoint["model_state_dict"])
model.eval()

# Load test data
initial_frame = torch.rand(1, 3, 128, 128)  # Your actual data
target_frames = torch.rand(50, 3, 128, 128)  # 50 future frames

# Compute metrics at each horizon
horizon_metrics = compute_prediction_horizon_metrics(
    model=model,
    initial_frame=initial_frame,
    target_frames=target_frames,
    device="cuda"
)

# Convert to serializable format
horizon_dict = {h: m.to_dict() for h, m in horizon_metrics.items()}

# Plot decay
plot_prediction_horizon_decay(
    horizon_metrics=horizon_dict,
    save_path="results/horizon_decay.png"
)

# This plot shows how quickly predictions degrade - key evidence for/against chaos!
```

---

## Advanced Usage

### Custom Dataset

```python
from src.dataset import LavaLampDataset
from torch.utils.data import DataLoader
import torchvision.transforms as transforms

# Custom transforms
custom_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((256, 256)),  # Different size
    transforms.RandomHorizontalFlip(p=0.5),  # Data augmentation
    transforms.ColorJitter(brightness=0.1, contrast=0.1),
    transforms.ToTensor(),
])

# Create dataset
dataset = LavaLampDataset(
    data_dir="data/samples",
    sequence_length=5,  # Use 5 frames as input
    prediction_horizon=3,  # Predict 3 frames ahead
    frame_size=(256, 256),
    transform=custom_transform
)

# Create dataloader
dataloader = DataLoader(dataset, batch_size=4, shuffle=True, num_workers=4)
```

### Video Dataset (Memory Efficient)

```python
from src.dataset import VideoDataset

# Load frames directly from video (doesn't extract all to disk)
dataset = VideoDataset(
    video_path="data/raw/long_video.mp4",
    sequence_length=1,
    prediction_horizon=1,
    frame_size=(128, 128),
    frame_skip=2  # Skip every other frame
)
```

### Resume Training from Checkpoint

```python
from src.train import Trainer
from src.models import SimpleCNN

# Create model
model = SimpleCNN()

# Create trainer
trainer = Trainer(model, checkpoint_dir="checkpoints")

# Load checkpoint
trainer.load_checkpoint("checkpoint_epoch_20.pt")

# Continue training
# (create dataloaders first)
trainer.train(train_loader, val_loader, epochs=30)  # Will continue from epoch 20
```

### Batch Prediction Evaluation

```python
from src.predict import Predictor
from src.evaluation import compute_metrics
from src.dataset import LavaLampDataset
import torch

# Load model
predictor = Predictor(
    model_path="checkpoints/best_model.pt",
    model_type="unet"
)

# Load test dataset
test_dataset = LavaLampDataset(
    data_dir="data/test",
    frame_size=(128, 128)
)

# Evaluate on entire test set
all_metrics = []

for i in range(len(test_dataset)):
    input_frame, target_frame = test_dataset[i]

    # Predict
    input_np = input_frame.numpy().transpose(1, 2, 0)
    prediction_np = predictor.predict_next_frame(input_np)

    # Convert back to tensor
    prediction = torch.from_numpy(prediction_np).permute(2, 0, 1).float() / 255.0

    # Compute metrics
    metrics = compute_metrics(
        prediction.unsqueeze(0),
        target_frame.unsqueeze(0)
    )

    all_metrics.append(metrics)

# Average metrics
avg_mse = sum(m.mse for m in all_metrics) / len(all_metrics)
avg_ssim = sum(m.ssim for m in all_metrics) / len(all_metrics)

print(f"Test Set - MSE: {avg_mse:.6f}, SSIM: {avg_ssim:.4f}")
```

### Comparing Models

```python
models_to_compare = [
    ("simple_cnn", "checkpoints/simple_cnn/best_model.pt"),
    ("unet", "checkpoints/unet/best_model.pt"),
    ("conv_lstm", "checkpoints/conv_lstm/best_model.pt"),
]

results = {}

for model_name, checkpoint_path in models_to_compare:
    predictor = Predictor(model_path=checkpoint_path, model_type=model_name)

    # Evaluate on test set (use code from above)
    # ...

    results[model_name] = {
        "mse": avg_mse,
        "ssim": avg_ssim,
        "psnr": avg_psnr
    }

# Print comparison
import json
print(json.dumps(results, indent=2))
```

---

## Tips and Best Practices

### For Training

1. **Start small**: Use synthetic data to verify pipeline works
2. **Monitor overfitting**: Watch train vs val loss divergence
3. **Experiment with horizons**: Try predicting 1, 5, 10 frames ahead
4. **Save checkpoints**: Training can take hours, save often
5. **Use GPU**: Training is 10-100x faster on GPU

### For Data Collection

1. **Consistency**: Same lighting, same lamp position
2. **Duration**: More data = better models (aim for hours of footage)
3. **Quality**: Higher resolution = better results (but slower training)
4. **Variety**: Different lamps, temperatures = better generalization

### For Evaluation

1. **Multiple metrics**: MSE alone isn't enough, use SSIM and PSNR
2. **Visual inspection**: Numbers don't tell the whole story
3. **Horizon testing**: The key experiment for chaos research
4. **Baseline comparison**: Compare to simple methods (e.g., just copy last frame)

---

## Troubleshooting

### Model produces blurry predictions
- Try UNet instead of SimpleCNN (skip connections help)
- Add perceptual loss instead of just MSE
- Train longer or with more data

### Predictions diverge quickly
- This is expected for chaotic systems!
- Try ConvLSTM to capture temporal dynamics
- Collect more training data
- This might confirm Hypothesis A (chaos dominates)

### Out of memory errors
- Reduce batch size
- Reduce frame size
- Use gradient accumulation
- Use CPU (slower but works)

---

For more information, see:
- [QUICKSTART.md](../QUICKSTART.md) - Getting started
- [README.md](../README.md) - Project overview
- [experiment_overview.md](experiment_overview.md) - Research methodology
- [theory_background.md](theory_background.md) - Chaos theory background
