## Quick Start Guide

This guide will help you get up and running with the Lava Lamp Chaos Lab.

### Prerequisites

- Python 3.11 or higher
- A webcam (for data collection) or sample video data
- 4GB+ RAM (8GB+ recommended for training)
- Optional: CUDA-compatible GPU for faster training

### Installation

1. **Clone the repository** (if you haven't already):
   ```bash
   git clone https://github.com/Dufius/lava-lamp-chaos-lab.git
   cd lava-lamp-chaos-lab
   ```

2. **Create a virtual environment**:
   ```bash
   # Windows
   python -m venv venv
   venv\Scripts\activate

   # Linux/Mac
   python3 -m venv venv
   source venv/bin/activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

### Usage

#### Option 1: Use Synthetic Test Data (Quickest Start)

Generate synthetic moving blob data for testing:

```bash
python scripts/generate_test_data.py
```

This creates 200 sample frames in `data/samples/`.

#### Option 2: Collect Real Lava Lamp Data

**Capture from webcam**:
```bash
python -m src.data_collection
# Follow the prompts to capture video
```

**Extract frames from existing video**:
```bash
python -c "from src.data_collection import extract_frames_from_video; extract_frames_from_video('your_video.mp4', 'data/samples')"
```

### Training Models

**Train with default settings** (SimpleCNN model):
```bash
python run_experiment.py
```

**Train with specific configuration**:
```bash
python -m src.train \
  --data-path data/samples \
  --epochs 20 \
  --batch-size 8 \
  --lr 0.001 \
  --model unet
```

**Available models**:
- `simple_cnn`: Basic encoder-decoder (fastest, baseline)
- `unet`: U-Net with skip connections (better quality)
- `conv_lstm`: Temporal model (for sequences, experimental)

### Making Predictions

**Predict next frame from image**:
```bash
python -m src.predict \
  --model-path checkpoints/best_model.pt \
  --model-type simple_cnn \
  --input data/samples/frame_000050.png \
  --output prediction.png
```

**Predict sequence (test prediction horizon)**:
```bash
python -m src.predict \
  --model-path checkpoints/best_model.pt \
  --model-type simple_cnn \
  --input data/samples/frame_000000.png \
  --steps 20 \
  --output prediction_sequence.mp4
```

**Process entire video**:
```bash
python -m src.predict \
  --model-path checkpoints/best_model.pt \
  --model-type simple_cnn \
  --input your_video.mp4 \
  --output output_comparison.mp4
```

### Project Structure

```
lava-lamp-chaos-lab/
├── src/
│   ├── data_collection.py   # Video capture and frame extraction
│   ├── dataset.py            # PyTorch dataset loaders
│   ├── train.py              # Training pipeline
│   ├── predict.py            # Inference pipeline
│   ├── models/               # Neural network architectures
│   │   ├── simple_cnn.py
│   │   ├── unet.py
│   │   └── conv_lstm.py
│   └── evaluation/           # Metrics and visualization
│       ├── metrics.py
│       └── visualize.py
├── data/
│   ├── samples/              # Training frames
│   └── raw/                  # Raw video files
├── checkpoints/              # Saved models
├── scripts/
│   └── generate_test_data.py # Synthetic data generator
└── run_experiment.py         # Quick start script
```

### Expected Workflow

1. **Collect Data**: Capture lava lamp footage or generate test data
2. **Train Model**: Run training with your chosen architecture
3. **Evaluate**: Check metrics and visualizations in `checkpoints/`
4. **Test Horizons**: Use prediction sequence to see how far ahead the model can predict
5. **Iterate**: Experiment with different models, hyperparameters, and training durations

### Troubleshooting

**ImportError: No module named 'torch'**
- Make sure you activated the virtual environment
- Run `pip install -r requirements.txt`

**CUDA out of memory**
- Reduce `--batch-size` to 2 or 1
- Use CPU with `--device cpu`

**Not enough frames error**
- You need at least 10-20 frames for training
- Generate more test data or capture longer video

**Poor prediction quality**
- Train for more epochs (try 50-100)
- Use larger model (try `unet` instead of `simple_cnn`)
- Collect more training data (hours of footage)

### Next Steps

- Read [docs/experiment_overview.md](docs/experiment_overview.md) for research methodology
- Read [docs/theory_background.md](docs/theory_background.md) for chaos theory background
- See [CONTRIBUTING.md](CONTRIBUTING.md) to get involved
- Share your results and experiments!

### Example: Full Pipeline

```bash
# 1. Generate test data
python scripts/generate_test_data.py

# 2. Train a model
python -m src.train --data-path data/samples --epochs 20 --model unet

# 3. Make predictions
python -m src.predict \
  --model-path checkpoints/best_model.pt \
  --model-type unet \
  --input data/samples/frame_000050.png \
  --steps 10 \
  --output sequence.mp4

# 4. View training history
python -c "from src.evaluation import plot_training_history; plot_training_history('checkpoints/training_history.json')"
```

Happy experimenting! 🌊🧠✨
