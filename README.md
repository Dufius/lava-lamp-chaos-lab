# Lava Lamp Chaos Lab

![header](chaos%20lavalamp.png)

[![CI](https://github.com/Dufius/lava-lamp-chaos-lab/workflows/Continuous%20Integration/badge.svg)](https://github.com/Dufius/lava-lamp-chaos-lab/actions)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

> **Can a neural network predict chaotic motion from pixels alone?**

This project measures the **prediction horizon** of chaotic physical systems using recurrent neural networks — first on a simulated double pendulum, then on real lava lamp footage — without any physics labels.

---

## Results

### Phase 0 — Double pendulum (Cartesian coordinates)

GRU and LSTM trained to predict the next state from a 30-step context window.

| Model | Params | Best val MSE | Lyapunov horizon |
|-------|--------|-------------|-----------------|
| GRU   | 42 k   | 0.00008     | ~0.4 s          |
| LSTM  | 56 k   | 0.00007     | ~0.5 s          |
| VRNN  | 69 k   | 0.00006     | unstable rollout |

MSE grows exponentially beyond ~0.4 s — matching the known Lyapunov time for a double pendulum at these parameters.

### Phase 1 — Visual encoder on simulation

LightEncoder (CNN) + pretrained frozen GRU + LightDecoder trained on rendered 32×32 pendulum frames.

- Val loss converges to **0.00460** in 15 epochs
- Horizon decay reproduces the same ~0.2 s Lyapunov signature through pixels
- Confirms the encoder successfully captures bob positions in latent space

### Phase 2 — Real lava lamp footage

End-to-end VideoPredictor (encoder → GRU → decoder) trained from scratch on 1 200 frames of a real lava lamp (5 fps, 64×64 grayscale).

| Horizon | Pixel MSE |
|---------|-----------|
| 0.2 s   | 0.007     |
| 2 s     | 0.035     |
| 6 s     | 0.063     |
| 10 s    | 0.063 (saturated) |

**The lava lamp is ~10× more predictable than the double pendulum** (Lyapunov horizon ~2–3 s vs ~0.4 s). Viscous thermal convection is slow chaos; mechanical chaos is fast. Both systems show the same qualitative signature: rapid early MSE rise followed by saturation as the model loses track of the state.

---

## Architecture

```
frames [B, T, 1, H, W]
    │
    ▼
LightEncoder (3 conv layers)
    │  [B, T, latent_dim]
    ▼
GRU (2 layers, hidden_dim=128)
    │  [B, latent_dim]
    ▼
LightDecoder (3 deconv layers)
    │
    ▼
predicted frame [B, 1, H, W]
```

Loss = `α · recon_loss + (1−α) · pred_loss`

Phase 0 uses raw Cartesian coordinates (no images). Phase 1 reuses Phase 0 GRU weights (frozen). Phase 2 trains everything from scratch on pixels.

---

## Project structure

```
lava-lamp-chaos-lab/
├── src/
│   ├── envs/
│   │   ├── double_pendulum.py   # RK45 simulator + Cartesian converter
│   │   └── render.py            # PIL-based frame renderer (0.05 s / traj)
│   ├── models/
│   │   ├── rnn_predictor.py     # GRUPredictor, LSTMPredictor, VRNNPredictor
│   │   └── light_encoder.py     # LightEncoder, LightDecoder, PendulumVideoPredictor
│   └── data/
│       └── extract_frames.py    # ffmpeg frame extraction pipeline
├── experiments/
│   ├── pendulum_baseline.py     # Phase 0 training + horizon decay
│   ├── pendulum_visual.py       # Phase 1 training
│   └── lavalamp_visual.py       # Phase 2 training
├── data/
│   ├── raw/                     # Source video (gitignored)
│   └── frames/                  # Extracted .npy frames (gitignored)
└── runs/                        # Checkpoints + plots (gitignored)
```

---

## Quick start

```bash
git clone https://github.com/Dufius/lava-lamp-chaos-lab.git
cd lava-lamp-chaos-lab
pip install -r requirements.txt
```

**Phase 0 — train RNN on double pendulum:**
```bash
python -m experiments.pendulum_baseline --model gru --epochs 30
python -m experiments.pendulum_baseline --model lstm --epochs 30
```

**Phase 1 — train visual encoder on simulation:**
```bash
python -m experiments.pendulum_visual \
    --rnn-checkpoint runs/gru_best.pt \
    --epochs 15 --n-train 10 --img-size 32 --freeze-rnn
```

**Phase 2 — extract lava lamp frames and train:**
```bash
# Extract frames from your own video
python -m src.data.extract_frames --video path/to/lavalamp.mp4

# Train end-to-end visual predictor
python -m experiments.lavalamp_visual \
    --frames data/frames/lavalamp_frames.npy \
    --epochs 30
```

---

## Hypotheses being tested

**A — Chaos dominates**: MSE grows exponentially beyond a system-specific Lyapunov horizon, regardless of model size. *Supported by Phase 0 results.*

**B — Patterns persist**: Sufficient data reveals stable attractors enabling medium-term prediction. *Partially supported — the lava lamp saturates rather than diverging to infinity.*

**C — Emergent physics**: Models implicitly learn fluid/mechanical dynamics purely from pixels. *Open question — Phase 2 shows the encoder captures meaningful structure, but interpretability is untested.*

---

## Tech stack

- **PyTorch** — models and training
- **PIL / ffmpeg** — fast frame rendering and video extraction
- **SciPy** — RK45 ODE integration for the pendulum
- **Matplotlib** — result plots

## License

MIT
