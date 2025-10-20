# 🧠 Lava Lamp Chaos Lab

[![CI](https://github.com/Dufius/lava-lamp-chaos-lab/workflows/Continuous%20Integration/badge.svg)](https://github.com/Dufius/lava-lamp-chaos-lab/actions)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

> **Can AI learn to predict deterministic chaos?**

This open research project explores the limits of pattern learning using chaotic physical systems—starting with the mesmerizing motion of lava lamps. We're testing where AI prediction meets the fundamental unpredictability of chaos theory.

## 🎯 The Question

Modern AI excels at pattern recognition. But can it predict chaos?

- **Deterministic**: Governed by known physics (Navier-Stokes equations, thermodynamics)
- **Unpredictable**: Sensitive to initial conditions, practically impossible to forecast long-term

We're using **lava lamps** as our test case because they're:
- Genuinely chaotic (Cloudflare uses them for cryptographic randomness!)
- Cheap and accessible (~$20 + webcam)
- Provide continuous, unlabeled training data
- Beautiful to watch 🌊

## 🚀 Quick Start

**See [QUICKSTART.md](QUICKSTART.md) for detailed setup instructions.**

```bash
# Clone the repository
git clone https://github.com/Dufius/lava-lamp-chaos-lab.git
cd lava-lamp-chaos-lab

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Generate test data
python scripts/generate_test_data.py

# Train a model
python -m src.train --data-path data/samples --epochs 20 --model unet

# Make predictions
python -m src.predict --model-path checkpoints/best_model.pt --model-type unet --input data/samples/frame_000050.png
```

**New to the project?** Start here:
- [QUICKSTART.md](QUICKSTART.md) - Step-by-step setup guide
- [docs/usage_examples.md](docs/usage_examples.md) - Comprehensive examples
- [IMPLEMENTATION_SUMMARY.md](IMPLEMENTATION_SUMMARY.md) - Full feature list

## 📊 Research Phases

### Phase 1: Single Lamp Baseline ⏳ *In Progress*
- Train models to predict next frame from current frame
- Test prediction horizons: 1 second → 1 hour
- Establish baseline metrics

### Phase 2: Complexity Scaling 📋 *Planned*
- Scale training data: 1 day → 1 year of footage
- Multiple lamps for generalization testing
- Parameter variations (temperature, lamp size)

### Phase 3: Model Comparison 📋 *Planned*
- Pure pattern learning vs physics-informed networks
- Compare to traditional CFD simulations
- Hybrid approaches

### Phase 4: Generalization 📋 *Planned*
- Double pendulums, smoke plumes, other chaotic systems
- Test cross-domain transfer learning

## 🔬 Hypotheses

We're testing three competing hypotheses:

**Hypothesis A (Chaos Dominates)**: Prediction accuracy decays exponentially beyond short horizons, regardless of model sophistication.

**Hypothesis B (Patterns Persist)**: Sufficient training reveals stable attractors, enabling medium-term prediction.

**Hypothesis C (Emergent Physics)**: Models implicitly learn fluid dynamics, enabling physics-aware prediction.

## 📁 Project Structure

```
lava-lamp-chaos-lab/
├── src/                    # Source code
│   ├── train.py           # Training pipeline
│   ├── models/            # Model architectures (coming soon)
│   └── evaluation/        # Metrics and evaluation
├── experiments/           # Experiment configs and results
├── data/                  # Training data and samples
├── notebooks/             # Analysis notebooks
├── docs/                  # Documentation
│   ├── experiment_overview.md
│   ├── theory_background.md
│   └── contribution_guidelines.md
├── tests/                 # Unit tests (coming soon)
└── .github/workflows/     # CI/CD pipeline
```

## 🤝 Contributing

We welcome contributions! This is **open science**—all experiments, data, and results should be reproducible and shareable.

**Ways to contribute:**
- 🧪 Run experiments with your own lava lamp
- 💻 Implement models and evaluation metrics
- 📊 Share datasets and results
- 📝 Improve documentation
- 🐛 Report bugs or suggest features

See [CONTRIBUTING.md](CONTRIBUTING.md) for detailed guidelines.

## 📚 Documentation

**Getting Started:**
- **[QUICKSTART.md](QUICKSTART.md)**: Fast setup and first steps
- **[Usage Examples](docs/usage_examples.md)**: Comprehensive code examples
- **[Implementation Summary](IMPLEMENTATION_SUMMARY.md)**: Complete feature overview

**Research:**
- **[Experiment Overview](docs/experiment_overview.md)**: Detailed research methodology
- **[Theory Background](docs/theory_background.md)**: Chaos theory and AI fundamentals

**Contributing:**
- **[Setup Guide](OSS_SETUP_GUIDE.md)**: Complete open-source project setup
- **[Contributing Guidelines](CONTRIBUTING.md)**: How to get involved

## 🎓 Research Background

This project bridges:
- **AI Research**: Limits of pattern learning on continuous physical systems
- **Physics**: Computational approaches to chaotic systems
- **Philosophy**: Nature of determinism and predictability

Related work includes:
- AlphaFold (protein folding—complex but not chaotic)
- Weather prediction AI (short-term forecasting of chaos)
- Physics-informed neural networks
- Classical chaos theory (Lorenz attractors)

## 🛠️ Tech Stack

- **Python 3.11+**
- **PyTorch** for deep learning
- **OpenCV** for video processing
- **NumPy** & **SciPy** for numerical computing
- **Matplotlib** & **scikit-image** for visualization
- **Jupyter** for interactive analysis

## 📈 Current Status

- ✅ Repository structure established
- ✅ CI/CD pipeline configured
- ✅ Comprehensive documentation written
- ✅ **Data collection pipeline implemented**
- ✅ **Three model architectures implemented** (SimpleCNN, U-Net, ConvLSTM)
- ✅ **Complete training pipeline with checkpointing**
- ✅ **Evaluation metrics and visualization tools**
- ✅ **Inference and prediction pipeline**
- ✅ **Synthetic test data generator**
- 📋 First dataset collection with real lava lamp (ready to use)
- 📋 Phase 1 experiments (ready to begin)

## 🌟 Why This Matters

**For AI**: Understanding the limits of pattern learning helps us build better, more reliable systems.

**For Physics**: Machine learning might discover new approaches to intractable problems.

**For Philosophy**: This probes fundamental questions about determinism, predictability, and whether "enough information" can overcome chaos.

**For Fun**: It's a excuse to stare at lava lamps and call it "research"! 🌈

## 📮 Get Involved

- **GitHub Discussions**: Ask questions, share ideas
- **Issues**: Report bugs, request features, propose experiments
- **Pull Requests**: Contribute code, docs, or data
- **Twitter**: Follow [@YourHandle] for updates (optional)

## 📄 License

MIT License - Open science encouraged! See [LICENSE](LICENSE) for details.

## 🙏 Acknowledgments

- Cloudflare for pioneering the use of lava lamps in cryptography
- The chaos theory community for decades of foundational work
- All contributors who help push the boundaries of AI understanding

## 📖 Citation

If you use this project in your research, please cite:

```bibtex
@software{lava_lamp_chaos_lab,
  title = {Lava Lamp Chaos Lab: Testing AI Prediction in Chaotic Systems},
  author = {Your Name},
  year = {2025},
  url = {https://github.com/Dufius/lava-lamp-chaos-lab}
}
```

---

*"In theory, there is no difference between theory and practice. But in practice, there is."*

Let's find out where AI meets chaos! 🌊🧠✨
