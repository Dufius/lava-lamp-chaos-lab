# Contributing to Lava Lamp Chaos Lab

Thank you for your interest in contributing to this open science project! We welcome contributions from researchers, developers, and anyone curious about the limits of AI pattern learning in chaotic systems.

## 🎯 Project Goal

We're investigating whether AI can learn to predict deterministic but chaotic physical systems, starting with lava lamp fluid dynamics. This is open, collaborative science—all experiments, data, and results should be reproducible and shareable.

## 🤝 How to Contribute

### Types of Contributions

1. **Code Contributions**
   - Model implementations
   - Data processing pipelines
   - Evaluation metrics
   - Visualization tools
   - Bug fixes

2. **Experiments**
   - Run experiments with your own lava lamp
   - Test new model architectures
   - Explore different prediction horizons
   - Try other chaotic systems

3. **Data**
   - Record and share lava lamp footage
   - Document recording conditions
   - Create synthetic test datasets

4. **Documentation**
   - Improve setup instructions
   - Write tutorials
   - Add examples
   - Clarify research methods

5. **Analysis**
   - Create visualization notebooks
   - Statistical analysis
   - Comparative studies
   - Theoretical insights

## 🔬 Scientific Rigor Requirements

All contributions must maintain scientific integrity:

- **Reproducibility**: Provide complete environment details, random seeds, and configurations
- **Documentation**: Explain methods, assumptions, and limitations
- **Data Quality**: Document recording conditions, preprocessing steps
- **Transparency**: Share negative results too—they're valuable!
- **Citations**: Credit prior work and related research

## 💻 Code Contribution Process

### 1. Setup Development Environment

```bash
# Clone the repository
git clone https://github.com/Dufius/lava-lamp-chaos-lab.git
cd lava-lamp-chaos-lab

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\\Scripts\\activate

# Install dependencies
pip install -r requirements.txt
pip install -r requirements-dev.txt  # Development tools

# Install pre-commit hooks
pre-commit install
```

### 2. Create a Branch

```bash
git checkout -b feature/your-feature-name
# or
git checkout -b experiment/experiment-name
# or
git checkout -b fix/bug-description
```

### 3. Make Your Changes

Follow our code standards:

- **Python Style**: We use Black for formatting (enforced by CI)
- **Linting**: We use flake8 (run `flake8 src`)
- **Type Hints**: Add type annotations where practical
- **Documentation**: Add docstrings to all public functions
- **Tests**: Add tests for new functionality

### 4. Test Your Changes

```bash
# Format code
black .

# Check linting
flake8 src --ignore=E501

# Run tests
pytest tests/

# Verify imports
python -m compileall src
```

### 5. Commit Your Changes

```bash
# Stage changes
git add .

# Commit with clear message
git commit -m "feat: Add ConvLSTM model for frame prediction"
# or
git commit -m "fix: Correct data loader batch handling"
# or
git commit -m "docs: Add setup guide for Raspberry Pi"
```

**Commit Message Convention:**
- `feat:` New feature
- `fix:` Bug fix
- `docs:` Documentation changes
- `test:` Adding tests
- `refactor:` Code refactoring
- `exp:` New experiment
- `data:` Data-related changes

### 6. Push and Create Pull Request

```bash
git push origin feature/your-feature-name
```

Then create a Pull Request on GitHub with:
- Clear title describing the change
- Description of what changed and why
- Results/screenshots if applicable
- Link to related issues

## 🧪 Experiment Contributions

### Submitting Experiment Results

1. **Create Experiment Config**
   ```yaml
   # experiments/configs/your_experiment.yaml
   name: "Your Experiment Name"
   description: "Brief description"
   date: "2025-10-20"
   
   hardware:
     lamp_model: "Standard 52oz lava lamp"
     camera: "Logitech C920"
     fps: 30
     resolution: "1920x1080"
   
   model:
     architecture: "ConvLSTM"
     parameters: {...}
   
   training:
     data_duration: "7 days"
     epochs: 50
     batch_size: 16
   ```

2. **Document Conditions**
   - Room temperature
   - Lighting setup
   - Lamp warmup time
   - Any disturbances

3. **Share Results**
   - Metrics (MSE, SSIM, prediction horizon)
   - Visualizations (predicted vs actual frames)
   - Trained model weights (if willing to share)
   - Analysis notebook

4. **Submit PR**
   - Add config to `experiments/configs/`
   - Add results to `experiments/results/`
   - Add analysis notebook to `notebooks/`
   - Update experiments README

### Data Sharing Guidelines

If sharing lava lamp footage:

1. **Format**
   - MP4 or AVI format
   - Minimum 720p resolution
   - Minimum 30 FPS
   - Constant frame rate

2. **Metadata** (include JSON file)
   ```json
   {
     "recording_date": "2025-10-20",
     "duration_minutes": 60,
     "fps": 30,
     "resolution": [1920, 1080],
     "lamp_model": "Standard 52oz",
     "camera": "Logitech C920",
     "lighting": "Ambient room light",
     "temperature_celsius": 22
   }
   ```

3. **Hosting**
   - Small clips (<100MB): Include in PR
   - Larger datasets: Upload to Zenodo/Hugging Face, link in PR

4. **Privacy**
   - Ensure no personal information visible
   - No identifiable people in frame
   - Check background for sensitive items

## 🐛 Reporting Issues

### Bug Reports

Include:
- **Description**: What went wrong?
- **Steps to Reproduce**: Exact steps to trigger the bug
- **Expected Behavior**: What should happen?
- **Actual Behavior**: What actually happened?
- **Environment**: OS, Python version, dependency versions
- **Logs**: Any error messages or stack traces

### Feature Requests

Include:
- **Problem**: What problem does this solve?
- **Proposed Solution**: How might it work?
- **Alternatives**: Other approaches considered?
- **Use Case**: Concrete example of how you'd use it

### Experiment Ideas

Include:
- **Hypothesis**: What are you testing?
- **Method**: How would you test it?
- **Expected Outcome**: What might you discover?
- **Resources Needed**: Hardware, compute, data?

## 📋 Code Review Process

All PRs will be reviewed for:

1. **Correctness**: Does the code work as intended?
2. **Style**: Does it follow our conventions?
3. **Tests**: Are there adequate tests?
4. **Documentation**: Is it well documented?
5. **Scientific Rigor**: Are methods sound and reproducible?

Expect feedback and iterations—this is normal and helps improve the science!

## 🌟 Recognition

Contributors will be:
- Listed in CONTRIBUTORS.md
- Credited in research papers (for significant contributions)
- Mentioned in release notes
- Featured in project updates

## 🤔 Questions?

- **GitHub Discussions**: Ask questions, share ideas
- **Issues**: Report bugs, request features
- **Email**: [Your contact email for scientific collaboration]

## 📜 License

By contributing, you agree that your contributions will be licensed under the MIT License.

## 🙏 Thank You!

Your contributions help advance our understanding of AI's capabilities in predicting chaotic systems. Every contribution, big or small, moves the research forward!

---

*"In theory, there is no difference between theory and practice. But in practice, there is." - Jan L.A. van de Snepscheut*

Let's find out where AI theory meets chaotic practice! 🌊🧠
