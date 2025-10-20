"""Test script to verify all components work correctly."""

import torch
import sys


def test_models():
    """Test all model architectures."""
    print("Testing model architectures...")

    from src.models import SimpleCNN, UNet, ConvLSTM

    models = [
        ("SimpleCNN", SimpleCNN()),
        ("UNet", UNet()),
        ("ConvLSTM", ConvLSTM()),
    ]

    dummy_input = torch.randn(2, 3, 128, 128)

    for name, model in models:
        try:
            output = model(dummy_input)
            assert output.shape == dummy_input.shape, f"{name}: Shape mismatch!"
            param_count = model.count_parameters()
            print(f"  ✓ {name}: {param_count:,} parameters")
        except Exception as e:
            print(f"  ✗ {name} failed: {e}")
            return False

    return True


def test_dataset():
    """Test dataset loading (without actual data)."""
    print("\nTesting dataset modules...")

    try:
        from src.dataset import LavaLampDataset, VideoDataset
        print("  ✓ Dataset modules imported successfully")
        return True
    except Exception as e:
        print(f"  ✗ Dataset import failed: {e}")
        return False


def test_evaluation():
    """Test evaluation metrics."""
    print("\nTesting evaluation metrics...")

    from src.evaluation import compute_metrics

    try:
        pred = torch.rand(4, 3, 128, 128)
        target = torch.rand(4, 3, 128, 128)

        metrics = compute_metrics(pred, target)

        assert hasattr(metrics, 'mse'), "Missing MSE metric"
        assert hasattr(metrics, 'ssim'), "Missing SSIM metric"
        assert hasattr(metrics, 'psnr'), "Missing PSNR metric"

        print(f"  ✓ Metrics computed: MSE={metrics.mse:.4f}, SSIM={metrics.ssim:.4f}")
        return True
    except Exception as e:
        print(f"  ✗ Metrics failed: {e}")
        return False


def test_predictor():
    """Test predictor module."""
    print("\nTesting predictor module...")

    try:
        from src.predict import Predictor
        print("  ✓ Predictor module imported successfully")
        return True
    except Exception as e:
        print(f"  ✗ Predictor import failed: {e}")
        return False


def test_trainer():
    """Test trainer module."""
    print("\nTesting trainer module...")

    try:
        from src.train import Trainer
        from src.models import SimpleCNN

        model = SimpleCNN()
        trainer = Trainer(model, device="cpu", checkpoint_dir="test_checkpoints")

        print("  ✓ Trainer created successfully")
        return True
    except Exception as e:
        print(f"  ✗ Trainer failed: {e}")
        return False


def test_data_collection():
    """Test data collection modules."""
    print("\nTesting data collection modules...")

    try:
        from src.data_collection import VideoCapture, extract_frames_from_video
        print("  ✓ Data collection modules imported successfully")
        return True
    except Exception as e:
        print(f"  ✗ Data collection import failed: {e}")
        return False


def main():
    """Run all tests."""
    print("=" * 60)
    print("Lava Lamp Chaos Lab - Implementation Test")
    print("=" * 60)

    tests = [
        test_models,
        test_dataset,
        test_evaluation,
        test_predictor,
        test_trainer,
        test_data_collection,
    ]

    results = []

    for test in tests:
        try:
            result = test()
            results.append(result)
        except Exception as e:
            print(f"  ✗ Test crashed: {e}")
            results.append(False)

    print("\n" + "=" * 60)
    print(f"Results: {sum(results)}/{len(results)} tests passed")
    print("=" * 60)

    if all(results):
        print("\n✅ All tests passed! Implementation is ready.")
        print("\nNext steps:")
        print("  1. Install dependencies: pip install -r requirements.txt")
        print("  2. Generate test data: python scripts/generate_test_data.py")
        print("  3. Run training: python run_experiment.py")
        print("  4. See QUICKSTART.md for more details")
        return 0
    else:
        print("\n⚠️  Some tests failed. Check the errors above.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
