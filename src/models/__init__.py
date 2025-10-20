"""Model architectures for lava lamp prediction."""

from .simple_cnn import SimpleCNN
from .unet import UNet
from .conv_lstm import ConvLSTM

__all__ = ["SimpleCNN", "UNet", "ConvLSTM"]
