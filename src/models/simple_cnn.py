"""Simple CNN encoder-decoder for frame prediction."""

import torch
import torch.nn as nn


class SimpleCNN(nn.Module):
    """
    Simple CNN encoder-decoder architecture for next-frame prediction.

    This is the baseline model for Phase 1 experiments.
    """

    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 3,
        base_channels: int = 64,
    ):
        """
        Initialize SimpleCNN model.

        Args:
            in_channels: Number of input channels (3 for RGB)
            out_channels: Number of output channels (3 for RGB)
            base_channels: Base number of channels for conv layers
        """
        super().__init__()

        # Encoder
        self.encoder = nn.Sequential(
            # Layer 1: 128 -> 64
            nn.Conv2d(in_channels, base_channels, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(base_channels),

            # Layer 2: 64 -> 32
            nn.Conv2d(base_channels, base_channels * 2, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(base_channels * 2),

            # Layer 3: 32 -> 16
            nn.Conv2d(base_channels * 2, base_channels * 4, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(base_channels * 4),

            # Layer 4: 16 -> 8
            nn.Conv2d(base_channels * 4, base_channels * 8, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(base_channels * 8),
        )

        # Decoder
        self.decoder = nn.Sequential(
            # Layer 1: 8 -> 16
            nn.ConvTranspose2d(base_channels * 8, base_channels * 4, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(base_channels * 4),

            # Layer 2: 16 -> 32
            nn.ConvTranspose2d(base_channels * 4, base_channels * 2, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(base_channels * 2),

            # Layer 3: 32 -> 64
            nn.ConvTranspose2d(base_channels * 2, base_channels, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(base_channels),

            # Layer 4: 64 -> 128
            nn.ConvTranspose2d(base_channels, out_channels, kernel_size=4, stride=2, padding=1),
            nn.Sigmoid(),  # Output in [0, 1] range
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input tensor [B, C, H, W]

        Returns:
            Predicted next frame [B, C, H, W]
        """
        # Encode
        encoded = self.encoder(x)

        # Decode
        output = self.decoder(encoded)

        return output

    def count_parameters(self) -> int:
        """Count total trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


if __name__ == "__main__":
    # Test the model
    model = SimpleCNN()
    print(f"SimpleCNN created")
    print(f"Parameters: {model.count_parameters():,}")

    # Test forward pass
    dummy_input = torch.randn(2, 3, 128, 128)
    output = model(dummy_input)
    print(f"Input shape: {dummy_input.shape}")
    print(f"Output shape: {output.shape}")
    assert output.shape == dummy_input.shape, "Output shape mismatch!"
    print("✅ Model test passed!")
