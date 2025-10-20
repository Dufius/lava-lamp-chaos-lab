"""ConvLSTM model for temporal frame prediction."""

import torch
import torch.nn as nn


class ConvLSTMCell(nn.Module):
    """Convolutional LSTM cell."""

    def __init__(self, input_dim: int, hidden_dim: int, kernel_size: int = 3):
        super().__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.kernel_size = kernel_size
        padding = kernel_size // 2

        # Gates: input, forget, cell, output
        self.conv = nn.Conv2d(
            in_channels=input_dim + hidden_dim,
            out_channels=4 * hidden_dim,
            kernel_size=kernel_size,
            padding=padding,
            bias=True
        )

    def forward(self, input_tensor, cur_state):
        h_cur, c_cur = cur_state

        # Concatenate input and hidden state
        combined = torch.cat([input_tensor, h_cur], dim=1)

        # Compute gates
        combined_conv = self.conv(combined)
        cc_i, cc_f, cc_o, cc_g = torch.split(combined_conv, self.hidden_dim, dim=1)

        i = torch.sigmoid(cc_i)  # Input gate
        f = torch.sigmoid(cc_f)  # Forget gate
        o = torch.sigmoid(cc_o)  # Output gate
        g = torch.tanh(cc_g)     # Cell gate

        # Update cell and hidden state
        c_next = f * c_cur + i * g
        h_next = o * torch.tanh(c_next)

        return h_next, c_next

    def init_hidden(self, batch_size, image_size):
        height, width = image_size
        return (
            torch.zeros(batch_size, self.hidden_dim, height, width, device=self.conv.weight.device),
            torch.zeros(batch_size, self.hidden_dim, height, width, device=self.conv.weight.device)
        )


class ConvLSTM(nn.Module):
    """
    ConvLSTM for spatiotemporal frame prediction.

    This model can handle sequences of frames and capture temporal dynamics,
    which may be important for chaotic systems.
    """

    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 3,
        hidden_dims: list = [64, 64, 64],
        kernel_size: int = 3,
    ):
        """
        Initialize ConvLSTM model.

        Args:
            in_channels: Number of input channels (3 for RGB)
            out_channels: Number of output channels (3 for RGB)
            hidden_dims: List of hidden dimensions for each LSTM layer
            kernel_size: Kernel size for convolutions
        """
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.hidden_dims = hidden_dims
        self.num_layers = len(hidden_dims)

        # Input convolution
        self.input_conv = nn.Conv2d(in_channels, hidden_dims[0], kernel_size=1)

        # Create ConvLSTM cells
        cell_list = []
        for i in range(self.num_layers):
            cur_input_dim = hidden_dims[i - 1] if i > 0 else hidden_dims[0]
            cell_list.append(
                ConvLSTMCell(
                    input_dim=cur_input_dim,
                    hidden_dim=hidden_dims[i],
                    kernel_size=kernel_size
                )
            )
        self.cell_list = nn.ModuleList(cell_list)

        # Output convolution
        self.output_conv = nn.Sequential(
            nn.Conv2d(hidden_dims[-1], hidden_dims[-1], kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dims[-1], out_channels, kernel_size=1),
            nn.Sigmoid()  # Output in [0, 1] range
        )

    def forward(self, x: torch.Tensor, hidden_state=None) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input tensor [B, T, C, H, W] or [B, C, H, W]
            hidden_state: Optional hidden state from previous timestep

        Returns:
            Predicted next frame [B, C, H, W]
        """
        # Handle both sequential and single frame input
        if x.dim() == 4:  # Single frame [B, C, H, W]
            x = x.unsqueeze(1)  # [B, 1, C, H, W]

        batch_size, seq_len, _, height, width = x.size()

        # Initialize hidden state
        if hidden_state is None:
            hidden_state = self._init_hidden(batch_size, (height, width))

        # Process sequence
        layer_output_list = []
        cur_layer_input = x

        for layer_idx in range(self.num_layers):
            h, c = hidden_state[layer_idx]
            output_inner = []

            for t in range(seq_len):
                # Apply input conv only for first layer
                if layer_idx == 0:
                    input_t = self.input_conv(cur_layer_input[:, t, :, :, :])
                else:
                    input_t = cur_layer_input[:, t, :, :, :]

                h, c = self.cell_list[layer_idx](input_t, (h, c))
                output_inner.append(h)

            layer_output = torch.stack(output_inner, dim=1)
            cur_layer_input = layer_output

            layer_output_list.append((h, c))

        # Use final hidden state to predict next frame
        final_hidden = layer_output_list[-1][0]
        output = self.output_conv(final_hidden)

        return output

    def _init_hidden(self, batch_size, image_size):
        """Initialize hidden states for all layers."""
        init_states = []
        for i in range(self.num_layers):
            init_states.append(
                self.cell_list[i].init_hidden(batch_size, image_size)
            )
        return init_states

    def count_parameters(self) -> int:
        """Count total trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


if __name__ == "__main__":
    # Test the model
    model = ConvLSTM()
    print(f"ConvLSTM created")
    print(f"Parameters: {model.count_parameters():,}")

    # Test with single frame
    dummy_input = torch.randn(2, 3, 128, 128)
    output = model(dummy_input)
    print(f"Single frame input shape: {dummy_input.shape}")
    print(f"Output shape: {output.shape}")

    # Test with sequence
    dummy_seq = torch.randn(2, 5, 3, 128, 128)
    output_seq = model(dummy_seq)
    print(f"Sequence input shape: {dummy_seq.shape}")
    print(f"Output shape: {output_seq.shape}")

    print("✅ Model test passed!")
