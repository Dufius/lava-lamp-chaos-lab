"""
Lightweight visual encoder/decoder for pendulum images.

LightEncoder  : [B, 1, H, W] -> [B, latent_dim]
LightDecoder  : [B, latent_dim] -> [B, 1, H, W]

The encoder is intentionally tiny (3 conv layers) because the task —
finding where the two bobs are — is geometrically simple.  The hard
part is the dynamics, which lives in the pretrained RNN.

PendulumVideoPredictor wires them together:
  frames -> LightEncoder -> [latent_dim] -> GRU/LSTM -> [latent_dim] -> LightDecoder -> frames

The RNN weights can be loaded from a Phase-0 checkpoint and optionally frozen
for the first training stage.
"""

import torch.nn as nn


class LightEncoder(nn.Module):
    """
    3-layer conv encoder.  64x64 input -> latent_dim output.

    Architecture (for size=64):
      Conv(1,16,4,2,1)  -> 32x32
      Conv(16,32,4,2,1) -> 16x16
      Conv(32,64,4,2,1) -> 8x8
      Flatten -> 64*8*8 = 4096
      Linear(4096, 256) -> ReLU -> Linear(256, latent_dim)
    """

    def __init__(self, latent_dim=4, img_size=64):
        super().__init__()
        self.latent_dim = latent_dim

        self.conv = nn.Sequential(
            nn.Conv2d(1, 16, 4, 2, 1),
            nn.ReLU(),
            nn.Conv2d(16, 32, 4, 2, 1),
            nn.ReLU(),
            nn.Conv2d(32, 64, 4, 2, 1),
            nn.ReLU(),
        )

        feat = 64 * (img_size // 8) * (img_size // 8)
        self.head = nn.Sequential(
            nn.Linear(feat, 256),
            nn.ReLU(),
            nn.Linear(256, latent_dim),
        )

    def forward(self, x):
        # x: [B, 1, H, W]
        h = self.conv(x)
        return self.head(h.flatten(1))  # [B, latent_dim]


class LightDecoder(nn.Module):
    """
    Mirror of LightEncoder: latent_dim -> 64x64 image.

      Linear(latent_dim, 256) -> ReLU -> Linear(256, 64*8*8)
      Reshape -> [B, 64, 8, 8]
      ConvT(64,32,4,2,1) -> 16x16
      ConvT(32,16,4,2,1) -> 32x32
      ConvT(16,1,4,2,1)  -> 64x64
      Sigmoid
    """

    def __init__(self, latent_dim=4, img_size=64):
        super().__init__()
        self.feat_size = img_size // 8
        feat = 64 * self.feat_size * self.feat_size

        self.stem = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.ReLU(),
            nn.Linear(256, feat),
            nn.ReLU(),
        )

        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(64, 32, 4, 2, 1),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 16, 4, 2, 1),
            nn.ReLU(),
            nn.ConvTranspose2d(16, 1, 4, 2, 1),
            nn.Sigmoid(),
        )

    def forward(self, z):
        # z: [B, latent_dim]
        h = self.stem(z)
        h = h.view(h.size(0), 64, self.feat_size, self.feat_size)
        return self.deconv(h)  # [B, 1, H, W]


class PendulumVideoPredictor(nn.Module):
    """
    End-to-end model: frames -> encoder -> RNN -> decoder -> frames.

    The RNN must expose a `rollout(context, horizon)` method that takes
    [B, T, latent_dim] and returns [B, horizon, latent_dim].

    Parameters
    ----------
    rnn         : pretrained GRU/LSTMPredictor from rnn_predictor.py
    freeze_rnn  : if True, RNN parameters are frozen during training
    img_size    : spatial resolution
    """

    def __init__(self, rnn, img_size=64, freeze_rnn=False):
        super().__init__()
        latent_dim = rnn.head[-1].out_features  # infer from RNN head
        self.encoder = LightEncoder(latent_dim=latent_dim, img_size=img_size)
        self.decoder = LightDecoder(latent_dim=latent_dim, img_size=img_size)
        self.rnn = rnn

        if freeze_rnn:
            for p in self.rnn.parameters():
                p.requires_grad_(False)

    def encode_sequence(self, frames):
        """frames: [B, T, 1, H, W] -> latents: [B, T, latent_dim]"""
        B, T, C, H, W = frames.shape
        flat = frames.view(B * T, C, H, W)
        z = self.encoder(flat)
        return z.view(B, T, -1)

    def forward(self, context_frames, target_frames):
        """
        Teacher-forced single-step prediction loss.

        Parameters
        ----------
        context_frames : [B, T, 1, H, W]
        target_frames  : [B, 1, H, W]

        Returns
        -------
        pred_frames : [B, 1, H, W]
        recon_loss  : pixel MSE on context reconstruction
        pred_loss   : pixel MSE on next-frame prediction
        """
        # Encode context
        z_context = self.encode_sequence(context_frames)  # [B, T, latent_dim]

        # Reconstruct context frames (autoencoder loss to keep encoder grounded)
        B, T, _ = z_context.shape
        recon = self.decoder(z_context.view(B * T, -1))  # [B*T, 1, H, W]
        recon = recon.view(B, T, 1, *recon.shape[2:])
        recon_loss = nn.functional.mse_loss(recon, context_frames)

        # Predict next latent via RNN
        z_pred, _ = self.rnn(z_context)  # [B, latent_dim]

        # Decode prediction
        pred_frames = self.decoder(z_pred)  # [B, 1, H, W]
        pred_loss = nn.functional.mse_loss(pred_frames, target_frames)

        return pred_frames, recon_loss, pred_loss

    def rollout(self, context_frames, horizon):
        """
        Auto-regressive visual rollout.

        Parameters
        ----------
        context_frames : [B, T, 1, H, W]
        horizon        : int

        Returns
        -------
        [B, horizon, 1, H, W]
        """
        z_context = self.encode_sequence(context_frames)
        z_preds = self.rnn.rollout(z_context, horizon)  # [B, horizon, latent_dim]

        B, H_out, latent_dim = z_preds.shape
        decoded = self.decoder(z_preds.view(B * H_out, latent_dim))
        return decoded.view(B, H_out, 1, *decoded.shape[2:])
