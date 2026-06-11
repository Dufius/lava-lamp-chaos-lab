"""
RNN-based state predictors for chaotic dynamical systems.

Three architectures, all sharing the same interface:
  forward(x)  -> (pred, aux_loss)
  rollout(context, horizon) -> predictions

GRUPredictor  — deterministic GRU baseline
LSTMPredictor — deterministic LSTM baseline
VRNNPredictor — Variational RNN (Chung et al. 2015)
                Adds a learned stochastic latent z at every step so the
                model can represent predictive uncertainty, which grows
                exponentially in chaotic regimes.
"""
import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _mlp(in_dim, hidden_dim, out_dim):
    return nn.Sequential(
        nn.Linear(in_dim, hidden_dim),
        nn.ELU(),
        nn.Linear(hidden_dim, out_dim),
    )


# ---------------------------------------------------------------------------
# Deterministic baselines
# ---------------------------------------------------------------------------


class GRUPredictor(nn.Module):
    """Single-step GRU predictor.  aux_loss is always 0."""

    def __init__(self, state_dim=4, hidden_dim=128, num_layers=2, dropout=0.0):
        super().__init__()
        self.rnn = nn.GRU(
            state_dim,
            hidden_dim,
            num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.head = _mlp(hidden_dim, hidden_dim, state_dim)

    def forward(self, x, hidden=None):
        """
        Parameters
        ----------
        x      : [B, T, state_dim]
        hidden : optional GRU hidden state

        Returns
        -------
        pred       : [B, state_dim]  prediction for step T+1
        aux_loss   : scalar tensor 0.0
        """
        out, hidden = self.rnn(x, hidden)
        pred = self.head(out[:, -1])
        return pred, torch.zeros(1, device=x.device)

    def rollout(self, context, horizon):
        """
        Auto-regressive rollout.

        Parameters
        ----------
        context : [B, T, state_dim]
        horizon : int

        Returns
        -------
        [B, horizon, state_dim]
        """
        _, hidden = self.rnn(context)
        cur = context[:, -1:]  # [B, 1, state_dim]
        preds = []
        for _ in range(horizon):
            out, hidden = self.rnn(cur, hidden)
            p = self.head(out[:, -1])  # [B, state_dim]
            preds.append(p)
            cur = p.unsqueeze(1)
        return torch.stack(preds, dim=1)  # [B, horizon, state_dim]


class LSTMPredictor(nn.Module):
    """Single-step LSTM predictor.  aux_loss is always 0."""

    def __init__(self, state_dim=4, hidden_dim=128, num_layers=2, dropout=0.0):
        super().__init__()
        self.rnn = nn.LSTM(
            state_dim,
            hidden_dim,
            num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.head = _mlp(hidden_dim, hidden_dim, state_dim)

    def forward(self, x, hidden=None):
        out, hidden = self.rnn(x, hidden)
        pred = self.head(out[:, -1])
        return pred, torch.zeros(1, device=x.device)

    def rollout(self, context, horizon):
        _, hidden = self.rnn(context)
        cur = context[:, -1:]
        preds = []
        for _ in range(horizon):
            out, hidden = self.rnn(cur, hidden)
            p = self.head(out[:, -1])
            preds.append(p)
            cur = p.unsqueeze(1)
        return torch.stack(preds, dim=1)


# ---------------------------------------------------------------------------
# Variational RNN
# ---------------------------------------------------------------------------


class VRNNPredictor(nn.Module):
    """
    Variational RNN (Chung et al. 2015).

    At every timestep t:
      1. Prior    p(z_t | h_{t-1})         learned from recurrent state
      2. Posterior q(z_t | h_{t-1}, x_t)  uses observed state during training
      3. h_t = GRU(h_{t-1}, [phi_x(x_t), phi_z(z_t)])
      4. Predict  x_{t+1} from (h_t, z_t)

    Training loss = MSE + beta * KL(posterior || prior)

    The stochastic latent z gives the model a principled way to represent
    growing predictive uncertainty in chaotic trajectories.
    """

    def __init__(self, state_dim=4, hidden_dim=128, latent_dim=32, beta=1.0):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.beta = beta

        # Feature embeddings
        self.phi_x = nn.Sequential(nn.Linear(state_dim, hidden_dim), nn.ELU())
        self.phi_z = nn.Sequential(nn.Linear(latent_dim, hidden_dim), nn.ELU())

        # Prior: p(z_t | h_{t-1})
        self.prior = _mlp(hidden_dim, hidden_dim, 2 * latent_dim)

        # Posterior: q(z_t | h_{t-1}, x_t)
        self.posterior = _mlp(2 * hidden_dim, hidden_dim, 2 * latent_dim)

        # Recurrent core
        self.cell = nn.GRUCell(2 * hidden_dim, hidden_dim)

        # Decoder: p(x_{t+1} | h_t, phi_z(z_t))
        self.decoder = _mlp(2 * hidden_dim, hidden_dim, state_dim)

    # ------------------------------------------------------------------

    def _rsample(self, mu, logvar):
        std = (0.5 * logvar).exp()
        return mu + std * torch.randn_like(std)

    def _kl(self, mu_q, lv_q, mu_p, lv_p):
        """KL(N(mu_q,exp(lv_q)) || N(mu_p,exp(lv_p))) per dimension, summed."""
        return 0.5 * (
            lv_p - lv_q + (lv_q - lv_p).exp() + (mu_q - mu_p).pow(2) * (-lv_p).exp() - 1
        ).sum(-1)

    # ------------------------------------------------------------------

    def forward(self, x):
        """
        Teacher-forced forward pass over a sequence.

        Parameters
        ----------
        x : [B, T, state_dim]

        Returns
        -------
        preds    : [B, T, state_dim]   one-step predictions x_{1..T+1}
        kl_mean  : scalar              mean KL per step (for the loss)
        """
        B, T, _ = x.shape
        h = x.new_zeros(B, self.hidden_dim)
        preds, kl_sum = [], 0.0

        for t in range(T):
            x_t = x[:, t]
            phi_x_t = self.phi_x(x_t)

            # Prior from h
            prior_params = self.prior(h)
            mu_p, lv_p = prior_params.chunk(2, dim=-1)

            # Posterior from h + x_t
            post_params = self.posterior(torch.cat([h, phi_x_t], dim=-1))
            mu_q, lv_q = post_params.chunk(2, dim=-1)

            z_t = self._rsample(mu_q, lv_q)
            phi_z_t = self.phi_z(z_t)

            kl_sum = kl_sum + self._kl(mu_q, lv_q, mu_p, lv_p).mean()

            pred = self.decoder(torch.cat([h, phi_z_t], dim=-1))
            preds.append(pred)

            h = self.cell(torch.cat([phi_x_t, phi_z_t], dim=-1), h)

        return torch.stack(preds, dim=1), kl_sum / T

    def rollout(self, context, horizon):
        """
        Auto-regressive rollout using the prior (no observed states).

        Parameters
        ----------
        context : [B, T, state_dim]
        horizon : int

        Returns
        -------
        [B, horizon, state_dim]
        """
        B = context.size(0)
        h = context.new_zeros(B, self.hidden_dim)

        # Burn-in context with posterior
        with torch.no_grad():
            for t in range(context.size(1)):
                x_t = context[:, t]
                phi_x_t = self.phi_x(x_t)
                post_params = self.posterior(torch.cat([h, phi_x_t], dim=-1))
                mu_q, lv_q = post_params.chunk(2, dim=-1)
                z_t = mu_q  # use mean during eval
                phi_z_t = self.phi_z(z_t)
                h = self.cell(torch.cat([phi_x_t, phi_z_t], dim=-1), h)

            # Rollout with prior
            preds = []
            x_cur = context[:, -1]
            for _ in range(horizon):
                prior_params = self.prior(h)
                mu_p, _ = prior_params.chunk(2, dim=-1)
                z_t = mu_p
                phi_z_t = self.phi_z(z_t)
                pred = self.decoder(torch.cat([h, phi_z_t], dim=-1))
                preds.append(pred)
                phi_x_cur = self.phi_x(x_cur)
                h = self.cell(torch.cat([phi_x_cur, phi_z_t], dim=-1), h)
                x_cur = pred

        return torch.stack(preds, dim=1)
