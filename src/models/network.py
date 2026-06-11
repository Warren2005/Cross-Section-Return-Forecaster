"""
ReturnPredictor: 5-layer feed-forward neural network for cross-sectional
return prediction.

Architecture matches Gu, Kelly & Xiu (2020) NN5 specification:
  Input → 32 → 16 → 8 → 4 → 1

Additions vs. Gu et al. (see LEARNING.md §4.1 for rationale):
  - BatchNorm1d before each ReLU (stabilises training on the heterogeneous
    cross-sectional feature distributions)
  - Skip connection from layer 1 to layer 3 (improves gradient flow through
    the deeper layers without adding parameters to the main path)
  - Huber loss (not part of the network, but used at training time) instead
    of MSE — robustness to fat-tailed monthly return outliers
"""

from __future__ import annotations

import torch
import torch.nn as nn


class ReturnPredictor(nn.Module):
    """
    Parameters
    ----------
    n_chars  : total input features (default 168 = 94 chars + 74 industry dummies)
    dropout  : dropout probability applied after layers 2 and 4 (default 0.10)
    """

    def __init__(self, n_chars: int = 168, dropout: float = 0.10):
        super().__init__()

        self.layer1 = nn.Sequential(
            nn.Linear(n_chars, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
        )
        self.layer2 = nn.Sequential(
            nn.Linear(32, 16),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.layer3 = nn.Sequential(
            nn.Linear(16, 8),
            nn.BatchNorm1d(8),
            nn.ReLU(),
        )
        self.layer4 = nn.Sequential(
            nn.Linear(8, 4),
            nn.BatchNorm1d(4),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.output_layer = nn.Linear(4, 1)

        # Skip connection: projects the 32-dim layer1 output to 8 dims so it
        # can be added to the layer3 output before layer4.
        self.skip_proj = nn.Linear(32, 8)

        self._init_weights()

    def _init_weights(self) -> None:
        """Kaiming uniform initialisation for all Linear layers."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight, nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : (batch_size, n_chars) float32 tensor — rank-normalised features,
            NaN already imputed to 0 by the DataLoader.

        Returns
        -------
        (batch_size,) float32 — predicted monthly excess returns.
        """
        h1 = self.layer1(x)            # (B, 32)
        h2 = self.layer2(h1)           # (B, 16)
        h3 = self.layer3(h2)           # (B, 8)
        h3 = h3 + self.skip_proj(h1)   # skip: add projected h1 to h3
        h4 = self.layer4(h3)           # (B, 4)
        return self.output_layer(h4).squeeze(-1)   # (B,)


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
