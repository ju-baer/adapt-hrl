"""
Value function heads for PPO (critic side of actor-critic).
One value head is shared across all hierarchy levels — consistent with
the parameter-sharing philosophy of AdaptHRL.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch.nn.utils import spectral_norm


class ValueNetwork(nn.Module):
    """
    V_l(s, g) → scalar value estimate.

    Used by PPO to compute advantages Â_t (GAE).
    Shared across all hierarchy levels (same as the policy).

    Parameters
    ----------
    state_dim : int
    goal_dim  : int
    hidden_dim : int  (default 256)
    n_layers  : int   (default 3)
    use_spectral_norm : bool (default True)
    """

    def __init__(
        self,
        state_dim: int,
        goal_dim: int,
        hidden_dim: int = 256,
        n_layers: int = 3,
        use_spectral_norm: bool = True,
    ) -> None:
        super().__init__()
        in_dim = state_dim + goal_dim
        layers: list[nn.Module] = []
        for _ in range(n_layers):
            linear = nn.Linear(in_dim, hidden_dim)
            nn.init.kaiming_normal_(linear.weight, nonlinearity="relu")
            nn.init.zeros_(linear.bias)
            if use_spectral_norm:
                linear = spectral_norm(linear)
            layers += [linear, nn.ReLU(inplace=True)]
            in_dim = hidden_dim
        layers.append(nn.Linear(hidden_dim, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, state: torch.Tensor, goal: torch.Tensor) -> torch.Tensor:
        """
        Returns
        -------
        value : (B, 1)
        """
        if state.dim() == 1:
            state = state.unsqueeze(0)
        if goal.dim() == 1:
            goal = goal.unsqueeze(0)
        x = torch.cat([state, goal], dim=-1)
        return self.net(x)
