"""
SharedPolicy (π_θ)
==================
A single MLP applied at every level of the AdaptHRL hierarchy.

Key design choices (from paper §3.2):
  - Shared weights across all hierarchy levels (parameter reuse).
  - Spectral normalisation on all linear layers to enforce Lipschitz
    continuity with constant K (Assumption 2 in paper §4).
  - Output dimension = max(|G|, |A|) so the same head produces both
    sub-goals and primitive actions depending on context.

The CoT correspondence (Table 2 in paper §3.5):
  - This module is the "single language model M" analogue.
  - Applied recursively: output at level l+1 feeds in as input at level l.
"""

from __future__ import annotations

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import spectral_norm


def _make_linear(in_dim: int, out_dim: int, use_spectral_norm: bool = True) -> nn.Linear:
    layer = nn.Linear(in_dim, out_dim)
    # Kaiming initialisation for ReLU networks
    nn.init.kaiming_normal_(layer.weight, mode="fan_in", nonlinearity="relu")
    nn.init.zeros_(layer.bias)
    if use_spectral_norm:
        layer = spectral_norm(layer)
    return layer


class SharedPolicy(nn.Module):
    """
    Shared policy network π_θ(s, g) → output ∈ R^{max(|G|, |A|)}.

    Used at every hierarchy level:
      - At intermediate levels: output is interpreted as a sub-goal g_l ∈ G.
      - At the bottom level: output is interpreted as a primitive action a ∈ A.

    Parameters
    ----------
    state_dim : int
        Dimensionality of the state vector s.
    goal_dim : int
        Dimensionality of the goal / sub-goal vector g.
    output_dim : int
        max(|G|, |A|) — shared output head dimension.
    hidden_dim : int
        Width of each hidden layer (default: 256, as in paper).
    n_layers : int
        Number of hidden layers (default: 3, as in paper).
    use_spectral_norm : bool
        Whether to apply spectral normalisation (default: True).
        Set False only for ablations (removes Lipschitz guarantee).
    """

    def __init__(
        self,
        state_dim: int,
        goal_dim: int,
        output_dim: int,
        hidden_dim: int = 256,
        n_layers: int = 3,
        use_spectral_norm: bool = True,
    ) -> None:
        super().__init__()
        self.state_dim = state_dim
        self.goal_dim = goal_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.use_spectral_norm = use_spectral_norm

        # Build MLP: [s, g] → hidden → ... → output
        in_dim = state_dim + goal_dim
        layers: list[nn.Module] = []
        for _ in range(n_layers):
            layers.append(_make_linear(in_dim, hidden_dim, use_spectral_norm))
            layers.append(nn.ReLU(inplace=True))
            in_dim = hidden_dim
        # Output layer — no spectral norm on output (standard practice)
        layers.append(nn.Linear(hidden_dim, output_dim))
        nn.init.zeros_(layers[-1].bias)

        self.net = nn.Sequential(*layers)

        # Separate log-std for stochastic policy (PPO)
        self.log_std = nn.Parameter(torch.zeros(output_dim))

    def forward(
        self,
        state: torch.Tensor,
        goal: torch.Tensor,
        deterministic: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass.

        Parameters
        ----------
        state : (B, state_dim) or (state_dim,)
        goal  : (B, goal_dim)  or (goal_dim,)
        deterministic : bool
            If True, return mean output only (used at eval time).

        Returns
        -------
        output : (B, output_dim) — sub-goal or action mean
        log_prob : (B,) or None — log-probability under the Gaussian policy
        """
        if state.dim() == 1:
            state = state.unsqueeze(0)
        if goal.dim() == 1:
            goal = goal.unsqueeze(0)

        x = torch.cat([state, goal], dim=-1)
        mean = self.net(x)

        if deterministic:
            return mean, None

        std = self.log_std.exp().expand_as(mean)
        dist = torch.distributions.Normal(mean, std)
        output = dist.rsample()
        log_prob = dist.log_prob(output).sum(dim=-1)
        return output, log_prob

    def evaluate_actions(
        self,
        state: torch.Tensor,
        goal: torch.Tensor,
        action: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute log-prob and entropy of given actions — used by PPO update.

        Returns
        -------
        log_prob : (B,)
        entropy  : (B,)
        """
        x = torch.cat([state, goal], dim=-1)
        mean = self.net(x)
        std = self.log_std.exp().expand_as(mean)
        dist = torch.distributions.Normal(mean, std)
        log_prob = dist.log_prob(action).sum(dim=-1)
        entropy = dist.entropy().sum(dim=-1)
        return log_prob, entropy

    @property
    def lipschitz_constant(self) -> float:
        """
        Estimate the Lipschitz constant K of the network.

        With spectral normalisation, each layer's spectral norm ≤ 1,
        so the Lipschitz constant of the composition ≤ product of
        layer norms × ReLU Lipschitz (= 1).
        """
        K = 1.0
        for module in self.net.modules():
            if isinstance(module, nn.Linear):
                sigma = torch.linalg.matrix_norm(module.weight, ord=2).item()
                K *= sigma
        return K

    def extra_repr(self) -> str:
        return (
            f"state_dim={self.state_dim}, goal_dim={self.goal_dim}, "
            f"output_dim={self.output_dim}, hidden_dim={self.hidden_dim}, "
            f"spectral_norm={self.use_spectral_norm}"
        )
