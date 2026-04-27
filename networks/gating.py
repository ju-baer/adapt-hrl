"""
GatingNetwork (G_φ)
===================
Decides at each hierarchy level whether to recurse deeper (generate a sub-goal)
or commit to a primitive action.

Key design choices (from paper §3.4):
  - 2-layer MLP with sigmoid output → p_l ∈ (0, 1).
  - Self-supervised labels: y_l = 1 if ||s - g||_2 > δ̄_l, else 0.
  - δ̄_l is an adaptive running mean of distances at level l.
  - No domain knowledge required — only Euclidean state-goal distance.

Ablation result (Table 3):
  - Random labels: -6% performance → gating learns meaningful signal.
  - ±40% label noise: -3% → robust to imperfect supervision.

The CoT correspondence:
  - This network is the "learned stopping criterion" analogue of the
    answer token / confidence threshold in CoT/ToT reasoning.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class GatingNetwork(nn.Module):
    """
    G_φ(s, g) → p_l ∈ (0, 1).

    p_l ≥ τ  →  recurse (generate sub-goal at next lower level)
    p_l <  τ  →  act    (emit primitive action)

    Parameters
    ----------
    state_dim : int
    goal_dim  : int
    hidden_dim : int
        Width of hidden layers (default: 128, as in paper).
    n_levels : int
        Maximum number of hierarchy levels (L_max), used to maintain
        per-level adaptive thresholds δ̄_l.
    tau : float
        Decision threshold (default: 0.5).
    beta : float
        EMA momentum for adaptive threshold update (default: 0.99).
    """

    def __init__(
        self,
        state_dim: int,
        goal_dim: int,
        hidden_dim: int = 128,
        n_levels: int = 10,
        tau: float = 0.5,
        beta: float = 0.99,
    ) -> None:
        super().__init__()
        self.state_dim = state_dim
        self.goal_dim = goal_dim
        self.n_levels = n_levels
        self.tau = tau
        self.beta = beta

        in_dim = state_dim + goal_dim
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid(),
        )
        self._init_weights()

        # Per-level adaptive distance thresholds δ̄_l (Eq. 6 in paper).
        # Initialised to 0; updated with EMA during rollout.
        self.register_buffer(
            "delta_bar",
            torch.zeros(n_levels),
        )
        # Track whether each level has been initialised yet
        self.register_buffer(
            "delta_initialised",
            torch.zeros(n_levels, dtype=torch.bool),
        )

    def _init_weights(self) -> None:
        for m in self.net.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    # ------------------------------------------------------------------ #
    # Forward / gating decision                                           #
    # ------------------------------------------------------------------ #

    def forward(
        self,
        state: torch.Tensor,
        goal: torch.Tensor,
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        state : (B, state_dim) or (state_dim,)
        goal  : (B, goal_dim)  or (goal_dim,)

        Returns
        -------
        p : (B, 1) — recursion probability
        """
        if state.dim() == 1:
            state = state.unsqueeze(0)
        if goal.dim() == 1:
            goal = goal.unsqueeze(0)
        x = torch.cat([state, goal], dim=-1)
        return self.net(x)

    def should_recurse(
        self,
        state: torch.Tensor,
        goal: torch.Tensor,
    ) -> bool:
        """
        Hard decision: recurse if p ≥ τ.
        Used during rollout (no gradients needed).
        """
        with torch.no_grad():
            p = self.forward(state, goal)
        return p.squeeze().item() >= self.tau

    # ------------------------------------------------------------------ #
    # Self-supervised label generation (Eq. 6 in paper)                  #
    # ------------------------------------------------------------------ #

    def compute_label(
        self,
        state: torch.Tensor,
        goal: torch.Tensor,
        level: int,
    ) -> torch.Tensor:
        """
        Compute binary gating label y_l for a state-goal pair at `level`.

        y_l = 1  (recurse)  if ||s - g||_2 > δ̄_l
        y_l = 0  (act)      otherwise

        Also updates the EMA threshold δ̄_l in place.

        Parameters
        ----------
        state : (state_dim,)
        goal  : (goal_dim,)   — must be in state space for distance to make sense
        level : int           — hierarchy level index (0 = bottom)

        Returns
        -------
        label : scalar tensor, dtype=float, value ∈ {0., 1.}
        """
        assert 0 <= level < self.n_levels, f"level {level} out of range [0, {self.n_levels})"

        # Compute Euclidean distance — use state dimensions that overlap with goal
        # (goal is a subset of state space: G ⊆ S)
        s = state.detach().float()
        g = goal.detach().float()

        # If goal_dim < state_dim, compare only the first goal_dim dimensions
        dim = min(s.shape[-1], g.shape[-1])
        dist = torch.norm(s[..., :dim] - g[..., :dim], p=2).item()

        # Initialise δ̄_l to the first observed distance
        if not self.delta_initialised[level]:
            self.delta_bar[level] = dist
            self.delta_initialised[level] = True

        # EMA update: δ̄_l ← β·δ̄_l + (1−β)·dist
        self.delta_bar[level] = (
            self.beta * self.delta_bar[level] + (1.0 - self.beta) * dist
        )

        label = 1.0 if dist > self.delta_bar[level].item() else 0.0
        return torch.tensor(label, dtype=torch.float32)

    def compute_labels_batch(
        self,
        states: torch.Tensor,
        goals: torch.Tensor,
        levels: torch.Tensor,
    ) -> torch.Tensor:
        """
        Vectorised label computation for a batch of transitions.
        Used during the PPO update when all buffer data is available.

        Parameters
        ----------
        states : (B, state_dim)
        goals  : (B, goal_dim)
        levels : (B,) — integer level indices

        Returns
        -------
        labels : (B,) float
        """
        labels = torch.zeros(states.shape[0])
        for i, (s, g, l) in enumerate(zip(states, goals, levels)):
            labels[i] = self.compute_label(s, g, int(l.item()))
        return labels

    # ------------------------------------------------------------------ #
    # Loss                                                                #
    # ------------------------------------------------------------------ #

    def loss(
        self,
        states: torch.Tensor,
        goals: torch.Tensor,
        levels: torch.Tensor,
    ) -> torch.Tensor:
        """
        Binary cross-entropy gating loss (Eq. 5 in paper):
            L_gate = -E[y_l log p_l + (1−y_l) log(1−p_l)]

        Parameters
        ----------
        states : (B, state_dim)
        goals  : (B, goal_dim)
        levels : (B,) integer

        Returns
        -------
        loss : scalar
        """
        labels = self.compute_labels_batch(states, goals, levels).to(states.device)
        p = self.forward(states, goals).squeeze(-1)           # (B,)
        loss = F.binary_cross_entropy(p, labels)
        return loss

    def extra_repr(self) -> str:
        return (
            f"state_dim={self.state_dim}, goal_dim={self.goal_dim}, "
            f"tau={self.tau}, beta={self.beta}, n_levels={self.n_levels}"
        )
