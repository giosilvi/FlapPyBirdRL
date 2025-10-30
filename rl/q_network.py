from typing import Tuple

import torch
from torch import nn


class QNetwork(nn.Module):
    def __init__(self, state_dim: int = 6, action_dim: int = 2) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

    def act(self, state: torch.Tensor) -> Tuple[int, torch.Tensor]:
        with torch.no_grad():
            q = self.forward(state)
            action = int(torch.argmax(q, dim=-1).item())
        return action, q


