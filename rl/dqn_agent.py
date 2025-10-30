from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch
from torch import nn
from torch.optim import Adam

from .q_network import QNetwork


@dataclass
class DQNConfig:
    state_dim: int = 6
    action_dim: int = 2
    gamma: float = 0.99
    lr: float = 1e-3
    batch_size: int = 64
    target_update_every: int = 1_000
    eps_start: float = 1.0
    eps_end: float = 0.05
    eps_decay_steps: int = 200_000
    grad_clip: float = 5.0
    device: str = "cpu"


class DQNAgent:
    def __init__(self, cfg: DQNConfig) -> None:
        self.cfg = cfg
        self.device = torch.device(cfg.device)

        self.q = QNetwork(cfg.state_dim, cfg.action_dim).to(self.device)
        self.target = QNetwork(cfg.state_dim, cfg.action_dim).to(self.device)
        self.target.load_state_dict(self.q.state_dict())
        self.target.eval()

        self.opt = Adam(self.q.parameters(), lr=cfg.lr)

        self.steps = 0
        self.eps = cfg.eps_start

    def epsilon_greedy(self, state_np: np.ndarray) -> int:
        self.steps += 1
        self._update_epsilon()

        if np.random.random() < self.eps:
            return int(np.random.randint(0, self.cfg.action_dim))

        state = torch.from_numpy(state_np).float().unsqueeze(0).to(self.device)
        with torch.no_grad():
            q_values = self.q(state)
            action = int(torch.argmax(q_values, dim=-1).item())
        return action

    def _update_epsilon(self) -> None:
        # Linear decay
        frac = min(1.0, self.steps / float(self.cfg.eps_decay_steps))
        self.eps = self.cfg.eps_start + frac * (self.cfg.eps_end - self.cfg.eps_start)

    def optimize(self, batch: dict) -> float:
        state = torch.from_numpy(batch["state"]).float().to(self.device)
        action = torch.from_numpy(batch["action"]).long().to(self.device)
        reward = torch.from_numpy(batch["reward"]).float().to(self.device)
        next_state = torch.from_numpy(batch["next_state"]).float().to(self.device)
        done = torch.from_numpy(batch["done"]).float().to(self.device)

        # Q(s,a)
        q = self.q(state).gather(1, action)

        with torch.no_grad():
            q_next = self.target(next_state).max(dim=1, keepdim=True)[0]
            target = reward + (1.0 - done) * self.cfg.gamma * q_next

        loss = nn.functional.smooth_l1_loss(q, target)

        self.opt.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(self.q.parameters(), self.cfg.grad_clip)
        self.opt.step()

        if self.steps % self.cfg.target_update_every == 0:
            self.target.load_state_dict(self.q.state_dict())

        return float(loss.item())

    def save(self, path: str) -> None:
        torch.save(
            {
                "q": self.q.state_dict(),
                "target": self.target.state_dict(),
                "opt": self.opt.state_dict(),
                "steps": self.steps,
                "eps": self.eps,
                "cfg": self.cfg.__dict__,
            },
            path,
        )

    def load(self, path: str, map_location: Optional[str] = None) -> None:
        ckpt = torch.load(path, map_location=map_location or self.device)
        self.q.load_state_dict(ckpt["q"])
        self.target.load_state_dict(ckpt["target"])
        self.opt.load_state_dict(ckpt["opt"])
        self.steps = ckpt.get("steps", 0)
        self.eps = ckpt.get("eps", self.cfg.eps_start)


