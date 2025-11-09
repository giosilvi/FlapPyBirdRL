from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch
from torch import nn
from torch.optim import Adam

from .q_network import QNetwork


@dataclass
class DQNConfig:
    state_dim: int = 8  # 6 base features + 2 gap velocities
    action_dim: int = 2
    gamma: float = 0.99
    lr: float = 1e-3
    batch_size: int = 64
    target_update_every: int = 1_000
    eps_start: float = 1.0
    eps_end: float = 0.01  # Lower minimum for better exploitation
    eps_decay_steps: int = 90_000  # Default: decay over 90k epsilon_greedy calls (matches 100k training steps with 10k warmup)
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

        self.steps = 0  # Steps where epsilon_greedy was called (for epsilon decay)
        self.optimize_steps = 0  # Number of optimization steps (for target network updates)
        self.eps = cfg.eps_start

    def greedy_action(self, state_np: np.ndarray) -> int:
        """Select action using greedy policy (no exploration, always best action)"""
        state = torch.from_numpy(state_np).float().unsqueeze(0).to(self.device)
        with torch.no_grad():
            q_values = self.q(state)
            action = int(torch.argmax(q_values, dim=-1).item())
        return action
    
    def epsilon_greedy(self, state_np: np.ndarray) -> int:
        """Select action using epsilon-greedy policy. Returns action."""
        self.steps += 1
        self._update_epsilon()

        if np.random.random() < self.eps:
            return int(np.random.randint(0, self.cfg.action_dim))

        # Use greedy action when not exploring
        return self.greedy_action(state_np)
    
    def epsilon_greedy_with_flag(self, state_np: np.ndarray, skip_epsilon_update: bool = False) -> tuple[int, bool]:
        """
        Select action using epsilon-greedy policy.
        Returns: (action, was_random_exploration)
        
        Args:
            skip_epsilon_update: If True, don't update epsilon (useful when epsilon is controlled externally)
        """
        self.steps += 1
        if not skip_epsilon_update:
            self._update_epsilon()

        if np.random.random() < self.eps:
            return int(np.random.randint(0, self.cfg.action_dim)), True

        # Use greedy action when not exploring
        return self.greedy_action(state_np), False

    def _update_epsilon(self) -> None:
        # Linear decay
        frac = min(1.0, self.steps / float(self.cfg.eps_decay_steps))
        new_eps = self.cfg.eps_start + frac * (self.cfg.eps_end - self.cfg.eps_start)
        # Don't increase epsilon if we're already below eps_end (e.g., from checkpoint)
        # This allows resuming with lower epsilon without it jumping back up
        self.eps = min(self.eps, new_eps)

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

        # Update target network based on optimization steps, not epsilon_greedy steps
        self.optimize_steps += 1
        if self.optimize_steps % self.cfg.target_update_every == 0:
            self.target.load_state_dict(self.q.state_dict())

        return float(loss.item())

    def save(self, path: str) -> None:
        torch.save(
            {
                "q": self.q.state_dict(),
                "target": self.target.state_dict(),
                "opt": self.opt.state_dict(),
                "steps": self.steps,
                "optimize_steps": self.optimize_steps,
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
        self.optimize_steps = ckpt.get("optimize_steps", 0)
        self.eps = ckpt.get("eps", self.cfg.eps_start)


