from typing import Dict, Tuple

import numpy as np


class ReplayBuffer:
    def __init__(
        self,
        state_dim: int,
        capacity: int = 100_000,
        seed: int = 123,
    ) -> None:
        self.capacity = int(capacity)
        self.state = np.zeros((self.capacity, state_dim), dtype=np.float32)
        self.action = np.zeros((self.capacity, 1), dtype=np.int64)
        self.reward = np.zeros((self.capacity, 1), dtype=np.float32)
        self.next_state = np.zeros((self.capacity, state_dim), dtype=np.float32)
        self.done = np.zeros((self.capacity, 1), dtype=np.float32)

        self.pos = 0
        self.full = False
        self.rng = np.random.default_rng(seed)

    def __len__(self) -> int:
        return self.capacity if self.full else self.pos

    def push(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ) -> None:
        idx = self.pos
        self.state[idx] = state
        self.action[idx] = action
        self.reward[idx] = reward
        self.next_state[idx] = next_state
        self.done[idx] = float(done)

        self.pos = (self.pos + 1) % self.capacity
        self.full = self.full or self.pos == 0

    def sample(self, batch_size: int) -> Dict[str, np.ndarray]:
        size = len(self)
        assert size >= batch_size, "Not enough samples in buffer"
        idxs = self.rng.integers(0, size, size=batch_size)
        return {
            "state": self.state[idxs],
            "action": self.action[idxs],
            "reward": self.reward[idxs],
            "next_state": self.next_state[idxs],
            "done": self.done[idxs],
        }


