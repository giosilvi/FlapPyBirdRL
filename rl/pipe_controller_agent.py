from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import numpy as np
import torch
from torch import nn
from torch.nn import init
from torch.optim import Adam

from .pipe_controller import PipeController


@dataclass
class PipeControllerConfig:
    state_dim: int = 5  # [dx, dy, bird_y, bird_vel_y, ahead_pipe_gap_center_y]
    lr: float = 1e-3
    batch_size: int = 64
    capacity: int = 50_000
    grad_clip: float = 5.0
    device: str = "cpu"
    num_controllers: int = 4
    eps_start: float = 0.5  # Epsilon for exploration (start)
    eps_end: float = 0.25  # Epsilon for exploration (end)
    eps_decay_steps: int = 50_000  # Steps to decay epsilon


class PipeControllerAgent:
    """
    Manages 4 pipe controller networks that learn adversarially
    (trying to make the game harder for the bird).
    """
    
    def __init__(self, cfg: PipeControllerConfig) -> None:
        self.cfg = cfg
        self.device = torch.device(cfg.device)
        
        self.eps = cfg.eps_start
        self.eps_steps = 0
        
        # Track previous action for each controller (for momentum)
        self.prev_actions = [0.0] * cfg.num_controllers
        
        # Create 4 pipe controller networks with diverse initialization
        self.controllers: List[PipeController] = []
        self.optimizers: List[Adam] = []
        
        for i in range(cfg.num_controllers):
            torch.manual_seed(42 + i * 100)
            controller = PipeController(cfg.state_dim).to(self.device)
            
            self._diverse_init(controller, i)
            
            optimizer = Adam(controller.parameters(), lr=cfg.lr)
            self.controllers.append(controller)
            self.optimizers.append(optimizer)
        
        self.replay_buffer = self._create_replay_buffer(cfg.state_dim, cfg.capacity, seed=42)
        
        self.optimize_steps = 0
    
    def _diverse_init(self, controller: PipeController, controller_idx: int) -> None:
        """Initialize controller with diverse weights for different behaviors."""
        for name, param in controller.named_parameters():
            if 'weight' in name:
                if controller_idx == 0:
                    # Xavier uniform initialization
                    init.xavier_uniform_(param)
                elif controller_idx == 1:
                    # He initialization (good for ReLU)
                    init.kaiming_uniform_(param, nonlinearity='relu')
                elif controller_idx == 2:
                    # Small random weights
                    init.normal_(param, mean=0.0, std=0.1)
                else:  # controller_idx == 3
                    # Larger random weights
                    init.normal_(param, mean=0.0, std=0.3)
            elif 'bias' in name:
                # Initialize biases differently for each controller
                if controller_idx == 0:
                    init.zeros_(param)
                elif controller_idx == 1:
                    init.constant_(param, 0.1)
                elif controller_idx == 2:
                    init.normal_(param, mean=0.0, std=0.05)
                else:  # controller_idx == 3
                    init.normal_(param, mean=0.0, std=0.1)
    
    def _create_replay_buffer(self, state_dim: int, capacity: int, seed: int):
        """Create a replay buffer that handles float actions."""
        rng = np.random.default_rng(seed)
        return {
            "capacity": capacity,
            "state": np.zeros((capacity, state_dim), dtype=np.float32),
            "action": np.zeros((capacity, 1), dtype=np.float32),  # Float for dy values
            "reward": np.zeros((capacity, 1), dtype=np.float32),
            "next_state": np.zeros((capacity, state_dim), dtype=np.float32),
            "done": np.zeros((capacity, 1), dtype=np.float32),
            "pos": 0,
            "full": False,
            "rng": rng,
        }
    
    def _buffer_len(self, buffer: dict) -> int:
        return buffer["capacity"] if buffer["full"] else buffer["pos"]
    
    def _buffer_push(
        self,
        buffer: dict,
        state: np.ndarray,
        action: float,
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ) -> None:
        idx = buffer["pos"]
        buffer["state"][idx] = state
        buffer["action"][idx] = action
        buffer["reward"][idx] = reward
        buffer["next_state"][idx] = next_state
        buffer["done"][idx] = float(done)
        buffer["pos"] = (buffer["pos"] + 1) % buffer["capacity"]
        buffer["full"] = buffer["full"] or buffer["pos"] == 0
    
    def _buffer_sample(self, buffer: dict, batch_size: int) -> dict:
        size = self._buffer_len(buffer)
        assert size >= batch_size, "Not enough samples in buffer"
        idxs = buffer["rng"].integers(0, size, size=batch_size)
        return {
            "state": buffer["state"][idxs],
            "action": buffer["action"][idxs],
            "reward": buffer["reward"][idxs],
            "next_state": buffer["next_state"][idxs],
            "done": buffer["done"][idxs],
        }
    
    def greedy_action(self, controller_idx: int, state_np: np.ndarray) -> float:
        """
        Get action (dy adjustment) using greedy policy (no exploration).
        
        Args:
            controller_idx: Index of controller (0-3)
            state_np: State as numpy array
            
        Returns:
            dy adjustment value (-1 to 1)
        """
        if controller_idx < 0 or controller_idx >= len(self.controllers):
            return 0.0
        
        # Greedy action: use network prediction directly (no exploration)
        controller = self.controllers[controller_idx]
        state = torch.from_numpy(state_np).float().unsqueeze(0).to(self.device)
        dy_value, _ = controller.act(state)
        return dy_value
    
    def act(self, controller_idx: int, state_np: np.ndarray, greedy: bool = False) -> float:
        """
        Get action (dy adjustment) from a specific controller.
        Uses epsilon-greedy exploration: with probability epsilon, returns random action.
        Includes momentum: more likely to continue previous movement direction.
        
        Args:
            controller_idx: Index of controller (0-3)
            state_np: State as numpy array
            greedy: If True, use greedy policy (no exploration)
            
        Returns:
            dy adjustment value (-1 to 1)
        """
        if controller_idx < 0 or controller_idx >= len(self.controllers):
            return 0.0
        
        # If greedy mode, skip exploration but still apply momentum
        if greedy:
            action = self.greedy_action(controller_idx, state_np)
            # Apply momentum bias (70% continue, 30% new action)
            if abs(self.prev_actions[controller_idx]) > 0.01:  # If there was previous movement
                if np.random.random() < 0.7:  # 70% chance to continue
                    # Blend previous action with new action
                    action = 0.7 * self.prev_actions[controller_idx] + 0.3 * action
            self.prev_actions[controller_idx] = action
            return action
        
        # Update epsilon (decay over time)
        self.eps_steps += 1
        frac = min(1.0, self.eps_steps / float(self.cfg.eps_decay_steps))
        self.eps = self.cfg.eps_start + frac * (self.cfg.eps_end - self.cfg.eps_start)
        
        # Epsilon-greedy: explore with probability epsilon
        if np.random.random() < self.eps:
            # Random exploration: biased towards continuing previous movement
            if abs(self.prev_actions[controller_idx]) > 0.01:  # If there was previous movement
                if np.random.random() < 0.6:  # 60% chance to continue direction
                    # Continue in same direction with some randomness
                    prev_dir = np.sign(self.prev_actions[controller_idx])
                    action = prev_dir * np.random.uniform(0.3, 1.0)  # Continue direction, random magnitude
                else:
                    # Random exploration
                    action = float(np.random.uniform(-1.0, 1.0))
            else:
                # No previous movement, pure random
                action = float(np.random.uniform(-1.0, 1.0))
        else:
            # Greedy action: use network prediction
            controller = self.controllers[controller_idx]
            state = torch.from_numpy(state_np).float().unsqueeze(0).to(self.device)
            action, _ = controller.act(state)
            # Convert to float (handle both tensor and float cases)
            if isinstance(action, torch.Tensor):
                action = float(action.item())
            else:
                action = float(action)
            
            # Apply momentum bias (70% continue, 30% new action)
            if abs(self.prev_actions[controller_idx]) > 0.01:  # If there was previous movement
                if np.random.random() < 0.7:  # 70% chance to continue
                    # Blend previous action with new action
                    action = 0.7 * self.prev_actions[controller_idx] + 0.3 * action
        
        # Store action for next step
        self.prev_actions[controller_idx] = action
        return action
    
    def push_experience(
        self,
        controller_idx: int,
        state: np.ndarray,
        action: float,
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ) -> None:
        """
        Store experience in replay buffer.
        Note: We use a shared buffer, but could track per-controller if needed.
        
        Args:
            controller_idx: Index of controller (0-3)
            state: Current state
            action: Action taken (dy value)
            reward: Adversarial reward (negative of bird's reward)
            next_state: Next state
            done: Whether episode ended
        """
        self._buffer_push(self.replay_buffer, state, action, reward, next_state, done)
    
    def optimize(self) -> Optional[float]:
        """Train controllers using Deterministic Policy Gradient (DPG) with stratified sampling."""
        buffer_size = self._buffer_len(self.replay_buffer)
        if buffer_size < self.cfg.batch_size:
            return None
        
        # Get all nonzero experiences from buffer for stratified sampling
        all_rewards = self.replay_buffer["reward"][:buffer_size].squeeze()
        nonzero_mask = np.abs(all_rewards) > 1e-6
        nonzero_indices = np.where(nonzero_mask)[0]
        
        if len(nonzero_indices) == 0:
            return None
        
        # Separate into death cases (+1.0) and pipe pass cases (-1.0)
        positive_reward_indices = nonzero_indices[all_rewards[nonzero_indices] > 0.5]  # Death cases
        negative_reward_indices = nonzero_indices[all_rewards[nonzero_indices] < -0.5]  # Pipe pass cases
        
        # Stratified sampling: sample equally from both types
        batch_size = self.cfg.batch_size
        half_batch = batch_size // 2
        
        selected_indices = []
        
        # Sample from death cases (+1.0)
        if len(positive_reward_indices) > 0:
            n_positive = min(half_batch, len(positive_reward_indices))
            positive_selected = self.replay_buffer["rng"].choice(
                positive_reward_indices, size=n_positive, replace=False
            )
            selected_indices.extend(positive_selected)
        
        # Sample from pipe pass cases (-1.0)
        if len(negative_reward_indices) > 0:
            n_negative = min(batch_size - len(selected_indices), len(negative_reward_indices))
            negative_selected = self.replay_buffer["rng"].choice(
                negative_reward_indices, size=n_negative, replace=False
            )
            selected_indices.extend(negative_selected)
        
        # If we still need more samples, fill from remaining nonzero experiences
        remaining_needed = batch_size - len(selected_indices)
        if remaining_needed > 0:
            remaining_indices = np.setdiff1d(nonzero_indices, selected_indices)
            if len(remaining_indices) > 0:
                n_remaining = min(remaining_needed, len(remaining_indices))
                remaining_selected = self.replay_buffer["rng"].choice(
                    remaining_indices, size=n_remaining, replace=False
                )
                selected_indices.extend(remaining_selected)
        
        if len(selected_indices) == 0:
            return None
        
        # Build batch from selected indices
        selected_indices = np.array(selected_indices)
        filtered_batch = {
            "state": self.replay_buffer["state"][selected_indices],
            "action": self.replay_buffer["action"][selected_indices],
            "reward": self.replay_buffer["reward"][selected_indices],
            "next_state": self.replay_buffer["next_state"][selected_indices],
            "done": self.replay_buffer["done"][selected_indices],
        }
        
        state = torch.from_numpy(filtered_batch["state"]).float().to(self.device)
        action_taken = torch.from_numpy(filtered_batch["action"]).float().squeeze(-1).to(self.device)
        reward = torch.from_numpy(filtered_batch["reward"]).float().squeeze(-1).to(self.device)
        
        filtered_size = len(reward)
        if filtered_size <= 2:
            advantage = reward
        else:
            advantage = reward - reward.mean()
        
        total_loss = 0.0
        
        for controller, optimizer in zip(self.controllers, self.optimizers):
            predicted_action = controller(state).squeeze(-1)
            
            aggressive_scale = 2.0
            target_action = torch.where(
                advantage > 0,
                action_taken,
                -action_taken * aggressive_scale
            )
            target_action = torch.clamp(target_action, -1.0, 1.0)
            
            weight = torch.clamp(torch.abs(advantage), min=0.1)
            action_error = predicted_action - target_action
            loss = (action_error ** 2 * weight).mean()
            
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(controller.parameters(), self.cfg.grad_clip)
            optimizer.step()
            
            total_loss += loss.item()
        
        self.optimize_steps += 1
        return total_loss / len(self.controllers)
    
    def save(self, path: str) -> None:
        """Save all controllers and optimizers."""
        checkpoint_dir = Path(path)
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        for i, (controller, optimizer) in enumerate(zip(self.controllers, self.optimizers)):
            ckpt_path = checkpoint_dir / f"pipe_controller_{i}.pt"
            torch.save(
                {
                    "controller": controller.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "optimize_steps": self.optimize_steps,
                    "eps": self.eps,
                    "eps_steps": self.eps_steps,
                    "cfg": self.cfg.__dict__,
                },
                ckpt_path,
            )
        
        # Save shared replay buffer info
        buffer_path = checkpoint_dir / "replay_buffer.npz"
        buffer_len = self._buffer_len(self.replay_buffer)
        np.savez(
            buffer_path,
            pos=self.replay_buffer["pos"],
            full=self.replay_buffer["full"],
            state=self.replay_buffer["state"][:buffer_len],
            action=self.replay_buffer["action"][:buffer_len],
            reward=self.replay_buffer["reward"][:buffer_len],
            next_state=self.replay_buffer["next_state"][:buffer_len],
            done=self.replay_buffer["done"][:buffer_len],
        )
    
    def load(self, path: str, map_location: Optional[str] = None) -> None:
        """Load all controllers and optimizers."""
        checkpoint_dir = Path(path)
        
        for i in range(self.cfg.num_controllers):
            ckpt_path = checkpoint_dir / f"pipe_controller_{i}.pt"
            if ckpt_path.exists():
                ckpt = torch.load(ckpt_path, map_location=map_location or self.device)
                self.controllers[i].load_state_dict(ckpt["controller"])
                self.optimizers[i].load_state_dict(ckpt["optimizer"])
                self.optimize_steps = ckpt.get("optimize_steps", 0)
                # Load epsilon state if available (for resuming exploration)
                if "eps" in ckpt:
                    self.eps = ckpt["eps"]
                if "eps_steps" in ckpt:
                    self.eps_steps = ckpt["eps_steps"]
        
        # Load replay buffer if available
        buffer_path = checkpoint_dir / "replay_buffer.npz"
        if buffer_path.exists():
            data = np.load(buffer_path)
            self.replay_buffer["pos"] = int(data["pos"])
            self.replay_buffer["full"] = bool(data["full"])
            # Restore buffer contents
            buffer_len = len(data["state"])
            if buffer_len > 0:
                self.replay_buffer["state"][:buffer_len] = data["state"]
                self.replay_buffer["action"][:buffer_len] = data["action"]
                self.replay_buffer["reward"][:buffer_len] = data["reward"]
                self.replay_buffer["next_state"][:buffer_len] = data["next_state"]
                self.replay_buffer["done"][:buffer_len] = data["done"]

