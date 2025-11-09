from typing import Tuple

import torch
from torch import nn


class PipeController(nn.Module):
    """
    Small neural network that controls pipe height adjustments.
    
    Input: [dx, dy, bird_y, bird_vel_y, ahead_pipe_gap_center_y] (5 features)
    - dx: normalized horizontal distance from bird to pipe
    - dy: normalized vertical distance from bird to gap center
    - bird_y: normalized bird y position
    - bird_vel_y: normalized bird vertical velocity
    - ahead_pipe_gap_center_y: normalized gap center y of pipe ahead (closer to bird)
    
    Output: dy adjustment value (-1 to 1)
    """
    
    def __init__(self, state_dim: int = 5) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Tanh(),  # Output in [-1, 1] range
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass returns dy adjustment value.
        
        Args:
            x: Input tensor of shape (batch_size, 5) or (5,)
            
        Returns:
            dy adjustment tensor of shape (batch_size, 1) or (1,)
        """
        return self.net(x)
    
    def act(self, state: torch.Tensor) -> Tuple[float, torch.Tensor]:
        """
        Get action (dy adjustment) from state.
        
        Args:
            state: Input state tensor
            
        Returns:
            Tuple of (dy_value, output_tensor)
        """
        with torch.no_grad():
            output = self.forward(state)
            dy_value = float(output.item())
        return dy_value, output

