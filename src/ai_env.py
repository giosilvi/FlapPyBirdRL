import os
import random
from typing import Dict, Optional, Tuple

import numpy as np
import pygame

from .entities.background import Background
from .entities.floor import Floor
from .entities.pipe import Pipes
from .entities.player import Player, PlayerMode
from .entities.score import Score
from .utils.game_config import GameConfig
from .utils.images import Images
from .utils.sounds import Sounds
from .utils.utils import clamp
from .utils.window import Window


class FlappyEnv:
    """
    Minimal gym-like environment wrapper for FlapPyBird.

    Observation (float32, shape=(6,)):
      [bird_y, bird_vel, dx1, dy1, dx2, dy2]
      - bird_y:           normalized by viewport height
      - bird_vel:         clamped to [-VEL_CAP, VEL_CAP] then normalized
      - dx*:              (pipe.x - bird.x) / window.width
      - dy*:              (gap_center_y - bird.y) / viewport height

    Actions (discrete):
      0 = do nothing
      1 = flap

    Rewards:
      +1  when passing a pipe
      -1  on death (episode ends)
      -0.01 per step (small time penalty)
    """

    def __init__(
        self,
        render: bool = False,
        seed: Optional[int] = None,
        step_penalty: float = -0.01,
        fps: int = 0,
        mute: Optional[bool] = None,
        flap_cost: float = 0.0,
        out_of_bounds_cost: float = 0.0,
        moving_gaps: bool = False,
        gap_amp_px: float = 20.0,
        gap_freq_hz: float = 0.5,
        include_gap_vel: bool = False,
    ) -> None:
        # Headless mode if not rendering
        if not render:
            os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

        pygame.init()
        pygame.display.set_caption("Flappy Bird (RL)")

        self.window = Window(288, 512)
        self.screen = pygame.display.set_mode((self.window.width, self.window.height))
        self.clock = pygame.time.Clock()

        # Seeding
        self._seed = seed
        if seed is not None:
            random.seed(seed)

        self.images = Images()
        # Default: mute when not rendering; allow explicit override via arg
        use_mute = (not render) if mute is None else bool(mute)
        self.sounds = Sounds(mute=use_mute)
        self.config = GameConfig(
            screen=self.screen,
            clock=self.clock,
            fps=30 if fps <= 0 else fps,
            window=self.window,
            images=self.images,
            sounds=self.sounds,
            moving_gaps=moving_gaps,
            gap_amp_px=gap_amp_px,
            gap_freq_hz=gap_freq_hz,
        )

        self.render_enabled = render
        self.step_penalty = step_penalty
        self.flap_cost = float(flap_cost)
        self.out_of_bounds_cost = float(out_of_bounds_cost)
        self.episode = 0
        self.total_steps = 0
        self.include_gap_vel = bool(include_gap_vel)

        self._init_episode_state()

    def _init_episode_state(self) -> None:
        # Entities
        self.background = Background(self.config)
        self.floor = Floor(self.config)
        self.player = Player(self.config)
        self.player.set_mode(PlayerMode.NORMAL)
        self.pipes = Pipes(self.config)
        self.score = Score(self.config)
        self.score.reset()

        # Bookkeeping
        self._prev_score = 0
        self._done = False
        # Track last seen gap centers for velocity estimate keyed by pipe index
        self._last_gap_centers = {}

    def seed(self, seed: Optional[int]) -> None:
        self._seed = seed
        if seed is not None:
            random.seed(seed)

    def reset(self) -> np.ndarray:
        if self._seed is not None:
            random.seed(self._seed)
            # re-randomize sprite choices deterministically
            self.images.randomize()

        self._init_episode_state()
        self.episode += 1
        obs = self._get_state()
        return obs

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, Dict]:
        if self._done:
            return self._get_state(), 0.0, True, {}

        # Handle minimal event processing (allow quitting cleanly)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                raise SystemExit

        # Action
        if action == 1:
            self.player.flap()

        # Physics/game update order mirrors main game loop
        self.background.tick()
        self.floor.tick()
        self.pipes.tick()
        # score must tick before player for correct draw order, logic independent
        self.score.tick()
        self.player.tick()

        # Collisions end the episode
        if self.player.collided(self.pipes, self.floor):
            self._done = True

        # Check pipes crossed to update score (as in flappy.play)
        for pipe in list(self.pipes.upper):
            if self.player.crossed(pipe):
                self.score.add()

        # Reward shaping
        reward = 0.0
        reward += (self.score.score - self._prev_score) * 1.0
        reward += self.step_penalty
        # Penalize flapping slightly to reduce over-flapping behavior
        if action == 1 and not self._done and self.flap_cost > 0.0:
            reward -= self.flap_cost
        if self._done:
            reward = -1.0  # override on death
        else:
            # Penalize being out of the visible viewport (above the top)
            if self.player.y < 0 and self.out_of_bounds_cost > 0.0:
                reward -= self.out_of_bounds_cost

        self._prev_score = self.score.score
        self.total_steps += 1

        # Render if requested (cap FPS only when rendering)
        if self.render_enabled:
            pygame.display.update()
            self.config.tick()

        obs = self._get_state()
        info = {
            "episode": self.episode,
            "score": self.score.score,
            "epsilon": None,
            "steps": self.total_steps,
        }
        return obs, float(reward), bool(self._done), info

    def render(self) -> None:
        if self.render_enabled:
            pygame.display.update()

    def close(self) -> None:
        pygame.quit()

    def draw_hud(self, info: Dict) -> None:
        if not self.render_enabled:
            return
        font = pygame.font.SysFont("Arial", 16, True)
        hud_lines = [
            f"ep: {info.get('episode', self.episode)}",
            f"score: {info.get('score', self.score.score)}",
            f"eps: {info.get('epsilon', 'n/a')}",
            f"steps: {info.get('steps', self.total_steps)}",
        ]
        y = 5
        for line in hud_lines:
            txt = font.render(str(line), True, (255, 255, 255))
            self.screen.blit(txt, (5, y))
            y += txt.get_height() + 2

    # ----- Feature extraction -----
    def _get_state(self) -> np.ndarray:
        # Find the next two pipes ahead of the player
        next_idx, next2_idx = self._next_two_pipe_indices()
        dx1, dy1, v1 = self._pipe_deltas(next_idx, with_vel=self.include_gap_vel)
        dx2, dy2, v2 = self._pipe_deltas(next2_idx, with_vel=self.include_gap_vel)

        vy_cap = max(abs(self.player.max_vel_y), abs(self.player.min_vel_y))
        vy = clamp(self.player.vel_y, -vy_cap, vy_cap) / float(vy_cap)
        bird_y = self.player.y / float(self.window.viewport_height)

        parts = [bird_y, vy, dx1, dy1, dx2, dy2]
        if self.include_gap_vel:
            parts.extend([v1, v2])
        state = np.array(parts, dtype=np.float32)
        return state

    def _next_two_pipe_indices(self) -> Tuple[int, int]:
        # Choose indices in upper/lower lists of Pipes
        px = self.player.x
        candidates = []
        for i, up in enumerate(self.pipes.upper):
            if up.x + up.w >= px:
                candidates.append(i)
        if not candidates:
            # fallback to last two
            n = len(self.pipes.upper)
            return max(0, n - 2), max(0, n - 1)
        i1 = candidates[0]
        i2 = min(i1 + 1, len(self.pipes.upper) - 1)
        return i1, i2

    def _pipe_deltas(self, idx: int, with_vel: bool = False) -> Tuple[float, float, float]:
        idx = max(0, min(idx, len(self.pipes.upper) - 1))
        up = self.pipes.upper[idx]
        low = self.pipes.lower[idx]

        # Horizontal distance from bird to the pipe's left x
        dx = (up.x - self.player.x) / float(self.window.width)

        # Gap center y using lower pipe y and pipe gap
        gap_center_y = low.y - self.pipes.pipe_gap / 2.0
        dy = (gap_center_y - self.player.y) / float(self.window.viewport_height)

        v_norm = 0.0
        if with_vel:
            last = self._last_gap_centers.get(idx, gap_center_y)
            v = gap_center_y - last  # px per step
            v_norm = v / float(self.window.viewport_height)
            self._last_gap_centers[idx] = gap_center_y

        return float(dx), float(dy), float(v_norm)


