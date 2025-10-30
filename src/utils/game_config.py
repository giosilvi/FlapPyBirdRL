import os

import pygame

from .images import Images
from .sounds import Sounds
from .window import Window


class GameConfig:
    def __init__(
        self,
        screen: pygame.Surface,
        clock: pygame.time.Clock,
        fps: int,
        window: Window,
        images: Images,
        sounds: Sounds,
        moving_gaps: bool = False,
        gap_amp_px: float = 20.0,
        gap_freq_hz: float = 0.5,
    ) -> None:
        self.screen = screen
        self.clock = clock
        self.fps = fps
        self.window = window
        self.images = images
        self.sounds = sounds
        self.debug = os.environ.get("DEBUG", False)
        # Difficulty modifiers
        self.moving_gaps = moving_gaps
        self.gap_amp_px = gap_amp_px
        self.gap_freq_hz = gap_freq_hz

    def tick(self) -> None:
        self.clock.tick(self.fps)
