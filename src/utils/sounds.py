import sys

import pygame


class Sounds:
    die: pygame.mixer.Sound
    hit: pygame.mixer.Sound
    point: pygame.mixer.Sound
    swoosh: pygame.mixer.Sound
    wing: pygame.mixer.Sound

    def __init__(self, mute: bool = False) -> None:
        # In headless/server environments without an audio device, initializing
        # the mixer will fail. When muted, we avoid loading sounds entirely and
        # use lightweight no-op placeholders.
        if mute:
            class _Silent:
                def set_volume(self, *args, **kwargs):
                    return None

                def play(self, *args, **kwargs):
                    return None

            self.die = _Silent()
            self.hit = _Silent()
            self.point = _Silent()
            self.swoosh = _Silent()
            self.wing = _Silent()
            return

        if "win" in sys.platform:
            ext = "wav"
        else:
            ext = "ogg"

        self.die = pygame.mixer.Sound(f"assets/audio/die.{ext}")
        self.hit = pygame.mixer.Sound(f"assets/audio/hit.{ext}")
        self.point = pygame.mixer.Sound(f"assets/audio/point.{ext}")
        self.swoosh = pygame.mixer.Sound(f"assets/audio/swoosh.{ext}")
        self.wing = pygame.mixer.Sound(f"assets/audio/wing.{ext}")
