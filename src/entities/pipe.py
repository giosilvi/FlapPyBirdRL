import random
from typing import List

from ..utils import GameConfig
from .entity import Entity


class Pipe(Entity):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.vel_x = -5

    def draw(self) -> None:
        self.x += self.vel_x
        super().draw()


class Pipes(Entity):
    upper: List[Pipe]
    lower: List[Pipe]

    def __init__(self, config: GameConfig) -> None:
        super().__init__(config)
        self.pipe_gap = 120
        self.top = 0
        self.bottom = self.config.window.viewport_height
        self.upper = []
        self.lower = []
        self.frame = 0
        self.spawn_initial_pipes()

    def tick(self) -> None:
        if self.can_spawn_pipes():
            self.spawn_new_pipes()
        self.remove_old_pipes()

        # Apply vertical oscillation to gap center if enabled
        if getattr(self.config, "moving_gaps", False):
            self.frame += 1
            # angular frequency per frame
            omega = 2 * 3.141592653589793 * getattr(self.config, "gap_freq_hz", 0.5) / max(1, self.config.fps)
            amp = float(getattr(self.config, "gap_amp_px", 20.0))
            # bounds to keep gap within viewport
            min_center = self.pipe_gap / 2 + 5
            max_center = self.config.window.viewport_height - self.pipe_gap / 2 - 5

            for up_pipe, low_pipe in zip(self.upper, self.lower):
                # Each pair stores its base center and random phase
                base_center = getattr(up_pipe, "gap_base_center_y", None)
                phase = getattr(up_pipe, "gap_phase", 0.0)
                if base_center is None:
                    # initialize from current positions
                    base_center = low_pipe.y - self.pipe_gap / 2
                    up_pipe.gap_base_center_y = base_center
                    low_pipe.gap_base_center_y = base_center
                    up_pipe.gap_phase = phase
                    low_pipe.gap_phase = phase

                offset = amp * __import__("math").sin(omega * self.frame + phase)
                center = min(max(base_center + offset, min_center), max_center)

                pipe_height = self.config.images.pipe[0].get_height()
                up_pipe.y = center - self.pipe_gap / 2 - pipe_height
                low_pipe.y = center + self.pipe_gap / 2

        for up_pipe, low_pipe in zip(self.upper, self.lower):
            up_pipe.tick()
            low_pipe.tick()

    def stop(self) -> None:
        for pipe in self.upper + self.lower:
            pipe.vel_x = 0

    def can_spawn_pipes(self) -> bool:
        last = self.upper[-1]
        if not last:
            return True

        return self.config.window.width - (last.x + last.w) > last.w * 2.5

    def spawn_new_pipes(self):
        # add new pipe when first pipe is about to touch left of screen
        upper, lower = self.make_random_pipes()
        # initialize oscillation params on the pair
        base_center = lower.y - self.pipe_gap / 2
        phase = random.random() * 2 * 3.141592653589793
        upper.gap_base_center_y = base_center
        lower.gap_base_center_y = base_center
        upper.gap_phase = phase
        lower.gap_phase = phase
        self.upper.append(upper)
        self.lower.append(lower)

    def remove_old_pipes(self):
        # remove first pipe if its out of the screen
        for pipe in self.upper:
            if pipe.x < -pipe.w:
                self.upper.remove(pipe)

        for pipe in self.lower:
            if pipe.x < -pipe.w:
                self.lower.remove(pipe)

    def spawn_initial_pipes(self):
        upper_1, lower_1 = self.make_random_pipes()
        upper_1.x = self.config.window.width + upper_1.w * 3
        lower_1.x = self.config.window.width + upper_1.w * 3
        base_center_1 = lower_1.y - self.pipe_gap / 2
        phase_1 = random.random() * 2 * 3.141592653589793
        upper_1.gap_base_center_y = base_center_1
        lower_1.gap_base_center_y = base_center_1
        upper_1.gap_phase = phase_1
        lower_1.gap_phase = phase_1
        self.upper.append(upper_1)
        self.lower.append(lower_1)

        upper_2, lower_2 = self.make_random_pipes()
        upper_2.x = upper_1.x + upper_1.w * 3.5
        lower_2.x = upper_1.x + upper_1.w * 3.5
        base_center_2 = lower_2.y - self.pipe_gap / 2
        phase_2 = random.random() * 2 * 3.141592653589793
        upper_2.gap_base_center_y = base_center_2
        lower_2.gap_base_center_y = base_center_2
        upper_2.gap_phase = phase_2
        lower_2.gap_phase = phase_2
        self.upper.append(upper_2)
        self.lower.append(lower_2)

    def make_random_pipes(self):
        """returns a randomly generated pipe"""
        # y of gap between upper and lower pipe
        base_y = self.config.window.viewport_height

        gap_y = random.randrange(0, int(base_y * 0.6 - self.pipe_gap))
        gap_y += int(base_y * 0.2)
        pipe_height = self.config.images.pipe[0].get_height()
        pipe_x = self.config.window.width + 10

        upper_pipe = Pipe(
            self.config,
            self.config.images.pipe[0],
            pipe_x,
            gap_y - pipe_height,
        )

        lower_pipe = Pipe(
            self.config,
            self.config.images.pipe[1],
            pipe_x,
            gap_y + self.pipe_gap,
        )

        return upper_pipe, lower_pipe
