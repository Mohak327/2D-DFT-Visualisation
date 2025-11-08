import numpy as np
import matplotlib.pyplot as plt
from .animator import Animator

class UIController:
    def __init__(
        self,
        image_data: np.ndarray,
        image_fft: np.ndarray,
        *,
        figsize=(14, 3.5),
        autoplay: bool = True,
        window_title: str | None = None,
    ):
        self.image_data = image_data
        self.image_fft = image_fft
        self.figsize = figsize
        self.autoplay = autoplay
        self.window_title = window_title or "2D FFT Visualisation"

        # Runtime state
        self.fig = None
        self.ax = None
        self.animator: Animator | None = None
        self._current_step = 0
        self._burst_end = 0
        self._timer = None
        self._bursting = False

    def initialize(self):
        """Entry point to build UI and begin autoplay (if enabled)."""
        self._build_figure()
        self._init_animator()
        self._set_titles()
        if self.autoplay:
            self.run_to_completion()
        plt.show()

    def run_to_completion(self):
        """Schedule a single smooth burst to process all remaining components."""
        self._schedule_burst(len(self.animator.frequencies_to_draw))

    # ---- Internal helpers ----
    def _build_figure(self):
        self.fig, self.ax = plt.subplots(
            nrows=1, ncols=4, figsize=self.figsize, constrained_layout=True
        )
        self.ax[0].imshow(self.image_data, cmap="gray", vmin=0, vmax=255)
        self.ax[0].set_title("Original")
        if hasattr(self.fig.canvas, "manager") and hasattr(self.fig.canvas.manager, "set_window_title"):
            self.fig.canvas.manager.set_window_title(self.window_title)

    def _init_animator(self):
        self.animator = Animator(
            image_ax=self.ax[3],
            layer_ax=self.ax[1],
            fft_ax=self.ax[2],
            fft=self.image_fft,
        )

    def _set_titles(self):
        self.fig.suptitle(
            f"Autoplay: {self.animator.steps_per_tick} comps/tick • {self.animator.tick_interval_ms}ms"
        )

    def _schedule_burst(self, total_steps: int):
        total_steps = max(1, int(total_steps))
        self._burst_end = min(self._current_step + total_steps, len(self.animator.frequencies_to_draw))
        if not self._bursting:
            self._timer = self.fig.canvas.new_timer(interval=max(1, int(self.animator.tick_interval_ms)))
            self._timer.add_callback(self._tick)
            self._bursting = True
            self._timer.start()

    def _tick(self):
        remaining = self._burst_end - self._current_step
        to_do = min(self.animator.steps_per_tick, remaining)
        for _ in range(to_do):
            if self._current_step >= self._burst_end:
                break
            self.animator.animate(self._current_step)
            self._current_step += 1
        self.fig.canvas.draw_idle()
        if self._current_step >= self._burst_end or self._current_step >= len(self.animator.frequencies_to_draw):
            if self._timer is not None:
                self._timer.stop()
            self._bursting = False

