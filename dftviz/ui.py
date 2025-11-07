import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from .animator import Animator


def start_with_slider(image_data: np.ndarray, image_fft: np.ndarray, window_title: str = "2D FFT Visualisation"):
    """Start the UI where a slider controls the number of components used in reconstruction."""
    fig, ax = plt.subplots(nrows=2, ncols=2)

    ax[0, 0].imshow(image_data, cmap="gray", vmin=0, vmax=255)
    ax[0, 0].set_title("Original Image")

    animator = Animator(
        image_ax=ax[1, 0],
        layer_ax=ax[1, 1],
        fft_ax=ax[0, 1],
        fft=image_fft,
    )

    fig.suptitle("Use the slider to control reconstruction progress", y=0.95)
    if hasattr(fig.canvas, "manager") and hasattr(fig.canvas.manager, "set_window_title"):
        fig.canvas.manager.set_window_title(window_title)

    # Add spacing between subplots and keep room for the slider
    plt.subplots_adjust(left=0.08, right=0.98, bottom=0.18, top=0.88, wspace=0.35, hspace=0.40)
    slider_ax = fig.add_axes([0.15, 0.04, 0.7, 0.03])
    progress_slider = Slider(
        ax=slider_ax,
        label="Progress (components)",
        valmin=0,
        valmax=len(animator.frequencies_to_draw),
        valinit=0,
        valstep=1,
    )

    def on_progress_change(val):
        target = int(val)
        animator.draw_up_to(target)
        fig.canvas.draw_idle()

    progress_slider.on_changed(on_progress_change)

    animator.reset_state()
    plt.show()


def start_with_keys(image_data: np.ndarray, image_fft: np.ndarray, window_title: str = "2D FFT Visualisation"):
    """Start the UI where click / space / right arrow advance step by step."""
    fig, ax = plt.subplots(nrows=2, ncols=2)

    ax[0, 0].imshow(image_data, cmap="gray", vmin=0, vmax=255)
    ax[0, 0].set_title("Original Image")

    animator = Animator(
        image_ax=ax[1, 0],
        layer_ax=ax[1, 1],
        fft_ax=ax[0, 1],
        fft=image_fft,
    )

    current_step = 0

    def draw_next_step():
        nonlocal current_step
        animator.animate(current_step)
        current_step += 1
        plt.draw()

    def on_click(event):
        if event.button == 1:
            draw_next_step()

    def on_key(event):
        if event.key == "right" or event.key == " ":
            draw_next_step()

    fig.canvas.mpl_connect("button_press_event", on_click)
    fig.canvas.mpl_connect("key_press_event", on_key)

    fig.suptitle("Click / Space / Right Arrow for next step", y=0.95)
    # Add spacing between subplots to prevent overlaps
    plt.subplots_adjust(left=0.08, right=0.98, bottom=0.08, top=0.88, wspace=0.35, hspace=0.40)
    if hasattr(fig.canvas, "manager") and hasattr(fig.canvas.manager, "set_window_title"):
        fig.canvas.manager.set_window_title(window_title)

    draw_next_step()
    plt.show()
