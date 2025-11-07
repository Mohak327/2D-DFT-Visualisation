import math
import numpy as np
import matplotlib.pyplot as plt

# Brightness / rendering parameters (kept here for now; could externalize later)
BRIGHTNESS_FACTOR = 2
BRIGHTNESS_BIAS = 2

COLORMAP = plt.get_cmap("viridis")
VISITED_COLORMAP = plt.get_cmap("plasma")

ORDER_BY_EUCLIDEAN_DIST = lambda pos: pos[0] ** 2 + pos[1] ** 2
ORDER_BY_CHEBYSHEV_DIST = lambda pos: max(abs(pos[0]), abs(pos[1]) + (0.1 if pos[1] > 0 else 0))
SINUSOID_DRAW_ORDER = ORDER_BY_CHEBYSHEV_DIST


def compute_2d_complex_sinusoid(size, freq_x, freq_y, coefficient):
    values = np.arange(size)
    x, y = np.meshgrid(values, values)
    return coefficient * np.exp(2j * np.pi / size * (freq_x * x + freq_y * y)) / (size ** 2)


class Animator:
    """Handles progressive reconstruction of an image from its 2D FFT components."""

    def __init__(self, image_ax, layer_ax, fft_ax, fft):
        self.image_ax = image_ax
        self.layer_ax = layer_ax
        self.fft_ax = fft_ax
        self.fft = fft
        self.size = len(fft)

        self.image = np.zeros((self.size, self.size), dtype=float)
        self.image_imshow = image_ax.imshow(self.image, cmap="gray", vmin=0, vmax=255)
        self.layer_imshow = layer_ax.imshow(np.zeros((self.size, self.size), dtype=float), cmap="gray")

        self.normalized_fft_image = np.log10(np.abs(self.fft) + 1)
        self.normalized_fft_image /= np.max(self.normalized_fft_image)
        self.normalized_fft_image = np.fft.fftshift(self.normalized_fft_image)
        self.visited_fft_image = COLORMAP(self.normalized_fft_image)[:, :, :3]

        self.fft_ax.set_title("FFT")
        self.fft_imshow = fft_ax.imshow(self.image)

        middle_pos = (self.size // 2, self.size // 2)
        self.highlight_circle = plt.Circle(
            middle_pos, radius=5, fill=True,
            linewidth=1, facecolor="#ff0000aa", edgecolor="#00000055"
        )
        self.highlight_circle_small = plt.Circle(middle_pos, radius=0.5, color="black")
        self.fft_ax.add_patch(self.highlight_circle)
        self.fft_ax.add_patch(self.highlight_circle_small)

        frequency_count = math.ceil(self.size / 2)
        self.frequencies_to_draw = [
            (x, y - frequency_count) for x in range(frequency_count) for y in range(self.size)
        ]
        self.frequencies_to_draw.sort(key=SINUSOID_DRAW_ORDER)

    def reset_state(self):
        self.image[:, :] = 0.0
        self.image_imshow.set_data(self.image)
        self.layer_imshow.set_data(np.zeros((self.size, self.size), dtype=float))
        self.visited_fft_image = COLORMAP(self.normalized_fft_image)[:, :, :3]
        self.fft_imshow.set_data(self.visited_fft_image)
        middle_pos = (self.size // 2, self.size // 2)
        self.highlight_circle.set_center(middle_pos)
        self.highlight_circle_small.set_center(middle_pos)

    def get_fft_pixel_position(self, x, y):
        img_x = (x + self.size // 2) % self.size
        img_y = (y + self.size // 2) % self.size
        return img_x, img_y

    def mark_fft_pixel_as_visited(self, x, y):
        img_x, img_y = self.get_fft_pixel_position(x, y)
        pixel_color = VISITED_COLORMAP(self.normalized_fft_image[img_y, img_x])[:3]
        self.visited_fft_image[img_y, img_x] = pixel_color

    def highlight_fft_pixel(self, x, y):
        img_x, img_y = self.get_fft_pixel_position(x, y)
        self.highlight_circle.set_center((img_x, img_y))
        self.highlight_circle_small.set_center((img_x, img_y))

    def compute_value_range_for_brightness(self, x, y, coefficient):
        if x == y == 0:
            return 0, 255
        actual_brightness = np.abs(coefficient) / (self.size ** 2)
        target_rel_brightness = (actual_brightness / 255) ** (1 / BRIGHTNESS_BIAS) * BRIGHTNESS_FACTOR
        target_rel_brightness = min(target_rel_brightness, 1)
        dynamic_range = max(actual_brightness, 2 * actual_brightness / target_rel_brightness)
        vmin = -actual_brightness
        vmax = dynamic_range - actual_brightness
        return vmin, vmax

    def animate(self, step):
        if step >= len(self.frequencies_to_draw):
            return
        x, y = self.frequencies_to_draw[step]
        coefficient = self.fft[y, x] * (1 if x == 0 else 2)
        sinusoid = compute_2d_complex_sinusoid(self.size, x, y, coefficient)
        sinusoid = np.real(sinusoid)
        self.image += sinusoid
        vmin, vmax = self.compute_value_range_for_brightness(x, y, coefficient)
        self.layer_imshow.set_data(sinusoid)
        self.layer_imshow.set_clim(vmin=vmin, vmax=vmax)
        self.image_imshow.set_data(self.image)
        self.layer_ax.set_title(f"Sinusoid freq x={x} y={y}")
        self.mark_fft_pixel_as_visited(x, y)
        self.mark_fft_pixel_as_visited(-x, y)
        self.fft_imshow.set_data(self.visited_fft_image)
        self.highlight_fft_pixel(x, y)

    def draw_up_to(self, n):
        n = max(0, min(n, len(self.frequencies_to_draw)))
        self.reset_state()
        last_layer = np.zeros((self.size, self.size), dtype=float)
        vmin, vmax = 0, 255
        last_x = last_y = 0
        for i in range(n):
            x, y = self.frequencies_to_draw[i]
            coefficient = self.fft[y, x] * (1 if x == 0 else 2)
            sinusoid = compute_2d_complex_sinusoid(self.size, x, y, coefficient)
            sinusoid = np.real(sinusoid)
            self.image += sinusoid
            vmin, vmax = self.compute_value_range_for_brightness(x, y, coefficient)
            last_layer = sinusoid
            last_x, last_y = x, y
            self.mark_fft_pixel_as_visited(x, y)
            self.mark_fft_pixel_as_visited(-x, y)
            self.highlight_fft_pixel(x, y)
        self.layer_imshow.set_data(last_layer)
        self.layer_imshow.set_clim(vmin=vmin, vmax=vmax)
        self.image_imshow.set_data(self.image)
        self.layer_ax.set_title(f"Sinusoid freq x={last_x} y={last_y}" if n > 0 else "Sinusoid")
        self.fft_imshow.set_data(self.visited_fft_image)
