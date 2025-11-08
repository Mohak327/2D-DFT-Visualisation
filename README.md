# 2D FFT / Manual Reconstruction Visualisation

Interactive visualisation of reconstructing a 2D image from its frequency components. The project computes a forward 2D FFT (`numpy.fft.fft2`) of a grayscale image, then manually (term-by-term) synthesises the spatial domain image by summing real parts of complex sinusoids. A Matplotlib UI animates this progressive reconstruction, showing:

1. Original image
2. Current sinusoid (layer) being added
3. FFT magnitude map with visited frequencies highlighted
4. Reconstructed image so far

## Why
Typical demonstrations jump straight from FFT coefficients to `ifft2`. This project instead exposes the contribution of each frequency component, helping build intuition for how low and high frequencies sum to form structure and fine detail.

## Core Concepts
- Forward transform: `fft2` for speed and correctness.
- Inverse visualisation: Manual summation of complex sinusoids (each term scaled by the FFT coefficient) to illustrate reconstruction.
- Ordering: Frequencies processed in a Chebyshev-like radial pattern to show coarse → fine emergence.
- Brightness scaling: Adaptive range per sinusoid to keep visibility of low-amplitude components.

## Features
- Smooth, timer-driven autoplay (no manual stepping required)
- Per-frequency highlight on FFT magnitude map
- Adaptive brightness scaling for current sinusoid layer
- Modular `UIController` for future extensions (pause, speed adjust, RGB support)

## Installation
Requires Python 3.10+ (earlier versions may work but not tested).

```powershell
# (Optional) create & activate a virtual environment
python -m venv .venv
.\.venv\Scripts\activate

pip install numpy matplotlib pillow
```

## Usage
```powershell
# Basic run with default image (expects input/test_img.jpg)
python .\app.py

# Custom image path & resize dimension
python .\app.py --image .\input\your_image.jpg --size 256
```

Arguments:
- `--image`: Path to an input image (will be converted to grayscale and resized square)
- `--size`: Target square dimension (power-of-two not required; larger sizes slow manual reconstruction)

## How It Works
1. Load and grayscale image → 2D array
2. Compute `fft2` → complex coefficient matrix
3. Animator builds an ordered list of frequency pairs `(x, y)` (half-spectrum with symmetry handling)
4. For each step:
   - Generate sinusoid: `coeff * exp(2πi/N (xX + yY)) / N^2` → take real part
   - Add to accumulating reconstruction buffer
   - Update layer subplot and overall image
   - Mark coefficient as visited in FFT view and move highlight
5. Continue until all selected frequencies are processed.

## Extending
Potential enhancements:
- Pause / resume controls
- Dynamic speed slider
- RGB reconstruction (per-channel manual summation, heavier CPU cost)
- Save animation frames / video export
- Frequency ordering strategies (Euclidean radius, magnitude sorting)
- Overlay of reconstruction error (difference image vs original)

## Performance Notes
- Manual synthesis is O(N^2 * K) where K = number of displayed frequency terms; large images can be slow.
- Use smaller `--size` to keep interactive frame rates.
- Brightness scaling introduces per-step min/max computation but remains minor relative to sinusoid generation.

---
Feel free to open issues or extend the controller for more interactive features.
