import argparse
from pathlib import Path
from PIL import Image
import numpy as np
from dftviz.ui import UIController

DEFAULT_IMAGE = Path("input/test_img_2.jpg")
DEFAULT_SIZE = 256


def load_image(path: Path, size: int):
    img = Image.open(path)
    img = img.convert("L").resize((size, size))
    data = np.array(img)
    fft = np.fft.fft2(data)
    return data, fft


def parse_args():
    p = argparse.ArgumentParser(description="2D DFT Visualisation")
    p.add_argument("--image", type=Path, default=DEFAULT_IMAGE, help="Path to input image")
    p.add_argument("--size", type=int, default=DEFAULT_SIZE, help="Square resize dimension")
    return p.parse_args()


def main():
    args = parse_args()
    if not args.image.exists():
        raise SystemExit(f"Image not found: {args.image}")

    image_data, image_fft = load_image(args.image, args.size)
    UIController(image_data, image_fft).initialize()

if __name__ == "__main__":
    main()


