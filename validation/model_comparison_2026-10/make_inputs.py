"""Crop a 1024x1024 region of each photo (ground truth) and downscale 4x to 256x256 (input)."""
from pathlib import Path
import numpy as np
from PIL import Image, ImageOps

P = Path(__file__).resolve().parent / "data/src"  # my photos, not included in the repo
SOURCES = {  # name: (photo, top-left of 1024 px crop)
    "timer": (P / "timer.jpg", (2100, 1300)),
    "butterfly": (P / "butterfly.jpg", (1792, 1280)),
    "cat": (P / "cat.jpg", (1750, 400)),
    "leaves": (P / "leaves.jpg", (2200, 700)),
}
OUT = Path(__file__).resolve().parent / "data"
N = 1024

for name, (src, (x, y)) in SOURCES.items():
    im = ImageOps.exif_transpose(Image.open(src)).convert("RGB")
    gt = im.crop((x, y, x + N, y + N))
    gt.save(OUT / "gt" / f"{name}.png")
    gt.resize((N // 4, N // 4), Image.BICUBIC).save(OUT / "in" / f"{name}.png")
    print(name, im.size, "crop at", (x, y))
