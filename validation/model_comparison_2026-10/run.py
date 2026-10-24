"""Upscale 256x256 inputs 4x with each method, time them, and build README comparison figures.

SeedVR2 runs separately via its CLI (see setup.md); its outputs are read from data/seedvr2/.
"""
import csv
import sys
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "src"))
from upscale import load_model

DATA, ASSETS = HERE / "data", ROOT / "assets"
NAMES = ["butterfly", "cat", "leaves", "timer"]
ZOOM = {"butterfly": (300, 120, 556, 376), "cat": (380, 180, 636, 436), "leaves": (380, 300, 636, 556), "timer": (768, 768, 1024, 1024)}


def psnr(a, b):
    mse = np.mean((np.asarray(a, np.float64) - np.asarray(b, np.float64)) ** 2)
    return 10 * np.log10(255 ** 2 / mse)


GPU_MODELS = [("realesrgan", "photo-x4", 0), ("swinir", "swinir-x4", 0), ("ldsr-50", "ldsr", 50), ("ldsr-100", "ldsr", 100)]


def run_models(inputs, only=None):
    rows = []
    for key, name, steps in GPU_MODELS:
        if only and key not in only:
            continue
        model = load_model(name, steps or 100)
        model(inputs[NAMES[0]].crop((0, 0, 64, 64)))  # warm-up
        (DATA / key).mkdir(exist_ok=True)
        for n in NAMES:
            torch.cuda.synchronize(); t0 = time.perf_counter()
            out = model(inputs[n]); torch.cuda.synchronize()
            rows.append((key, n, time.perf_counter() - t0))
            out.save(DATA / key / f"{n}.png")
        del model; torch.cuda.empty_cache()
    return rows


def label_font(size=18):
    for f in ("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", "/usr/share/fonts/TTF/DejaVuSans.ttf"):
        if Path(f).exists():
            return ImageFont.truetype(f, size)
    return ImageFont.load_default()


def figure(box, panels, S=384):
    """One row: each panel shows the same zoomed region, S px wide."""
    H = 30
    ph = round(S * (box[3] - box[1]) / (box[2] - box[0]))
    fig = Image.new("RGB", (S * len(panels), ph + H), "white")
    d, font = ImageDraw.Draw(fig), label_font()
    for i, (label, img) in enumerate(panels):
        fig.paste(img.crop(box).resize((S, ph), Image.LANCZOS), (i * S, H))
        d.text((i * S + S // 2, H // 2), label, fill="black", font=font, anchor="mm")
    return fig


if __name__ == "__main__":
    inputs = {n: Image.open(DATA / "in" / f"{n}.png").convert("RGB") for n in NAMES}
    gts = {n: Image.open(DATA / "gt" / f"{n}.png").convert("RGB") for n in NAMES}
    only = sys.argv[sys.argv.index("--only") + 1].split(",") if "--only" in sys.argv else None
    rows = [] if "--figures-only" in sys.argv else run_models(inputs, only)

    ASSETS.mkdir(exist_ok=True)
    methods = [("Input", "nearest"), ("Bicubic", "bicubic"), ("Real-ESRGAN", "realesrgan"), ("SwinIR", "swinir"), ("LDSR", "ldsr-100"), ("SeedVR2", "seedvr2")]
    classical = [("Nearest", "nearest"), ("Bilinear", "bilinear"), ("Bicubic", "bicubic"),
                 ("Lanczos", "lanczos"), ("Bicubic + sharpen", "bicubic-sharp")]
    scores, ctimes = [], []
    for n in NAMES:
        outs = {}
        for _, key in classical:
            m = load_model(key)
            t0 = time.perf_counter()
            for _ in range(20):
                outs[key] = m(inputs[n])
            ctimes.append((key, n, (time.perf_counter() - t0) / 20))
        for key in ("realesrgan", "swinir", "ldsr-50", "ldsr-100", "seedvr2"):
            p = DATA / key / f"{n}.png"
            if p.exists():
                outs[key] = Image.open(p).convert("RGB").resize((1024, 1024), Image.LANCZOS)
        for key, img in outs.items():
            scores.append((key, n, psnr(img, gts[n])))
        panels = [(label, outs[k]) for label, k in methods if k in outs] + [("Ground truth", gts[n])]
        figure(ZOOM[n], panels).save(ASSETS / f"compare_{n}.jpg", quality=88)
        if n == "butterfly":
            panels = [(label, outs[k]) for label, k in classical] + [("Ground truth", gts[n])]
            figure(ZOOM[n], panels).save(ASSETS / f"classical_{n}.jpg", quality=88)

    if rows:  # merge: replace rows for the methods just run, keep the rest
        path = DATA / "timings.csv"
        ran = {k for k, _, _ in rows}
        old = [r for r in csv.reader(open(path))][1:] if path.exists() else []
        keep = [r for r in old if r[0] not in ran]
        with open(path, "w", newline="") as f:
            csv.writer(f).writerows([["method", "image", "seconds"], *keep, *[(k, n, f"{t:.2f}") for k, n, t in rows]])
    with open(DATA / "timings_classical.csv", "w", newline="") as f:
        csv.writer(f).writerows([["method", "image", "seconds"], *[(k, n, f"{t:.5f}") for k, n, t in ctimes]])
    with open(DATA / "psnr.csv", "w", newline="") as f:
        csv.writer(f).writerows([["method", "image", "psnr_db"], *[(k, n, f"{p:.2f}") for k, n, p in scores]])
    for k, n, t in rows:
        print(f"{k:12s} {n:10s} {t:6.2f}s")
