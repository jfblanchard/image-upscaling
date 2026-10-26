"""Damage clean 512x512 crops in known ways, so every repair can be scored against the original.

Writes data/gt/<name>.png, data/in/<damage>_<name>.png and data/mask/<damage>_<name>.png (scratch masks).
"""
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"
SRC = HERE.parent / "model_comparison_2026-10" / "data" / "gt", HERE.parent / "gallery_2026-10" / "data" / "gt"

# damage: [(image, crop top-left in the 1024 ground truth)]
CASES = {
    "noise": [("kitten", (256, 300)), ("cat", (300, 150))],
    "jpeg": [("butterfly", (200, 60)), ("leaves", (300, 250))],
    "blur": [("timer", (512, 512)), ("neon", (256, 200))],
    "scratches": [("seascape", (256, 256)), ("dandelion", (256, 256))],
    "old_print": [("butterfly", (200, 60)), ("seascape", (256, 256))],
}
MOTION_LEN, MOTION_ANGLE = 15, 30  # px, degrees


def srgb_to_linear(x):
    return np.where(x <= 0.04045, x / 12.92, ((x + 0.055) / 1.055) ** 2.4)


def linear_to_srgb(x):
    return np.where(x <= 0.0031308, x * 12.92, 1.055 * np.clip(x, 0, None) ** (1 / 2.4) - 0.055)


def sensor_noise(img, rng, photons=40, read=0.004):
    """High-ISO noise: shot noise (Poisson) plus read noise, applied in linear light."""
    lin = srgb_to_linear(img)
    noisy = rng.poisson(lin * photons) / photons + rng.normal(0, read, lin.shape)
    return linear_to_srgb(np.clip(noisy, 0, 1))


def motion_kernel(length=MOTION_LEN, angle=MOTION_ANGLE):
    k = np.zeros((length, length), np.float32)
    k[length // 2, :] = 1
    rot = cv2.getRotationMatrix2D((length / 2 - 0.5, length / 2 - 0.5), angle, 1)
    k = cv2.warpAffine(k, rot, (length, length))
    return k / k.sum()


def blur(img, k):
    return cv2.filter2D(img, -1, k, borderType=cv2.BORDER_REFLECT)


def scratches(img, rng, n_lines=14, n_dust=250):
    """Thin light scratches and dark/light dust specks, as on an old scanned print. Returns image and mask."""
    h, w = img.shape[:2]
    mask = np.zeros((h, w), np.uint8)
    out = img.copy()
    for _ in range(n_lines):
        x0, y0 = rng.integers(0, w), rng.integers(0, h)
        ang, length = rng.normal(np.pi / 2, 0.25), rng.integers(80, 400)
        x1, y1 = int(x0 + length * np.cos(ang)), int(y0 + length * np.sin(ang))
        t = int(rng.integers(1, 3))
        cv2.line(out, (x0, y0), (x1, y1), (0.92, 0.92, 0.88), t, cv2.LINE_AA)
        cv2.line(mask, (x0, y0), (x1, y1), 255, t + 2)
    for _ in range(n_dust):
        x, y, r = rng.integers(0, w), rng.integers(0, h), int(rng.integers(1, 4))
        v = 0.05 if rng.random() < 0.5 else 0.95
        cv2.circle(out, (int(x), int(y)), r, (v, v, v), -1, cv2.LINE_AA)
        cv2.circle(mask, (int(x), int(y)), r + 2, 255, -1)
    return out, mask


def jpeg(img, quality):
    ok, buf = cv2.imencode(".jpg", (img * 255).round().astype(np.uint8), [cv2.IMWRITE_JPEG_QUALITY, quality])
    return cv2.imdecode(buf, cv2.IMREAD_UNCHANGED).astype(np.float32) / 255


def load(name, xy):
    for d in SRC:
        if (d / f"{name}.png").exists():
            x, y = xy
            return np.asarray(Image.open(d / f"{name}.png").convert("RGB").crop((x, y, x + 512, y + 512)), np.float32) / 255
    raise FileNotFoundError(name)


def save(img, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray((np.clip(img, 0, 1) * 255).round().astype(np.uint8)).save(path)


if __name__ == "__main__":
    k = motion_kernel()
    for damage, items in CASES.items():
        for name, xy in items:
            rng = np.random.default_rng(sum(map(ord, damage + name)))  # fixed seed per case
            gt = load(name, xy)
            save(gt, DATA / "gt" / f"{name}.png")
            if damage == "noise":
                out = sensor_noise(gt, rng)
            elif damage == "jpeg":
                out = jpeg(gt, 15)
            elif damage == "blur":
                out = blur(gt, k) + rng.normal(0, 1 / 255, gt.shape)
            elif damage == "scratches":
                out, mask = scratches(gt, rng)
                save(mask / 255.0, DATA / "mask" / f"{damage}_{name}.png")
            elif damage == "old_print":  # soft focus, grain, scratches, then saved as a low-quality JPEG
                out = cv2.GaussianBlur(gt, (0, 0), 1.2)
                out = sensor_noise(out, rng, photons=120)
                out, mask = scratches(out, rng, n_lines=8, n_dust=120)
                out = jpeg(np.clip(out, 0, 1), 40)
                save(mask / 255.0, DATA / "mask" / f"old_print_{name}.png")
            save(out, DATA / "in" / f"{damage}_{name}.png")
    np.save(DATA / "motion_kernel.npy", k)
    print("done")
