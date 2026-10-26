"""Repair each damaged input with classical and learned methods, score against ground truth, build figures.

SeedVR2 runs separately via its CLI (see setup.md); its outputs are read from data/seedvr2/.
  python run.py             # run all methods, then figures
  python run.py --figures   # figures and scores only
"""
import csv
import importlib.util
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image, ImageFilter

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "src"))
from make_inputs import CASES as CROPS  # noqa: E402
from upscale import WEIGHTS_DIR, load_model  # noqa: E402

_spec = importlib.util.spec_from_file_location("comparison", HERE.parent / "model_comparison_2026-10" / "run.py")
_comparison = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_comparison)
figure, psnr = _comparison.figure, _comparison.psnr

DATA, ASSETS = HERE / "data", ROOT / "assets"
CASES = {d: [n for n, _ in v] for d, v in CROPS.items()}
K = np.load(DATA / "motion_kernel.npy")

# damage: [(label, method)], in figure order
METHODS = {
    "noise": [("Non-local means", "nlmeans"), ("SCUNet", "denoise"), ("SCUNet GAN", "denoise-gan"), ("SeedVR2", "seedvr2")],
    "jpeg": [("Bilateral filter", "bilateral"), ("FBCNN", "dejpeg"), ("SCUNet", "denoise"), ("SeedVR2", "seedvr2")],
    "blur": [("Unsharp mask", "unsharp"), ("Wiener, known blur", "wiener"), ("FFTformer", "deblur"), ("SeedVR2", "seedvr2")],
    "scratches": [("Telea inpaint", "telea"), ("LaMa", "inpaint"), ("SeedVR2", "seedvr2")],
    "old_print": [("Classical chain", "classical-chain"), ("SCUNet + LaMa", "learned-chain"), ("SeedVR2", "seedvr2")],
}
# damage: (image shown in the README figure, zoom box in the 512 crop)
FIGURE = {"noise": ("kitten", (0, 0, 256, 256)), "jpeg": ("butterfly", (128, 64, 384, 320)),
          "blur": ("neon", (128, 128, 384, 384)), "scratches": ("seascape", (128, 128, 384, 384)),
          "old_print": ("butterfly", (128, 64, 384, 320))}


def u8(a):
    return np.asarray(a, np.uint8)


def nlmeans(img):
    return cv2.fastNlMeansDenoisingColored(u8(img), None, 10, 10, 7, 21)


def bilateral(img):
    return cv2.bilateralFilter(u8(img), 9, 40, 5)


def unsharp(img):
    return np.asarray(Image.fromarray(u8(img)).filter(ImageFilter.UnsharpMask(4, 150, 2)))


def wiener(img, k=K, nsr=0.02):  # noise-to-signal ratio, tuned on these images
    """Frequency-domain Wiener deconvolution, given the exact blur kernel (a best case no real photo offers)."""
    x = np.asarray(img, np.float32) / 255
    pad = k.shape[0]
    x = np.pad(x, ((pad, pad), (pad, pad), (0, 0)), mode="reflect")
    H = np.fft.fft2(k, s=x.shape[:2])
    G = np.conj(H) / (np.abs(H) ** 2 + nsr)
    out = np.stack([np.real(np.fft.ifft2(np.fft.fft2(x[..., c]) * G)) for c in range(3)], -1)
    c = k.shape[0] // 2
    out = np.roll(out, (c, c), (0, 1))[pad:-pad, pad:-pad]  # undo the kernel's centre offset
    return (np.clip(out, 0, 1) * 255).round().astype(np.uint8)


def telea(img, mask):
    return cv2.inpaint(u8(img), mask, 5, cv2.INPAINT_TELEA)


class LaMa:
    def __init__(self):
        from spandrel import ModelLoader

        self.model = ModelLoader().load_from_file(WEIGHTS_DIR / "inpaint.pt").eval().cuda()

    @torch.inference_mode()
    def __call__(self, img, mask):
        x = torch.from_numpy(np.asarray(img, np.float32) / 255).permute(2, 0, 1)[None].cuda()
        m = torch.from_numpy((mask > 127).astype(np.float32))[None, None].cuda()
        y = self.model(x, m)[0].permute(1, 2, 0).clamp(0, 1).mul(255).round().byte().cpu().numpy()
        return y


def run():
    imgs = {p.stem: Image.open(p).convert("RGB") for p in sorted((DATA / "in").glob("*.png"))}
    masks = {p.stem: cv2.imread(str(p), cv2.IMREAD_GRAYSCALE) for p in (DATA / "mask").glob("*.png")}
    rows = []

    def save(key, case, out, dt):
        (DATA / key).mkdir(exist_ok=True)
        Image.fromarray(np.asarray(out, np.uint8)).save(DATA / key / f"{case}.png")
        rows.append((key, case, dt))

    def timed(fn, *a):
        t0 = time.perf_counter(); out = fn(*a)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        return out, time.perf_counter() - t0

    classical = {"nlmeans": nlmeans, "bilateral": bilateral, "unsharp": unsharp, "wiener": wiener}
    for damage, methods in METHODS.items():
        for _, key in methods:
            for name in CASES[damage]:
                case = f"{damage}_{name}"
                if key in classical:
                    save(key, case, *timed(classical[key], imgs[case]))
                elif key == "telea":
                    save(key, case, *timed(telea, imgs[case], masks[case]))
                elif key == "classical-chain":
                    save(key, case, *timed(lambda i, m: telea(nlmeans(i), m), imgs[case], masks[case]))

    lama = LaMa()
    lama(imgs["scratches_seascape"], masks["scratches_seascape"])  # warm-up
    for name in CASES["scratches"]:
        case = f"scratches_{name}"
        save("inpaint", case, *timed(lama, imgs[case], masks[case]))

    for key in ("denoise", "denoise-gan", "dejpeg", "deblur"):
        model = load_model(key)
        model(imgs["noise_cat"].crop((0, 0, 64, 64)))  # warm-up
        for damage, methods in METHODS.items():
            if key in [k for _, k in methods]:
                for name in CASES[damage]:
                    case = f"{damage}_{name}"
                    save(key, case, *timed(model, imgs[case]))
        if key == "denoise":  # learned chain for the old print: denoise and deblock, then fill scratches
            for name in CASES["old_print"]:
                case = f"old_print_{name}"
                clean, t1 = timed(model, imgs[case])
                filled, t2 = timed(lama, clean, masks[case])
                save("learned-chain", case, filled, t1 + t2)
        del model; torch.cuda.empty_cache()

    with open(DATA / "timings.csv", "w", newline="") as f:
        csv.writer(f).writerows([["method", "case", "seconds"], *[(k, c, f"{t:.3f}") for k, c, t in rows]])


def figures():
    scores = []
    for damage, methods in METHODS.items():
        for name in CASES[damage]:
            case = f"{damage}_{name}"
            gt = Image.open(DATA / "gt" / f"{name}.png").convert("RGB")
            dmg = Image.open(DATA / "in" / f"{case}.png").convert("RGB")
            scores.append(("input", case, psnr(dmg, gt)))
            panels = [("Damaged", dmg)]
            for label, key in methods:
                p = DATA / key / f"{case}.png"
                if not p.exists():
                    continue
                out = Image.open(p).convert("RGB").resize(gt.size, Image.LANCZOS)
                scores.append((key, case, psnr(out, gt)))
                panels.append((label, out))
            panels.append(("Ground truth", gt))
            if name == FIGURE[damage][0]:
                figure(FIGURE[damage][1], panels, S=320).save(ASSETS / f"restore_{damage}.jpg", quality=88)
    with open(DATA / "psnr.csv", "w", newline="") as f:
        csv.writer(f).writerows([["method", "case", "psnr_db"], *[(k, c, f"{p:.2f}") for k, c, p in scores]])
    for k, c, p in scores:
        print(f"{c:24s} {k:16s} {p:6.2f}")


if __name__ == "__main__":
    if "--figures" not in sys.argv:
        run()
    figures()
