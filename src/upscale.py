"""Local batch upscaler: classical interpolation, Real-ESRGAN, SwinIR and LDSR.

  python src/upscale.py photo.jpg                          # Real-ESRGAN photo 4x
  python src/upscale.py shots/ -o out/ -m photo-x2         # whole folder
  python src/upscale.py art.png -m illustration-x4
  python src/upscale.py small.png -m ldsr --steps 50       # diffusion, slow
  python src/upscale.py photo.jpg -m lanczos               # classical baseline, CPU

Outputs are written as <name>_<model>.png. Existing outputs are skipped.
"""

import argparse
import sys
import time
import urllib.request
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageFilter, ImageOps

ROOT = Path(__file__).resolve().parent.parent
WEIGHTS_DIR = ROOT / "weights"
IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff"}

WEIGHTS = {
    # name: url -- official Real-ESRGAN release weights, BSD-3-Clause
    "photo-x2": "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.1/RealESRGAN_x2plus.pth",
    "photo-x4": "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth",
    "illustration-x4": "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.2.4/RealESRGAN_x4plus_anime_6B.pth",
    # SwinIR-L real-world 4x (GAN), Apache-2.0
    "swinir-x4": "https://github.com/JingyunLiang/SwinIR/releases/download/v0.0/003_realSR_BSRGAN_DFOWMFC_s64w8_SwinIR-L_x4_GAN.pth",
    # SwinIR-M classical 4x: trained on clean bicubic downscaling only, Apache-2.0
    "swinir-classical-x4": "https://github.com/JingyunLiang/SwinIR/releases/download/v0.0/001_classicalSR_DF2K_s64w8_SwinIR-M_x4.pth",
}
LDSR_REPO = "CompVis/ldm-super-resolution-4x-openimages"  # Apache-2.0
CLASSICAL = {
    "nearest": Image.NEAREST,
    "bilinear": Image.BILINEAR,
    "bicubic": Image.BICUBIC,
    "lanczos": Image.LANCZOS,
    "bicubic-sharp": Image.BICUBIC,  # bicubic followed by an unsharp mask
}
MODELS = [*CLASSICAL, *WEIGHTS, "ldsr"]


class Classical:
    """Interpolation baselines, CPU only."""

    def __init__(self, name, scale=4):
        self.name, self.scale = name, scale

    def __call__(self, img):
        out = img.resize((img.width * self.scale, img.height * self.scale), CLASSICAL[self.name])
        if self.name == "bicubic-sharp":
            out = out.filter(ImageFilter.UnsharpMask(radius=self.scale, percent=120, threshold=2))
        return out


class SpandrelModel:
    def __init__(self, name, tile=512, pad=16):
        from spandrel import ModelLoader

        path = WEIGHTS_DIR / f"{name}.pth"
        if not path.exists():
            WEIGHTS_DIR.mkdir(exist_ok=True)
            print(f"downloading {name} weights")
            urllib.request.urlretrieve(WEIGHTS[name], path)
        m = ModelLoader().load_from_file(path).eval().cuda()
        if name.startswith("swinir"):
            tile = 256  # window attention is memory-hungry
        self.model = m.half() if m.supports_half else m
        self.dtype = next(self.model.model.parameters()).dtype
        self.scale, self.tile, self.pad = m.scale, tile, pad

    @torch.inference_mode()
    def __call__(self, img):
        """Tiled inference: run on overlapping tiles, keep each tile's centre."""
        s, T, P = self.scale, self.tile, self.pad
        x = torch.from_numpy(np.array(img)).permute(2, 0, 1)[None].to("cuda", self.dtype) / 255
        _, _, h, w = x.shape
        out = torch.empty((3, h * s, w * s), dtype=torch.uint8, device="cuda")
        for y0 in range(0, h, T):
            for x0 in range(0, w, T):
                y1, x1 = min(y0 + T, h), min(x0 + T, w)
                py0, px0, py1, px1 = max(y0 - P, 0), max(x0 - P, 0), min(y1 + P, h), min(x1 + P, w)
                t = self.model(x[:, :, py0:py1, px0:px1])
                t = t[0, :, (y0 - py0) * s:(y1 - py0) * s, (x0 - px0) * s:(x1 - px0) * s]
                out[:, y0 * s:y1 * s, x0 * s:x1 * s] = t.float().clamp(0, 1).mul(255).round().byte()
        return Image.fromarray(out.permute(1, 2, 0).cpu().numpy())


class LDSR:
    """Latent diffusion 4x. Large inputs run as overlapping tiles blended with a linear ramp."""

    scale = 4

    def __init__(self, steps=100, tile=256, overlap=32, seed=0):
        from diffusers import LDMSuperResolutionPipeline

        self.pipe = LDMSuperResolutionPipeline.from_pretrained(LDSR_REPO, torch_dtype=torch.float16, variant="fp16", use_safetensors=True).to("cuda")
        self.pipe.set_progress_bar_config(disable=True)
        self.steps, self.tile, self.overlap, self.seed = steps, tile, overlap, seed

    def _run(self, img):
        g = torch.Generator("cuda").manual_seed(self.seed)
        return self.pipe(img, num_inference_steps=self.steps, eta=1, generator=g).images[0]

    @torch.inference_mode()
    def __call__(self, img):
        w, h = img.size
        T, O, s = self.tile, self.overlap, self.scale
        if max(w, h) <= T:
            return self._run(img)

        def starts(n):
            if n <= T:
                return [0]
            step = T - O
            pts = list(range(0, n - T, step)) + [n - T]
            return sorted(set(pts))

        acc = np.zeros((h * s, w * s, 3), np.float32)
        wsum = np.zeros((h * s, w * s, 1), np.float32)
        for y0 in starts(h):
            for x0 in starts(w):
                tw, th = min(T, w - x0), min(T, h - y0)
                tile = np.asarray(self._run(img.crop((x0, y0, x0 + tw, y0 + th))), np.float32)
                ramp = lambda n: np.clip(np.minimum(np.arange(n) + 1, n - np.arange(n)) / (O * s), 0, 1)
                wt = (ramp(th * s)[:, None] * ramp(tw * s)[None, :])[..., None] + 1e-6
                acc[y0 * s:(y0 + th) * s, x0 * s:(x0 + tw) * s] += tile * wt
                wsum[y0 * s:(y0 + th) * s, x0 * s:(x0 + tw) * s] += wt
        return Image.fromarray((acc / wsum).clip(0, 255).round().astype(np.uint8))


def load_model(name, steps=100):
    if name in CLASSICAL:
        return Classical(name)
    return LDSR(steps=steps) if name == "ldsr" else SpandrelModel(name)


def collect(paths):
    for p in map(Path, paths):
        if p.is_dir():
            yield from sorted(f for f in p.iterdir() if f.suffix.lower() in IMAGE_EXTS)
        else:
            yield p


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("inputs", nargs="+", help="image files or folders")
    ap.add_argument("-m", "--model", choices=MODELS, default="photo-x4")
    ap.add_argument("-o", "--out", type=Path, help="output folder (default: next to each input)")
    ap.add_argument("--steps", type=int, default=100, help="LDSR diffusion steps (default 100)")
    ap.add_argument("--format", choices=["png", "jpg"], default="png")
    args = ap.parse_args(argv)

    files = list(collect(args.inputs))
    if not files:
        sys.exit("no images found")
    model = load_model(args.model, args.steps)

    for f in files:
        dest_dir = args.out or f.parent
        dest_dir.mkdir(parents=True, exist_ok=True)
        dest = dest_dir / f"{f.stem}_{args.model}.{args.format}"
        if dest.exists():
            print(f"skip {dest.name} (exists)")
            continue
        img = ImageOps.exif_transpose(Image.open(f)).convert("RGB")
        t0 = time.perf_counter()
        out = model(img)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        dt = time.perf_counter() - t0
        out.save(dest, quality=95) if args.format == "jpg" else out.save(dest)
        print(f"{f.name} {img.width}x{img.height} -> {out.width}x{out.height}  {dt:.1f}s  {dest}")


if __name__ == "__main__":
    main()
