"""Time Real-ESRGAN and LDSR on the local RTX 3060 across input sizes."""
import csv, sys, time
from pathlib import Path
import torch
from PIL import Image

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / "src"))
from upscale import load_model

SRC = Path(__file__).resolve().parent / "data/src/sunset.jpg"  # my photo, not included in the repo
DATA = HERE / "data"

def inputs():
    full = Image.open(SRC).convert("RGB")
    full.thumbnail((2000, 2000), Image.LANCZOS)          # long side 2000
    w, h = full.size
    out = {"2000xN": full}
    for n in (128, 256, 512):
        crop = full.crop((w // 2 - 256, h // 2 - 256, w // 2 + 256, h // 2 + 256)).resize((n, n), Image.LANCZOS)
        out[f"{n}x{n}"] = crop
    for k, im in out.items():
        im.save(DATA / f"in_{k}.png")
    return out

def bench(name, cases, steps=100):
    model = load_model(name, steps)
    tag = name if name != "ldsr" else f"ldsr-{steps}"
    model(cases[0][1].crop((0, 0, 64, 64)))  # warm-up
    rows = []
    for key, img in cases:
        torch.cuda.reset_peak_memory_stats(); torch.cuda.synchronize()
        t0 = time.perf_counter(); out = model(img); torch.cuda.synchronize()
        dt = time.perf_counter() - t0
        peak = torch.cuda.max_memory_allocated() / 2**30
        out.save(DATA / f"out_{tag}_{key}.png")
        rows.append([tag, key, f"{img.width}x{img.height}", f"{out.width}x{out.height}", f"{dt:.2f}", f"{peak:.2f}"])
        print(*rows[-1], flush=True)
    del model; torch.cuda.empty_cache()
    return rows

if __name__ == "__main__":
    ins = inputs()
    rows = []
    rows += bench("photo-x4", [(k, ins[k]) for k in ("128x128", "256x256", "512x512", "2000xN")])
    rows += bench("photo-x2", [(k, ins[k]) for k in ("512x512", "2000xN")])
    rows += bench("illustration-x4", [(k, ins[k]) for k in ("512x512", "2000xN")])
    for steps in (100, 50, 25):
        rows += bench("ldsr", [(k, ins[k]) for k in ("128x128", "256x256")], steps)
    rows += bench("ldsr", [("512x512", ins["512x512"])], 50)
    with open(DATA / "timings.csv", "w", newline="") as f:
        csv.writer(f).writerows([["model", "input", "in_size", "out_size", "seconds", "peak_vram_gib"], *rows])
