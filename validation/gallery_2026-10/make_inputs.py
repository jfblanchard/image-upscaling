"""Gallery inputs from Creative Commons / public-domain images (see CREDITS.md).
Each 1024x1024 crop is the ground truth; 4x bicubic downscale gives the 256x256 input."""
from pathlib import Path
from PIL import Image

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"
N = 1024
CROPS = {  # name: (source file, top-left of crop in the downloaded image)
    "kitten": ("src_1.jpg", (1680, 830)),
    "dandelion": ("src_2.jpg", (512, 300)),
    "whale": ("src_3.jpg", (300, 150)),
    "neon": ("src_4.jpg", (512, 512)),
    "seascape": ("src_5.jpg", (1760, 880)),
}

if __name__ == "__main__":
    for d in ("gt", "in"):
        (DATA / d).mkdir(exist_ok=True)
    for name, (src, (x, y)) in CROPS.items():
        gt = Image.open(DATA / "src" / src).convert("RGB").crop((x, y, x + N, y + N))
        gt.save(DATA / "gt" / f"{name}.png")
        gt.resize((N // 4, N // 4), Image.BICUBIC).save(DATA / "in" / f"{name}.png")
    print("ok")
