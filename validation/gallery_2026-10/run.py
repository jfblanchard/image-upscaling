"""Run every method on the gallery inputs and build one zoomed comparison strip per image.
SeedVR2 runs separately via its CLI into data/seedvr2/ (see setup.md)."""
import sys
from pathlib import Path
from PIL import Image

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(HERE.parent / "model_comparison_2026-10"))
from upscale import load_model
from run import figure

DATA = HERE / "data"
NAMES = ["kitten", "dandelion", "neon", "seascape", "whale"]
MODELS = ["photo-x4", "illustration-x4", "swinir-x4", "ldsr"]
ZOOM = {"kitten": (200, 320, 840, 960), "dandelion": (256, 256, 768, 768), "neon": (256, 200, 768, 712),
        "seascape": (252, 250, 764, 762), "whale": (400, 80, 912, 592)}
PANELS = [("Input", "nearest"), ("Lanczos", "lanczos"), ("Real-ESRGAN", "photo-x4"), ("Real-ESRGAN anime", "illustration-x4"),
          ("SwinIR", "swinir-x4"), ("LDSR", "ldsr"), ("SeedVR2", "seedvr2"), ("Ground truth", "gt")]


def run():
    inputs = {n: Image.open(DATA / "in" / f"{n}.png").convert("RGB") for n in NAMES}
    for name in MODELS:
        model = load_model(name, steps=50)
        (DATA / name).mkdir(exist_ok=True)
        for n, im in inputs.items():
            model(im).save(DATA / name / f"{n}.png")
        print(name, "done", flush=True)
        del model


def figures():
    for n in NAMES:
        small = Image.open(DATA / "in" / f"{n}.png").convert("RGB")
        panels = []
        for label, key in PANELS:
            if key == "nearest":
                im = small.resize((1024, 1024), Image.NEAREST)
            elif key == "lanczos":
                im = small.resize((1024, 1024), Image.LANCZOS)
            else:
                im = Image.open(DATA / key / f"{n}.png").convert("RGB").resize((1024, 1024), Image.LANCZOS)
            panels.append((label, im))
        figure(ZOOM[n], panels, S=288).save(ROOT / "assets" / f"gallery_{n}.jpg", quality=88)
        print(n)


if __name__ == "__main__":
    figures() if "--figures" in sys.argv else run()
