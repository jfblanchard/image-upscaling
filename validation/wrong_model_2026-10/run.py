"""Run each model on each wrong-model input; SeedVR2 runs separately via its CLI into data/seedvr2/."""
import sys
from pathlib import Path
from PIL import Image

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "src"))
from upscale import load_model

DATA = HERE / "data"
RUNS = {  # model: inputs
    "photo-x4": ["photo", "illustration", "text", "photo_jpeg"],
    "illustration-x4": ["photo", "illustration"],
    "swinir-classical-x4": ["photo_jpeg"],
    "swinir-x4": ["photo_jpeg"],
    "ldsr": ["illustration", "text"],
}

def run(only_input=None):
    for name, inputs in RUNS.items():
        inputs = [n for n in inputs if only_input in (None, n)]
        if not inputs:
            continue
        model = load_model(name, steps=50)
        (DATA / name).mkdir(exist_ok=True)
        for n in inputs:
            model(Image.open(DATA / "in" / f"{n}.png").convert("RGB")).save(DATA / name / f"{n}.png")
        print(name, "done", flush=True)
        del model


def figures():
    sys.path.insert(0, str(HERE.parent / "model_comparison_2026-10"))
    from run import figure

    def img(model, n):
        if model == "nearest":
            return Image.open(DATA / "in" / f"{n}.png").convert("RGB").resize((1024, 1024), Image.NEAREST)
        if model == "bicubic":
            return Image.open(DATA / "in" / f"{n}.png").convert("RGB").resize((1024, 1024), Image.BICUBIC)
        sub = "gt" if model == "original" else model
        return Image.open(DATA / sub / f"{n}.png").convert("RGB").resize((1024, 1024), Image.LANCZOS)

    specs = {
        "illustration_model_on_photo": ("photo", (300, 120, 556, 376),
            [("Input", "nearest"), ("Real-ESRGAN photo", "photo-x4"), ("Real-ESRGAN anime", "illustration-x4"), ("Ground truth", "original")]),
        "photo_models_on_illustration": ("illustration", (480, 360, 736, 616),
            [("Input", "nearest"), ("Real-ESRGAN anime", "illustration-x4"), ("Real-ESRGAN photo", "photo-x4"),
             ("LDSR", "ldsr"), ("SeedVR2", "seedvr2"), ("Ground truth", "original")]),
        "clean_model_on_jpeg": ("photo_jpeg", (300, 120, 556, 376),
            [("JPEG input", "nearest"), ("SwinIR, clean-trained", "swinir-classical-x4"),
             ("SwinIR, real-world", "swinir-x4"), ("Real-ESRGAN", "photo-x4"), ("Ground truth", "original")]),
        "generative_models_on_text": ("text", (50, 178, 450, 488),
            [("Input", "nearest"), ("Bicubic", "bicubic"), ("Real-ESRGAN", "photo-x4"), ("LDSR", "ldsr"), ("SeedVR2", "seedvr2"), ("Ground truth", "original")]),
    }
    for out, (n, box, panels) in specs.items():
        figure(box, [(label, img(m, n)) for label, m in panels]).save(ROOT / "assets" / f"wrong_{out}.jpg", quality=88)
        print(out)


if __name__ == "__main__":
    if "--figures" in sys.argv:
        figures()
    else:
        run(sys.argv[sys.argv.index("--input") + 1] if "--input" in sys.argv else None)
