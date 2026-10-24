"""Inputs for the wrong-model experiments. Each has a 1024x1024 ground truth and a 256x256 input.

- illustration: flat-color line drawing, drawn here (supersampled 4x for clean edges)
- text: small printed text, drawn here
- photo_jpeg: the butterfly photo crop, downscaled then saved as JPEG quality 25
"""
import io
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"
PHOTO_GT = HERE.parent / "model_comparison_2026-10/data/gt/butterfly.png"
FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
N, SS = 1024, 4


def illustration():
    W = N * SS
    im = Image.new("RGB", (W, W), (150, 205, 240))
    d = ImageDraw.Draw(im)
    k = lambda *v: [x * SS for x in v]
    line = 6 * SS
    d.ellipse(k(700, 90, 900, 290), fill=(255, 210, 60), outline="black", width=line)            # sun
    for cx, cy in [(220, 170), (330, 150), (430, 175)]:                                            # cloud
        d.ellipse(k(cx - 90, cy - 60, cx + 90, cy + 60), fill="white", outline="black", width=line)
    d.ellipse(k(-300, 560, 700, 1400), fill=(110, 190, 90), outline="black", width=line)          # hills
    d.ellipse(k(380, 620, 1400, 1500), fill=(80, 165, 70), outline="black", width=line)
    d.rectangle(k(520, 520, 820, 780), fill=(235, 110, 90), outline="black", width=line)          # house
    d.polygon(k(490, 530, 670, 380, 850, 530), fill=(120, 70, 50), outline="black", width=line)
    d.rectangle(k(630, 640, 710, 780), fill=(90, 60, 40), outline="black", width=line)            # door
    d.rectangle(k(550, 580, 610, 640), fill=(250, 240, 170), outline="black", width=line)         # windows
    d.rectangle(k(740, 580, 800, 640), fill=(250, 240, 170), outline="black", width=line)
    return im.resize((N, N), Image.LANCZOS)


def text():
    im = Image.new("RGB", (N, N), (250, 248, 240))
    d = ImageDraw.Draw(im)
    lines = ["IMAGE UPSCALING", "Invoice 2026-0417", "Total due: $1,284.50",
             "The quick brown fox jumps", "over the lazy dog.", "0123456789 ABCDEFGHIJ"]
    y = 90
    for i, t in enumerate(lines):
        size = 52 if i == 0 else 24  # 24 px becomes 6 px in the 256 px input
        d.text((70, y), t, fill=(20, 20, 30), font=ImageFont.truetype(FONT, size))
        y += size + 40
    return im


def photo_jpeg():
    gt = Image.open(PHOTO_GT).convert("RGB")
    buf = io.BytesIO()
    gt.resize((N // 4, N // 4), Image.BICUBIC).save(buf, "JPEG", quality=25)
    return gt, Image.open(io.BytesIO(buf.getvalue())).convert("RGB")


if __name__ == "__main__":
    for name, gt in [("illustration", illustration()), ("text", text())]:
        gt.save(DATA / "gt" / f"{name}.png")
        gt.resize((N // 4, N // 4), Image.BICUBIC).save(DATA / "in" / f"{name}.png")
    gt, small = photo_jpeg()
    gt.save(DATA / "gt" / "photo_jpeg.png")
    small.save(DATA / "in" / "photo_jpeg.png")  # PNG of the decoded JPEG, so artifacts are preserved exactly
    Image.open(PHOTO_GT).convert("RGB").save(DATA / "gt" / "photo.png")
    Image.open(PHOTO_GT).convert("RGB").resize((N // 4, N // 4), Image.BICUBIC).save(DATA / "in" / "photo.png")
    print("ok")
