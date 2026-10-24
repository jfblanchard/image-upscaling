"""Real-ESRGAN upscaling service on Modal.

Deploy:  modal deploy demo/modal_app.py
Dev:     modal serve demo/modal_app.py

POST /upscale  (multipart: file, model, format) -> upscaled image
GET  /health
"""

import time

import modal

WEIGHTS = {
    # name: (url, scale) -- official Real-ESRGAN release weights, BSD-3-Clause
    "photo-x2": ("https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.1/RealESRGAN_x2plus.pth", 2),
    "photo-x4": ("https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth", 4),
    "illustration-x4": ("https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.2.4/RealESRGAN_x4plus_anime_6B.pth", 4),
}

MAX_INPUT_SIDE = 2000          # px, long side
MAX_UPLOAD_BYTES = 15 * 2**20  # 15 MB
TILE = 512                     # px, input tile size; bounds GPU memory for any image size
TILE_PAD = 16                  # px, overlap so tile seams don't show

# Rate limits (fixed one-hour windows): per client IP, plus a global daily cap as a budget guard
LIMIT_PER_IP_PER_HOUR = 20
LIMIT_GLOBAL_PER_DAY = 500

ALLOWED_ORIGINS = [
    "https://image-upscaling.pages.dev",
    "http://localhost:8788",
    "http://127.0.0.1:8788",
]

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("torch==2.5.1", "spandrel==0.4.1", "pillow==11.0.0", "numpy<2.2", "fastapi[standard]==0.115.6")
    .run_commands(
        "mkdir -p /weights",
        *[f"curl -fsSL -o /weights/{name}.pth {url}" for name, (url, _) in WEIGHTS.items()],
    )
)

app = modal.App("image-upscaling", image=image)
usage = modal.Dict.from_name("image-upscaling-usage", create_if_missing=True)


def client_ip(request):
    # Modal's proxy appends the connecting address, so the rightmost X-Forwarded-For entry is trusted
    xff = request.headers.get("x-forwarded-for", "")
    return xff.split(",")[-1].strip() if xff else request.client.host


async def check_rate_limit(ip):
    """Return None if allowed (and count it), else a message. Not atomic; fine at demo scale."""
    now = int(time.time())
    hour, day = now // 3600, now // 86400
    k_ip, k_day = f"ip:{ip}:{hour}", f"all:{day}"
    n_ip, n_day = await usage.get.aio(k_ip, 0), await usage.get.aio(k_day, 0)
    if n_day >= LIMIT_GLOBAL_PER_DAY:
        return "The demo has reached its daily limit. Please try again tomorrow."
    if n_ip >= LIMIT_PER_IP_PER_HOUR:
        mins = 60 - (now % 3600) // 60
        return f"Limit of {LIMIT_PER_IP_PER_HOUR} upscales per hour reached. Try again in {mins} min."
    await usage.put.aio(k_ip, n_ip + 1)
    await usage.put.aio(k_day, n_day + 1)
    return None


@app.cls(gpu="L4", scaledown_window=120, max_containers=2, timeout=300)
class Upscaler:
    @modal.enter()
    def load(self):
        import torch
        from spandrel import ModelLoader

        self.torch = torch
        self.models = {}
        for name in WEIGHTS:
            m = ModelLoader().load_from_file(f"/weights/{name}.pth").eval().cuda()
            self.models[name] = m.half() if m.supports_half else m

    def upscale(self, img, name):
        """Tiled inference: run the model on overlapping tiles, keep each tile's centre."""
        import numpy as np

        torch = self.torch
        model = self.models[name]
        scale = model.scale
        dtype = next(model.model.parameters()).dtype

        x = torch.from_numpy(np.array(img)).permute(2, 0, 1).unsqueeze(0)
        x = x.to("cuda", dtype) / 255.0
        _, _, h, w = x.shape
        out = torch.empty((3, h * scale, w * scale), dtype=torch.uint8, device="cuda")

        with torch.inference_mode():
            for y0 in range(0, h, TILE):
                for x0 in range(0, w, TILE):
                    y1, x1 = min(y0 + TILE, h), min(x0 + TILE, w)
                    py0, px0 = max(y0 - TILE_PAD, 0), max(x0 - TILE_PAD, 0)
                    py1, px1 = min(y1 + TILE_PAD, h), min(x1 + TILE_PAD, w)
                    tile = model(x[:, :, py0:py1, px0:px1])
                    tile = tile[0, :, (y0 - py0) * scale:(y1 - py0) * scale, (x0 - px0) * scale:(x1 - px0) * scale]
                    out[:, y0 * scale:y1 * scale, x0 * scale:x1 * scale] = (
                        tile.float().clamp(0, 1).mul(255).round().to(torch.uint8)
                    )

        from PIL import Image
        return Image.fromarray(out.permute(1, 2, 0).cpu().numpy())

    @modal.asgi_app()
    def web(self):
        import asyncio
        import io

        from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
        from fastapi.middleware.cors import CORSMiddleware
        from fastapi.responses import Response
        from PIL import Image, ImageOps

        api = FastAPI(title="image-upscaling")
        api.add_middleware(
            CORSMiddleware,
            allow_origins=ALLOWED_ORIGINS,
            allow_methods=["GET", "POST"],
            allow_headers=["*"],
            expose_headers=["X-Upscale-Seconds", "X-Output-Size"],
        )

        @api.get("/health")
        def health():
            return {"ok": True, "models": list(WEIGHTS), "max_input_side": MAX_INPUT_SIDE}

        @api.post("/upscale")
        async def upscale(
            request: Request,
            file: UploadFile = File(...),
            model: str = Form("photo-x4"),
            format: str = Form("png"),
        ):
            if model not in WEIGHTS:
                raise HTTPException(400, f"Unknown model '{model}'")
            if format not in ("png", "jpeg"):
                raise HTTPException(400, "format must be png or jpeg")

            data = await file.read()
            if len(data) > MAX_UPLOAD_BYTES:
                raise HTTPException(413, "File is larger than 15 MB")
            try:
                img = ImageOps.exif_transpose(Image.open(io.BytesIO(data))).convert("RGB")
            except Exception:
                raise HTTPException(400, "Could not read image")
            if max(img.size) > MAX_INPUT_SIDE:
                raise HTTPException(413, f"Image is {img.width}x{img.height}; long side must be {MAX_INPUT_SIDE} px or less")
            if msg := await check_rate_limit(client_ip(request)):
                raise HTTPException(429, msg)

            t0 = time.perf_counter()
            result = await asyncio.to_thread(self.upscale, img, model)
            elapsed = time.perf_counter() - t0

            buf = io.BytesIO()
            if format == "png":
                result.save(buf, "PNG", compress_level=3)
            else:
                result.save(buf, "JPEG", quality=95)
            return Response(
                buf.getvalue(),
                media_type=f"image/{format}",
                headers={
                    "X-Upscale-Seconds": f"{elapsed:.2f}",
                    "X-Output-Size": f"{result.width}x{result.height}",
                },
            )

        return api
