# Demo: hosted upscaler

Static front end on Cloudflare Pages (`web/`), GPU inference on Modal (`modal_app.py`).

## Deploy

1. Backend: `modal deploy demo/modal_app.py`. Copy the printed `web` URL into `web/config.js`.
2. Front end: `cd demo/web && wrangler pages deploy . --project-name image-upscaling`
3. If the Pages domain differs from `image-upscaling.pages.dev`, add it to `ALLOWED_ORIGINS` in `modal_app.py` and redeploy.

Local front end: `cd demo/web && npx wrangler pages dev .` (serves on :8788, already allowed by CORS).

## Models

| Key | Weights | Scale |
|---|---|---|
| `photo-x2` | RealESRGAN_x2plus | 2x |
| `photo-x4` | RealESRGAN_x4plus | 4x |
| `illustration-x4` | RealESRGAN_x4plus_anime_6B | 4x |

All BSD-3-Clause, loaded with [spandrel](https://github.com/chaiNNer-org/spandrel).

## Limits

- Input long side ≤ 2000 px, upload ≤ 15 MB. A 1000×2000 input at 4x gives 4000×8000.
- Inference runs on 512 px tiles with 16 px overlap, so GPU memory is bounded for any input size.
- L4 GPU, up to 2 containers, scales to zero after 2 minutes idle. The first request after idle waits for a cold start.
