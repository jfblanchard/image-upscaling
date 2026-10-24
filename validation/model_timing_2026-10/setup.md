# Model timing, local RTX 3060 (2026-10-24)

- GPU: RTX 3060 12 GB, with other processes holding ~4.5 GB. torch 2.5.1+cu124, diffusers 0.33.1, spandrel 0.4.1.
- Source: one of my sunset photos (not included), downscaled to 2000 px long side (2000x1333); 128/256/512 px squares are resized centre crops.
- Real-ESRGAN: fp16, 512 px tiles, 16 px overlap (`src/upscale.py`).
- LDSR: `CompVis/ldm-super-resolution-4x-openimages` fp16, eta=1, seed 0; inputs over 256 px run as 256 px tiles with 32 px linear-blend overlap.
- Times exclude model load; one warm-up call first. Peak VRAM is torch-allocated memory.
- Script: `run.py`. Raw: `data/timings.csv`, `data/run.log`.
