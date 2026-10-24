# Model comparison, 256 to 1024 (2026-10-24)

- Ground truth: 1024x1024 crops from my own photos (not included), chosen by hand for texture, edges and print. `make_inputs.py` lists sources and crop positions.
- Input: ground truth downscaled 4x with bicubic to 256x256.
- Methods: nearest and bicubic (baselines), Real-ESRGAN x4plus (fp16), LDSR 50 and 100 steps (fp16, eta=1, seed 0), SeedVR2 3B FP8 (ComfyUI-SeedVR2 CLI commit 4490bd1, `--resolution 1024 --batch_size 1`, DiT and VAE offloaded to CPU between images).
- GPU: RTX 3060 12 GB, with other processes holding ~4.5 GB. Times exclude model load and the first warm-up call.
- Figures: `run.py` writes `assets/compare_*.jpg`; `--figures-only` rebuilds them without rerunning models.
- SeedVR2 command, from `vendor/seedvr2`:
  `python inference_cli.py <data/in> --output <data/seedvr2> --model_dir <models dir> --resolution 1024 --batch_size 1 --cache_dit --cache_vae --dit_offload_device cpu --vae_offload_device cpu`
