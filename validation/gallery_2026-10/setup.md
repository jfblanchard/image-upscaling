# Gallery on Creative Commons images (2026-10-24)

- Sources: five Wikimedia Commons images, listed with licenses in `CREDITS.md` and `titles.txt`. Downloaded at Commons' standard 2048 or 3840 px renditions (original size if smaller) into `data/src/` (not committed).
- Ground truth: 1024x1024 crops (`make_inputs.py`); input: 4x bicubic downscale to 256x256.
- Methods: nearest, Lanczos, Real-ESRGAN x4plus, Real-ESRGAN x4plus anime 6B, SwinIR-L real-world x4, LDSR 50 steps, SeedVR2 3B FP8.
- `run.py` runs the local models; SeedVR2 runs via its CLI with the same settings as `model_comparison_2026-10`. `run.py --figures` builds `assets/gallery_*.jpg`.
