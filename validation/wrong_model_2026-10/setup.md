# Wrong-model experiments (2026-10-24)

Inputs (`make_inputs.py`), each a 1024x1024 ground truth and a 256x256 input:
- `photo`: the butterfly crop, bicubic downscale.
- `illustration`: flat-color line drawing, drawn with PIL at 4096 px and downsampled.
- `text`: printed text; body text is 24 px at 1024, so 6 px in the input.
- `photo_jpeg`: the butterfly crop, downscaled, then saved as JPEG quality 25.

Models (`run.py`): Real-ESRGAN photo and anime, SwinIR-M classical x4 (trained on clean bicubic downscaling), SwinIR-L real-world x4, LDSR 50 steps; SeedVR2 3B FP8 via its CLI. `run.py --figures` builds `assets/wrong_*.jpg`.
