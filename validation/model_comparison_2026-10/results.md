# Results

Seconds per image, 256x256 to 1024x1024, RTX 3060:

| Method | butterfly | cat | leaves | timer |
|---|---|---|---|---|
| Real-ESRGAN | 0.32 | 0.31 | 0.30 | 0.30 |
| LDSR 50 steps | 6.65 | 6.79 | 6.82 | 6.72 |
| LDSR 100 steps | 11.94 | 11.95 | 12.12 | 12.13 |
| SeedVR2 3B FP8 | 7.1 (first) | 5.9 | 6.0 | 5.9 |

SeedVR2 times are from the CLI log timestamps between images.

PSNR vs ground truth, dB:

| Method | butterfly | cat | leaves | timer | mean |
|---|---|---|---|---|---|
| Nearest | 30.98 | 37.16 | 39.89 | 36.04 | 36.02 |
| Bicubic | 34.49 | 39.97 | 43.32 | 38.41 | 39.05 |
| Real-ESRGAN | 29.48 | 31.44 | 35.07 | 32.78 | 32.19 |
| LDSR 50 | 31.97 | 34.61 | 30.73 | 33.78 | 32.77 |
| LDSR 100 | 31.99 | 34.51 | 30.12 | 33.62 | 32.56 |
| SeedVR2 | 29.80 | 30.31 | 31.02 | 34.24 | 31.34 |

Findings:
- Bicubic wins PSNR while looking worst: the perception-distortion tradeoff. PSNR is not a usable ranking for generative upscalers.
- LDSR 50 vs 100 steps: visually near identical on butterfly and cat crops; 50 steps is the better default.
- Both diffusion models hallucinate texture on the smooth leaf surface (LDSR crystalline, SeedVR2 streaks).
- SeedVR2 recovers the most convincing fine detail (wing scales, fur) at about half LDSR-100's time.

## Classical baselines (added)

CPU, PIL, mean of 20 runs per image; PSNR mean over the four images.

| Method | ms per image | Mean PSNR dB |
|---|---|---|
| Nearest | 0.3 | 36.02 |
| Bilinear | 3.6 | 38.20 |
| Bicubic | 5.4 | 39.05 |
| Lanczos | 8.0 | 39.36 |
| Bicubic + unsharp mask (radius 4, 120%, threshold 2) | 27.2 | 35.15 |

Sharpening lowers PSNR and adds visible halos; it raises apparent sharpness without adding information.

## SwinIR (added)

SwinIR-L real-world 4x GAN (`003_realSR_BSRGAN_DFOWMFC_s64w8_SwinIR-L_x4_GAN.pth`), fp16, 256 px tiles. 2.14 to 2.18 s per image; PSNR 29.77 / 31.67 / 32.79 / 32.35 (mean 31.65). Visually close to Real-ESRGAN.
