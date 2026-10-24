# Results

| Model | Input | Output | Seconds | Peak VRAM GiB |
|---|---|---|---|---|
| Real-ESRGAN photo 4x | 512x512 | 2048x2048 | 1.2 | 2.0 |
| Real-ESRGAN photo 4x | 2000x1333 | 8000x5332 | 13.4 | 2.5 |
| Real-ESRGAN photo 2x | 2000x1333 | 4000x2666 | 3.6 | 0.7 |
| Real-ESRGAN illustration 4x | 2000x1333 | 8000x5332 | 4.6 | 2.4 |
| LDSR 100 steps | 256x256 | 1024x1024 | 12.9 | 2.7 |
| LDSR 50 steps | 256x256 | 1024x1024 | 6.8 | 2.7 |
| LDSR 25 steps | 256x256 | 1024x1024 | 4.2 | 2.7 |
| LDSR 50 steps (9 tiles) | 512x512 | 2048x2048 | 62.3 | 2.7 |

- LDSR time is linear in steps; tiling keeps VRAM flat at 2.7 GiB.
- LDSR on 2000x1333 at 50 steps extrapolates to ~6-7 min per image on this GPU.
- Quality: the sunset crop is too smooth to separate the models or to reveal tile seams. Needs a detailed test image (texture, text, faces) before choosing step count.
