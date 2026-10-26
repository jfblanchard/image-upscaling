# Results

Mean PSNR over two images per damage type, in dB. Higher is closer to the original.

| Damage | Damaged input | Classical | Learned | SeedVR2 |
|---|---|---|---|---|
| Noise | 21.5 | 24.7 non-local means | **29.3** SCUNet (29.0 GAN) | 23.4 |
| JPEG q15 | 33.1 | 34.0 bilateral | **35.8** FBCNN (35.5 SCUNet) | 28.4 |
| Motion blur | 31.5 | 31.0 unsharp, 32.8 Wiener | **35.4** FFTformer | 29.9 |
| Scratches and dust | 20.6 | 33.2 Telea | **35.8** LaMa | 18.1 |
| Old print (all of the above) | 20.8 | 29.3 chain | **31.2** SCUNet + LaMa | 19.9 |

Time per 512×512 image, RTX 3060: classical 6–250 ms (CPU), SCUNet 0.18 s, FBCNN 0.11 s, FFTformer 1.4 s, LaMa 0.10 s, SeedVR2 about 2.5 s.

## Findings

- **A task-specific network wins every damage type**, by about 2–5 dB over the best classical method, in 1.5 s or less.
- **Wiener deconvolution needs the exact blur and still loses.** Given the true kernel, it beats unsharp masking but amplifies grain; FFTformer, which is told nothing, does better.
- **Inpainting needs a mask.** With the true mask, both Telea and LaMa remove scratches almost completely; LaMa fills texture more convincingly. Painting or detecting the mask is most of the work on a real scan.
- **SeedVR2 is an upscaler, not a repair tool.** It keeps scratches and dust as image content and sharpens them. On heavy noise it invents colored texture and shifts color. On JPEG it adds sharp, plausible wing detail that is not in the original, which costs 7 dB.
