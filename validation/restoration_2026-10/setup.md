# Restoration comparison (2026-10-25)

What is measured: how classical filters, task-specific networks and one generative model repair common photo damage, scored against the undamaged original. No upscaling; every output is the input size.

- Ground truth: 512×512 crops from the comparison photos (mine, not included) and the CC gallery images (see CREDITS.md). `make_inputs.py` lists sources and crop positions.
- Damage, applied by `make_inputs.py` with fixed seeds:

  | Damage | How |
  |---|---|
  | `noise` | High-ISO sensor noise: Poisson shot noise (40 photons at full scale) plus Gaussian read noise (σ 0.004), in linear light |
  | `jpeg` | JPEG quality 15 |
  | `blur` | Linear motion blur, 15 px at 30°, plus Gaussian noise σ 1/255 |
  | `scratches` | 14 thin light scratches and 250 dark or light dust specks; the exact mask is saved |
  | `old_print` | Gaussian blur σ 1.2, lighter sensor noise, 8 scratches and 120 specks, then JPEG quality 40 |

- Methods:

  | Damage | Classical | Learned | Generative |
  |---|---|---|---|
  | noise | OpenCV non-local means (h=10) | SCUNet real-world PSNR and GAN weights | SeedVR2 3B |
  | jpeg | Bilateral filter | FBCNN color; SCUNet PSNR | SeedVR2 3B |
  | blur | Unsharp mask; Wiener deconvolution given the true kernel (NSR 0.02, tuned) | FFTformer GoPro | SeedVR2 3B |
  | scratches | OpenCV Telea inpaint, true mask | LaMa (big-lama), true mask | SeedVR2 3B, no mask |
  | old_print | Non-local means, then Telea | SCUNet PSNR, then LaMa | SeedVR2 3B, no mask |

- Masks are the true damage masks: a best case. On a real scan the mask has to be painted or detected first.
- Weights and licenses: SCUNet (Apache-2.0, weights from KAIR, MIT), FBCNN (Apache-2.0), FFTformer (MIT), LaMa (Apache-2.0), SeedVR2 (Apache-2.0). All loaded with spandrel except SeedVR2.
- SeedVR2 command, from `vendor/seedvr2` (ComfyUI-SeedVR2 commit 4490bd1):
  `python inference_cli.py <data/in> --output <data/seedvr2> --model_dir <models dir> --resolution 512 --batch_size 1 --cache_dit --cache_vae --dit_offload_device cpu --vae_offload_device cpu`
- GPU: RTX 3060 12 GB. Times exclude model load and a warm-up call.
- `python run.py` runs everything except SeedVR2 and writes `data/timings.csv`, `data/psnr.csv` and `assets/restore_*.jpg`; `--figures` rebuilds figures and scores only.
