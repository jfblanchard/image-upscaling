# Upscaling model survey, October 2026

Goal: pick models for a public upscaling service. Commercial use matters.

| Model | Type | License | Notes |
|---|---|---|---|
| Real-ESRGAN (x2plus, x4plus, anime_6B) | GAN, fixed 2x/4x | BSD-3 | Fast, low VRAM, tiles to any size. Cannot invent detail in very degraded images. Current service default. |
| Community ESRGAN/DAT/HAT finetunes (4x-UltraSharp, Nomos8k DAT/HAT) | Transformer/GAN, 4x | varies per model; some CC-BY-NC | Sharper than stock Real-ESRGAN on many images. Drop-in via spandrel. Check each license on OpenModelDB. |
| SeedVR2 (ByteDance, 3B/7B) | One-step diffusion, arbitrary scale | Apache-2.0 (per model cards; confirm on official HF page) | Strong generative restoration, images and video. Needs a larger GPU. Best candidate for a "high quality" tier. |
| VOSR 2.0 (CVPR 2026, 1.4B DiT) | One-step diffusion | Code Apache-2.0; weights and upstream parts (SD 2.1, Qwen-Image VAE, DINOv2) need checking | Newest research option; tiling built in. |
| SUPIR | Multi-step diffusion (SDXL prior) | Restrictive: SupPixel license bars paid services | Quality leader for heavily degraded input, 10-50x slower. Not usable commercially without a license. |
| LDSR (CompVis) | Multi-step diffusion, 4x | see model card | Used in the notebook. Slow; superseded. |
| super-resolution (cartoon) | SRGAN-based, 4x/8x | see package | Used in the notebook. Superseded by Real-ESRGAN anime and DAT illustration models. |

Sources: openmodeldb.info, github.com/cswry/VOSR, docs.comfy.org/tutorials/utility/seedvr2, github.com/Fanghua-Yu/SUPIR/issues/51.
