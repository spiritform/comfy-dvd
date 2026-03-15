# ComfyUI-DVD-Depth

A ComfyUI custom node for **DVD (Deterministic Video Depth)** — single-pass, temporally consistent depth estimation from video using Wan2.1.

Based on the paper ["Video Diffusion Models are Overqualified Depth Estimators"](https://dvd-project.github.io/) by EnVision Research.

![DVD Depth Output](https://github.com/EnVision-Research/DVD/raw/main/assets/teaser.png)

## Features

- **Single-pass depth estimation** — no iterative denoising, just one forward pass through the DiT
- **Temporally consistent** — no flicker between frames, zero scale drift across infinite-length videos
- **SOTA quality** — 5.5 AbsRel on ScanNet benchmark
- **Flexible input** — accepts VIDEO (from Load Video) or IMAGE frames (from any node)
- **Proportional scaling** — auto-detects input resolution, scales by factor (1.0, 0.75, 0.5, 0.25)
- **Sliding window** — handles videos of any length with overlap blending
- **Model caching** — loads once, stays in VRAM across runs

## Installation

### Via ComfyUI Manager
Search for `comfy-dvd` in ComfyUI Manager and install.

### Manual Installation

1. Clone into your ComfyUI custom nodes folder:
```bash
cd ComfyUI/custom_nodes
git clone --recursive https://github.com/spiritform/comfy-dvd.git ComfyUI-DVD-Depth
```

2. Install dependencies (use ComfyUI's Python if using portable):
```bash
pip install omegaconf peft accelerate safetensors einops sentencepiece matplotlib modelscope pandas "imageio[ffmpeg]"
```

3. Download the DVD checkpoint (~4.5GB):
```bash
huggingface-cli download FayeHongfeiZhang/DVD model.safetensors --local-dir ComfyUI/models/dvd_depth
```

The Wan2.1-T2V-1.3B base weights will auto-download from HuggingFace on first run.

## Node: DVD Depth

### Inputs

| Input | Type | Description |
|-------|------|-------------|
| **video** | VIDEO (optional) | Video from Load Video node |
| **images** | IMAGE (optional) | Image frames from any node |
| **checkpoint** | dropdown | DVD model checkpoint |
| **scale** | dropdown | Processing scale: 1.0, 0.75, 0.5, 0.25 (output resizes back to original) |
| **window_size** | INT (default 81) | Frames per chunk. Lower = less VRAM |
| **overlap** | INT (default 9) | Overlap frames for temporal blending |
| **colormap** | dropdown | grayscale or spectral visualization |

Connect either `video` OR `images` — at least one is required.

### Output

| Output | Type | Description |
|--------|------|-------------|
| **depth** | IMAGE | Depth map frames matching input resolution |

## VRAM Requirements

The model loads ~4.3GB in bf16 (DiT + VAE + CLIP, text encoder is skipped). Remaining VRAM goes to inference activations.

| Scale | Resolution (from 1280x720) | Approx VRAM |
|-------|---------------------------|-------------|
| 1.0 | 1280x720 | ~16GB+ |
| 0.75 | 960x544 | ~12GB |
| 0.5 | 640x368 | ~8GB |
| 0.25 | 320x192 | ~6GB |

Lower `window_size` (e.g., 21 or 45) also reduces VRAM significantly.

## Credits

- [DVD: Deterministic Video Depth](https://github.com/EnVision-Research/DVD) by EnVision Research
- Built on [Wan2.1-T2V-1.3B](https://huggingface.co/Wan-AI/Wan2.1-T2V-1.3B)

## License

This wrapper follows the license of the original DVD repository.
