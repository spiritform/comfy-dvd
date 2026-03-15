import os
import sys
import gc
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from comfy import model_management as mm
from comfy.utils import ProgressBar
import folder_paths

# Add DVD repo to path so its imports work
DVD_REPO_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "dvd_repo")
if DVD_REPO_DIR not in sys.path:
    sys.path.insert(0, DVD_REPO_DIR)

# Register model folder for DVD checkpoints
DVD_MODEL_DIR = os.path.join(folder_paths.models_dir, "dvd_depth")
os.makedirs(DVD_MODEL_DIR, exist_ok=True)
folder_paths.add_model_folder_path("dvd_depth", DVD_MODEL_DIR)

# Cached model singleton
_dvd_model_cache = {"model": None, "ckpt": None}


# =============================
# Helper functions
# =============================

def compute_scale_and_shift(curr_frames, ref_frames, mask=None):
    if mask is None:
        mask = np.ones_like(ref_frames)
    a_00 = np.sum(mask * curr_frames * curr_frames)
    a_01 = np.sum(mask * curr_frames)
    a_11 = np.sum(mask)
    b_0 = np.sum(mask * curr_frames * ref_frames)
    b_1 = np.sum(mask * ref_frames)
    det = a_00 * a_11 - a_01 * a_01
    if det != 0:
        scale = (a_11 * b_0 - a_01 * b_1) / det
        shift = (-a_01 * b_0 + a_00 * b_1) / det
    else:
        scale, shift = 1.0, 0.0
    return scale, shift


def pad_time_mod4(video_tensor):
    B, T, C, H, W = video_tensor.shape
    remainder = T % 4
    if remainder != 1:
        pad_len = (4 - remainder + 1) % 4
        pad_frames = video_tensor[:, -1:, :, :, :].repeat(1, pad_len, 1, 1, 1)
        video_tensor = torch.cat([video_tensor, pad_frames], dim=1)
    return video_tensor, T


def get_window_index(T, window_size, overlap):
    if T <= window_size:
        return [(0, T)]
    res = [(0, window_size)]
    start = window_size - overlap
    while start < T:
        end = start + window_size
        if end < T:
            res.append((start, end))
            start += window_size - overlap
        else:
            start = max(0, T - window_size)
            res.append((start, T))
            break
    return res


def resize_for_model(input_tensor, scale):
    """Resize tensor (1, T, C, H, W) by scale factor, aligned to 16px."""
    B, T, C, H, W = input_tensor.shape
    if scale == 1.0:
        # Still need 16px alignment
        new_H = (H + 15) // 16 * 16
        new_W = (W + 15) // 16 * 16
        if new_H == H and new_W == W:
            return input_tensor, (H, W)
    else:
        new_H = int(H * scale)
        new_W = int(W * scale)
        new_H = (new_H + 15) // 16 * 16
        new_W = (new_W + 15) // 16 * 16
    video_reshape = input_tensor.view(B * T, C, H, W)
    resized = F.interpolate(video_reshape, size=(new_H, new_W),
                            mode="bilinear", align_corners=False)
    return resized.view(B, T, C, new_H, new_W), (H, W)


def generate_depth_sliced(model, input_rgb, window_size=45, overlap=9, pbar=None):
    B, T, C, H, W = input_rgb.shape
    depth_windows = get_window_index(T, window_size, overlap)
    print(f"[DVD] Processing {T} frames in {len(depth_windows)} windows")

    depth_res_list = []

    for idx, (start, end) in enumerate(depth_windows):
        _input_rgb_slice = input_rgb[:, start:end]
        _input_rgb_slice, origin_T = pad_time_mod4(_input_rgb_slice)
        _input_frame = _input_rgb_slice.shape[1]
        _input_height, _input_width = _input_rgb_slice.shape[-2:]

        outputs = model.pipe(
            prompt=[""] * B,
            negative_prompt=[""] * B,
            mode=model.args.mode,
            height=_input_height,
            width=_input_width,
            num_frames=_input_frame,
            batch_size=B,
            input_image=_input_rgb_slice[:, 0],
            extra_images=_input_rgb_slice,
            extra_image_frame_index=torch.ones(
                [B, _input_frame]).to(model.pipe.device),
            input_video=_input_rgb_slice,
            cfg_scale=1,
            seed=0,
            tiled=False,
            denoise_step=model.args.denoise_step,
        )
        depth_res_list.append(outputs['depth'][:, :origin_T])
        if pbar:
            pbar.update(1)

    # Overlap alignment
    depth_list_aligned = None
    prev_end = None

    for i, (t, (start, end)) in enumerate(zip(depth_res_list, depth_windows)):
        if i == 0:
            depth_list_aligned = t
            prev_end = end
            continue

        real_overlap = prev_end - start

        if real_overlap > 0:
            ref_frames = depth_list_aligned[:, -real_overlap:]
            curr_frames = t[:, :real_overlap]
            scale, shift = compute_scale_and_shift(curr_frames, ref_frames)
            scale = np.clip(scale, 0.7, 1.5)
            aligned_t = t * scale + shift
            aligned_t[aligned_t < 0] = 0

            alpha = np.linspace(0, 1, real_overlap, dtype=np.float32).reshape(
                1, real_overlap, 1, 1, 1)
            smooth_overlap = (1 - alpha) * ref_frames + alpha * aligned_t[:, :real_overlap]
            depth_list_aligned = np.concatenate(
                [depth_list_aligned[:, :-real_overlap], smooth_overlap,
                 aligned_t[:, real_overlap:]], axis=1)
        else:
            depth_list_aligned = np.concatenate(
                [depth_list_aligned, t], axis=1)
        prev_end = end

    return depth_list_aligned[:, :T]


def get_or_load_model(checkpoint):
    """Load model with caching — only reloads if checkpoint changes."""
    global _dvd_model_cache

    ckpt_path = folder_paths.get_full_path("dvd_depth", checkpoint)

    if _dvd_model_cache["model"] is not None and _dvd_model_cache["ckpt"] == ckpt_path:
        print("[DVD] Using cached model")
        return _dvd_model_cache["model"]

    # Clear old model
    if _dvd_model_cache["model"] is not None:
        del _dvd_model_cache["model"]
        _dvd_model_cache["model"] = None
        gc.collect()
        torch.cuda.empty_cache()

    from omegaconf import OmegaConf
    from safetensors.torch import load_file
    from accelerate import Accelerator
    from examples.wanvideo.model_training.WanTrainingModule import WanTrainingModule

    config_path = os.path.join(DVD_REPO_DIR, "ckpt", "model_config.yaml")
    yaml_args = OmegaConf.load(config_path)

    print("[DVD] Loading model...")
    accelerator = Accelerator()
    model = WanTrainingModule(
        accelerator=accelerator,
        model_id_with_origin_paths=yaml_args.model_id_with_origin_paths,
        trainable_models=None,
        use_gradient_checkpointing=False,
        lora_rank=yaml_args.lora_rank,
        lora_base_model=yaml_args.lora_base_model,
        args=yaml_args,
    )

    print(f"[DVD] Loading checkpoint: {ckpt_path}")
    state_dict = load_file(ckpt_path, device="cpu")
    dit_state_dict = {k.replace("pipe.dit.", ""): v for k, v in state_dict.items() if "pipe.dit." in k}
    model.pipe.dit.load_state_dict(dit_state_dict, strict=True)
    model.merge_lora_layer()

    device = mm.get_torch_device()
    model = model.to(device)
    print(f"[DVD] Model loaded on {device}")

    _dvd_model_cache["model"] = model
    _dvd_model_cache["ckpt"] = ckpt_path

    return model


def run_depth(model, input_tensor, orig_size, scale,
              window_size, overlap, colormap):
    """Core depth estimation."""
    import matplotlib.cm as cm

    T = input_tensor.shape[1]

    # Resize proportionally for model
    input_tensor, _ = resize_for_model(input_tensor, scale)
    new_H, new_W = input_tensor.shape[-2], input_tensor.shape[-1]
    print(f"[DVD] {T} frames, {orig_size[0]}x{orig_size[1]} -> processing at {new_H}x{new_W}")

    # Progress bar
    depth_windows = get_window_index(T, window_size, overlap)
    pbar = ProgressBar(len(depth_windows))

    # Inference
    with torch.no_grad():
        depth = generate_depth_sliced(model, input_tensor, window_size, overlap, pbar=pbar)

    depth = depth[0]  # (T, H, W, C)

    # Resize back to original
    depth_tensor = torch.from_numpy(depth).permute(0, 3, 1, 2).float()
    depth_tensor = F.interpolate(depth_tensor, size=orig_size,
                                 mode='bilinear', align_corners=False)
    depth = depth_tensor.permute(0, 2, 3, 1).cpu().numpy()

    # Mono depth
    depth_mono = np.mean(depth, axis=-1)
    d_min, d_max = depth_mono.min(), depth_mono.max()
    depth_norm = (depth_mono - d_min) / (d_max - d_min + 1e-8)

    # Apply colormap
    if colormap == "spectral":
        cmap = cm.get_cmap("Spectral_r")
        depth_out = cmap(depth_norm)[:, :, :, :3].astype(np.float32)
    else:
        depth_out = np.stack([depth_norm] * 3, axis=-1).astype(np.float32)

    print(f"[DVD] Done! {T} depth frames at {orig_size[0]}x{orig_size[1]}")

    return torch.from_numpy(depth_out)


# =============================
# Node: DVD Depth Estimation
# =============================
class DVDDepth:
    @classmethod
    def INPUT_TYPES(s):
        ckpt_files = folder_paths.get_filename_list("dvd_depth")
        return {
            "required": {
                "checkpoint": (ckpt_files if ckpt_files else ["model.safetensors"], {
                    "tooltip": "DVD checkpoint (model.safetensors)"
                }),
                "scale": (["1.0", "0.75", "0.5", "0.25"], {"default": "0.5",
                    "tooltip": "Processing scale relative to input. 1.0 = full res, 0.5 = half. Output always matches original size."}),
                "window_size": ("INT", {"default": 81, "min": 5, "max": 161, "step": 4,
                    "tooltip": "Frames per chunk (4n+1). Lower = less VRAM."}),
                "overlap": ("INT", {"default": 9, "min": 1, "max": 41, "step": 1}),
                "colormap": (["grayscale", "spectral"], {"default": "grayscale"}),
            },
            "optional": {
                "video": ("VIDEO", {"tooltip": "Video input from Load Video node"}),
                "images": ("IMAGE", {"tooltip": "Image frames from any node (T, H, W, C)"}),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("depth",)
    FUNCTION = "process"
    CATEGORY = "DVD Depth"
    DESCRIPTION = "Single-pass depth estimation using DVD (WanV2.1). Accepts video or image input."

    def process(self, checkpoint, scale,
                window_size, overlap, colormap, video=None, images=None):
        if video is None and images is None:
            raise ValueError("Connect either a VIDEO or IMAGE input")

        model = get_or_load_model(checkpoint)

        if video is not None:
            # Extract frames from VIDEO type
            components = video.get_components()
            frames = components.images  # torch.Tensor (T, H, W, C) or (T, C, H, W)
            # Ensure (1, T, C, H, W) format
            if frames.shape[-1] in (1, 3, 4):
                # (T, H, W, C) -> (1, T, C, H, W)
                input_tensor = frames.permute(0, 3, 1, 2).unsqueeze(0).float()
            else:
                # Already (T, C, H, W)
                input_tensor = frames.unsqueeze(0).float()
            # Normalize to [0,1] if needed
            if input_tensor.max() > 1.0:
                input_tensor = input_tensor / 255.0
            orig_size = (input_tensor.shape[3], input_tensor.shape[4])
        else:
            # IMAGE input: (T, H, W, C) float [0,1]
            orig_size = (images.shape[1], images.shape[2])
            input_tensor = images.permute(0, 3, 1, 2).unsqueeze(0).float()

        scale_float = float(scale)

        depth = run_depth(
            model, input_tensor, orig_size,
            scale_float, window_size, overlap, colormap)

        return (depth,)


# =============================
# Mappings
# =============================
NODE_CLASS_MAPPINGS = {
    "DVDDepth": DVDDepth,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "DVDDepth": "DVD Depth",
}
