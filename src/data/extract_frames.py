"""
Extract and preprocess frames from a lava lamp video.

Pipeline:
  video → crop lamp glass → grayscale → resize → numpy array

Output: data/frames/lavalamp_frames.npy  shape [T, 1, H, W] float32 in [0, 1]

Usage
-----
python -m src.data.extract_frames \\
    --video data/raw/lavalamp.webm \\
    --out   data/frames/lavalamp_frames.npy \\
    --fps   5 \\
    --size  64
"""

import argparse
import os
import subprocess

import numpy as np
from PIL import Image

# Crop region for the lamp glass in the 854×480 source video.
# Adjust if you use a different source video.
CROP_X, CROP_Y = 317, 10  # top-left pixel
CROP_W, CROP_H = 220, 400  # width × height of crop


def extract_frames(video_path, fps, size, crop):
    """
    Use ffmpeg to decode video, crop to lamp, resize, convert to grayscale.

    Returns np.ndarray [T, 1, size, size] float32 in [0, 1].
    """
    cx, cy, cw, ch = crop
    vf = f"crop={cw}:{ch}:{cx}:{cy},fps={fps},scale={size}:{size},format=gray"

    cmd = [
        "ffmpeg",
        "-i",
        video_path,
        "-vf",
        vf,
        "-f",
        "rawvideo",
        "-pix_fmt",
        "gray",
        "pipe:1",
    ]
    result = subprocess.run(cmd, capture_output=True)
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg failed:\n{result.stderr.decode()}")

    raw = np.frombuffer(result.stdout, dtype=np.uint8)
    n_frames = len(raw) // (size * size)
    frames = raw[: n_frames * size * size].reshape(n_frames, 1, size, size)
    return frames.astype(np.float32) / 255.0


def save_preview(frames, path, n=8):
    """Save a contact sheet of n evenly-spaced frames."""
    idx = np.linspace(0, len(frames) - 1, n, dtype=int)
    row = np.concatenate([frames[i, 0] for i in idx], axis=1)
    img = Image.fromarray((row * 255).astype(np.uint8), mode="L")
    img.save(path)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--video", default="data/raw/lavalamp.webm")
    p.add_argument("--out", default="data/frames/lavalamp_frames.npy")
    p.add_argument(
        "--fps", type=float, default=5.0, help="frames per second to extract"
    )
    p.add_argument("--size", type=int, default=64, help="output image size (square)")
    p.add_argument("--crop-x", type=int, default=CROP_X)
    p.add_argument("--crop-y", type=int, default=CROP_Y)
    p.add_argument("--crop-w", type=int, default=CROP_W)
    p.add_argument("--crop-h", type=int, default=CROP_H)
    args = p.parse_args()

    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    print(f"Extracting frames from {args.video} at {args.fps} fps ...")
    crop = (args.crop_x, args.crop_y, args.crop_w, args.crop_h)
    frames = extract_frames(args.video, args.fps, args.size, crop)
    print(
        f"  → {len(frames)} frames  shape {frames.shape}  ({frames.nbytes/1e6:.1f} MB)"
    )

    np.save(args.out, frames)
    print(f"Saved: {args.out}")

    preview_path = args.out.replace(".npy", "_preview.png")
    save_preview(frames, preview_path)
    print(f"Preview: {preview_path}")


if __name__ == "__main__":
    main()
