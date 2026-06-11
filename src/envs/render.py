"""
Render double-pendulum states as small greyscale images using PIL.

Output: float32 tensor [1, H, W] in [0, 1].
"""

import numpy as np
import torch
from PIL import Image, ImageDraw


def render_state(cart_state, size=64, params=None):
    """
    Render a single Cartesian state [x1, y1, x2, y2] (normalised [-1,1])
    into a float32 numpy array [size, size].
    """
    img = Image.new("L", (size, size), color=0)
    draw = ImageDraw.Draw(img)

    half = size // 2
    scale = half * 0.42

    x1_n, y1_n, x2_n, y2_n = cart_state

    px, py = half, half
    b1x = int(round(px + x1_n * scale))
    b1y = int(round(py - y1_n * scale))
    b2x = int(round(px + x2_n * scale))
    b2y = int(round(py - y2_n * scale))

    # Arms
    draw.line([(px, py), (b1x, b1y)], fill=153, width=1)  # 0.6 * 255
    draw.line([(b1x, b1y), (b2x, b2y)], fill=153, width=1)

    # Pivot
    r = 2
    draw.ellipse([px - r, py - r, px + r, py + r], fill=128)  # 0.5 * 255
    # Bob 1
    r = 3
    draw.ellipse([b1x - r, b1y - r, b1x + r, b1y + r], fill=255)
    # Bob 2
    draw.ellipse([b2x - r, b2y - r, b2x + r, b2y + r], fill=217)  # 0.85 * 255

    return np.array(img, dtype=np.float32) / 255.0


def render_trajectory(cart_traj, size=64):
    """
    Render a full trajectory.

    Parameters
    ----------
    cart_traj : np.ndarray [T, 4]

    Returns
    -------
    torch.Tensor [T, 1, size, size]  float32 in [0, 1]
    """
    frames = [render_state(cart_traj[t], size) for t in range(len(cart_traj))]
    arr = np.stack(frames)[:, None]  # [T, 1, H, W]
    return torch.from_numpy(arr)
