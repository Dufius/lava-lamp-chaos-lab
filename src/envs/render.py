"""
Render double-pendulum states as small greyscale images.

Output: float32 tensor [1, H, W] in [0, 1].
The pendulum is drawn as two line segments (pivot→bob1→bob2) on a black
background, with white/grey anti-aliased circles at each bob.
"""

import numpy as np
import torch


def _draw_line(img, x0, y0, x1, y1, value=1.0, thickness=1):
    """Bresenham line draw into a 2-D float numpy array."""
    h, w = img.shape
    x0, y0, x1, y1 = int(round(x0)), int(round(y0)), int(round(x1)), int(round(y1))
    dx, dy = abs(x1 - x0), abs(y1 - y0)
    sx = 1 if x0 < x1 else -1
    sy = 1 if y0 < y1 else -1
    err = dx - dy
    while True:
        for tx in range(-thickness + 1, thickness):
            for ty in range(-thickness + 1, thickness):
                nx, ny = x0 + tx, y0 + ty
                if 0 <= nx < w and 0 <= ny < h:
                    img[ny, nx] = value
        if x0 == x1 and y0 == y1:
            break
        e2 = 2 * err
        if e2 > -dy:
            err -= dy
            x0 += sx
        if e2 < dx:
            err += dx
            y0 += sy


def _draw_circle(img, cx, cy, r, value=1.0):
    h, w = img.shape
    for y in range(max(0, cy - r), min(h, cy + r + 1)):
        for x in range(max(0, cx - r), min(w, cx + r + 1)):
            if (x - cx) ** 2 + (y - cy) ** 2 <= r**2:
                img[y, x] = value


def render_state(cart_state, size=64, params=None):
    """
    Render a single Cartesian state [x1, y1, x2, y2] (normalised [-1,1])
    into a float32 numpy array [size, size].

    Parameters
    ----------
    cart_state : array-like [4]  — normalised Cartesian coordinates
    size       : int             — image side length in pixels
    """
    img = np.zeros((size, size), dtype=np.float32)

    half = size // 2
    scale = half * 0.42  # leave a small border

    # Pivot is at image centre
    px, py = half, half

    # Unnormalise: multiply back by the scale factor
    # cart_state is already in [-1, 1]; map to pixel coords
    x1_n, y1_n, x2_n, y2_n = cart_state

    b1x = int(round(px + x1_n * scale))
    b1y = int(round(py - y1_n * scale))  # y-axis flipped in image space
    b2x = int(round(px + x2_n * scale))
    b2y = int(round(py - y2_n * scale))

    # Arm 1: pivot → bob 1
    _draw_line(img, px, py, b1x, b1y, value=0.6, thickness=1)
    # Arm 2: bob 1 → bob 2
    _draw_line(img, b1x, b1y, b2x, b2y, value=0.6, thickness=1)

    # Pivot dot
    _draw_circle(img, px, py, r=2, value=0.5)
    # Bob 1
    _draw_circle(img, b1x, b1y, r=3, value=1.0)
    # Bob 2
    _draw_circle(img, b2x, b2y, r=3, value=0.85)

    return img


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
