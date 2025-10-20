"""Generate synthetic test data for testing the pipeline."""

import numpy as np
import cv2
from pathlib import Path
from tqdm import tqdm


def generate_moving_blobs(
    num_frames: int = 100,
    frame_size: tuple = (128, 128),
    num_blobs: int = 3,
) -> np.ndarray:
    """
    Generate synthetic frames with moving blobs.

    This simulates fluid motion (albeit simpler than actual lava lamps).

    Args:
        num_frames: Number of frames to generate
        frame_size: Size of each frame (height, width)
        num_blobs: Number of moving blobs

    Returns:
        Array of frames [num_frames, height, width, 3]
    """
    height, width = frame_size
    frames = []

    # Initialize blob positions and velocities
    blobs = []
    for _ in range(num_blobs):
        blob = {
            "x": np.random.uniform(20, width - 20),
            "y": np.random.uniform(20, height - 20),
            "vx": np.random.uniform(-2, 2),
            "vy": np.random.uniform(-2, 2),
            "radius": np.random.uniform(10, 20),
            "color": np.random.uniform(0.3, 1.0, size=3),
        }
        blobs.append(blob)

    print(f"🎨 Generating {num_frames} synthetic frames...")

    for frame_idx in tqdm(range(num_frames)):
        # Create blank frame
        frame = np.zeros((height, width, 3), dtype=np.float32)

        # Add gradient background
        y_gradient = np.linspace(0.1, 0.3, height).reshape(-1, 1)
        x_gradient = np.linspace(0.1, 0.3, width).reshape(1, -1)
        background = (y_gradient + x_gradient) / 2

        frame[:, :, 0] = background * 0.2  # R
        frame[:, :, 1] = background * 0.3  # G
        frame[:, :, 2] = background * 0.5  # B

        # Update and draw blobs
        for blob in blobs:
            # Update position
            blob["x"] += blob["vx"]
            blob["y"] += blob["vy"]

            # Bounce off walls
            if blob["x"] < blob["radius"] or blob["x"] > width - blob["radius"]:
                blob["vx"] *= -1
                blob["x"] = np.clip(blob["x"], blob["radius"], width - blob["radius"])

            if blob["y"] < blob["radius"] or blob["y"] > height - blob["radius"]:
                blob["vy"] *= -1
                blob["y"] = np.clip(blob["y"], blob["radius"], height - blob["radius"])

            # Add some random perturbation (chaos!)
            blob["vx"] += np.random.uniform(-0.1, 0.1)
            blob["vy"] += np.random.uniform(-0.1, 0.1)

            # Clamp velocity
            blob["vx"] = np.clip(blob["vx"], -3, 3)
            blob["vy"] = np.clip(blob["vy"], -3, 3)

            # Draw blob
            y, x = np.ogrid[:height, :width]
            mask = (x - blob["x"]) ** 2 + (y - blob["y"]) ** 2 <= blob["radius"] ** 2

            for c in range(3):
                frame[:, :, c][mask] = blob["color"][c]

        # Add some noise
        noise = np.random.normal(0, 0.02, frame.shape)
        frame = np.clip(frame + noise, 0, 1)

        frames.append(frame)

    return np.array(frames)


def save_frames_as_images(frames: np.ndarray, output_dir: str):
    """
    Save frames as individual images.

    Args:
        frames: Array of frames [num_frames, height, width, 3]
        output_dir: Directory to save frames
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    print(f"\n💾 Saving {len(frames)} frames to {output_dir}...")

    for i, frame in enumerate(tqdm(frames)):
        # Convert from float [0, 1] to uint8 [0, 255]
        frame_uint8 = (frame * 255).astype(np.uint8)

        # Convert RGB to BGR for OpenCV
        frame_bgr = cv2.cvtColor(frame_uint8, cv2.COLOR_RGB2BGR)

        # Save
        filename = output_path / f"frame_{i:06d}.png"
        cv2.imwrite(str(filename), frame_bgr)

    print(f"✅ Saved {len(frames)} frames")


def main():
    """Generate test data."""
    # Generate frames
    frames = generate_moving_blobs(
        num_frames=200,
        frame_size=(128, 128),
        num_blobs=5,
    )

    # Save to data/samples directory
    save_frames_as_images(frames, "data/samples")

    print("\n✨ Test data generation complete!")
    print("   You can now run: python run_experiment.py")


if __name__ == "__main__":
    main()
