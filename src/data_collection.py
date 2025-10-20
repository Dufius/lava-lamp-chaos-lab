"""Data collection module for capturing lava lamp video footage."""

import cv2
import numpy as np
from pathlib import Path
from typing import Optional, Tuple
from datetime import datetime


class VideoCapture:
    """Capture video from webcam or video file for lava lamp footage."""

    def __init__(
        self,
        source: int | str = 0,
        output_dir: str = "data/raw",
        frame_size: Tuple[int, int] = (640, 480),
        fps: int = 30,
    ):
        """
        Initialize video capture.

        Args:
            source: Camera index (0 for default webcam) or path to video file
            output_dir: Directory to save captured footage
            frame_size: Target frame size (width, height)
            fps: Frames per second for capture
        """
        self.source = source
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.frame_size = frame_size
        self.fps = fps
        self.cap: Optional[cv2.VideoCapture] = None

    def start_capture(self, duration_seconds: Optional[int] = None) -> str:
        """
        Start capturing video footage.

        Args:
            duration_seconds: Duration to capture (None for indefinite)

        Returns:
            Path to the saved video file
        """
        self.cap = cv2.VideoCapture(self.source)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.frame_size[0])
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.frame_size[1])
        self.cap.set(cv2.CAP_PROP_FPS, self.fps)

        if not self.cap.isOpened():
            raise RuntimeError(f"Failed to open video source: {self.source}")

        # Create output filename with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = self.output_dir / f"lavalamp_{timestamp}.mp4"

        # Define codec and create VideoWriter
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(
            str(output_path), fourcc, self.fps, self.frame_size
        )

        print(f"🎥 Starting capture to {output_path}")
        print(f"   Target duration: {duration_seconds}s" if duration_seconds else "   Press 'q' to stop")

        frame_count = 0
        max_frames = duration_seconds * self.fps if duration_seconds else None

        try:
            while True:
                ret, frame = self.cap.read()
                if not ret:
                    print("⚠️  Failed to read frame")
                    break

                # Resize if needed
                if frame.shape[1::-1] != self.frame_size:
                    frame = cv2.resize(frame, self.frame_size)

                out.write(frame)
                frame_count += 1

                # Show preview
                cv2.imshow("Lava Lamp Capture (press 'q' to stop)", frame)

                # Check for stop conditions
                if max_frames and frame_count >= max_frames:
                    print(f"✅ Reached target duration ({duration_seconds}s)")
                    break

                if cv2.waitKey(1) & 0xFF == ord("q"):
                    print("🛑 Capture stopped by user")
                    break

        finally:
            self.cap.release()
            out.release()
            cv2.destroyAllWindows()

        print(f"📊 Captured {frame_count} frames")
        print(f"💾 Saved to {output_path}")

        return str(output_path)

    def release(self):
        """Release video capture resources."""
        if self.cap:
            self.cap.release()
            cv2.destroyAllWindows()


def extract_frames_from_video(
    video_path: str,
    output_dir: str,
    frame_interval: int = 1,
    max_frames: Optional[int] = None,
) -> int:
    """
    Extract frames from a video file.

    Args:
        video_path: Path to the video file
        output_dir: Directory to save extracted frames
        frame_interval: Extract every Nth frame (1 = all frames)
        max_frames: Maximum number of frames to extract (None = all)

    Returns:
        Number of frames extracted
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video file: {video_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    print(f"📹 Extracting frames from {video_path}")
    print(f"   Total frames: {total_frames}")
    print(f"   FPS: {fps}")
    print(f"   Frame interval: {frame_interval}")

    frame_idx = 0
    extracted_count = 0

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Extract frame at specified interval
            if frame_idx % frame_interval == 0:
                frame_filename = output_path / f"frame_{extracted_count:06d}.png"
                cv2.imwrite(str(frame_filename), frame)
                extracted_count += 1

                if max_frames and extracted_count >= max_frames:
                    break

            frame_idx += 1

    finally:
        cap.release()

    print(f"✅ Extracted {extracted_count} frames to {output_dir}")
    return extracted_count


if __name__ == "__main__":
    # Example usage
    print("Lava Lamp Video Capture Tool")
    print("1. Capture from webcam")
    print("2. Extract frames from video")

    choice = input("Select option (1 or 2): ").strip()

    if choice == "1":
        duration = input("Duration in seconds (or press Enter for manual stop): ").strip()
        duration_sec = int(duration) if duration else None

        capturer = VideoCapture()
        try:
            video_path = capturer.start_capture(duration_seconds=duration_sec)
            print(f"\n✨ Video saved: {video_path}")
        finally:
            capturer.release()

    elif choice == "2":
        video_path = input("Enter video file path: ").strip()
        interval = input("Frame interval (default 1): ").strip()
        interval = int(interval) if interval else 1

        extract_frames_from_video(
            video_path=video_path,
            output_dir="data/frames",
            frame_interval=interval,
        )
