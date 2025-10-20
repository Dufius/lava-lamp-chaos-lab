"""Dataset loader for lava lamp frame prediction."""

import torch
from torch.utils.data import Dataset, DataLoader
import cv2
import numpy as np
from pathlib import Path
from typing import Tuple, Optional, List
import torchvision.transforms as transforms


class LavaLampDataset(Dataset):
    """
    Dataset for lava lamp frame prediction.

    Loads consecutive frames for next-frame prediction tasks.
    """

    def __init__(
        self,
        data_dir: str,
        sequence_length: int = 1,
        prediction_horizon: int = 1,
        frame_size: Tuple[int, int] = (128, 128),
        transform: Optional[transforms.Compose] = None,
        normalize: bool = True,
    ):
        """
        Initialize lava lamp dataset.

        Args:
            data_dir: Directory containing frame images
            sequence_length: Number of input frames (for temporal models)
            prediction_horizon: How many frames ahead to predict
            frame_size: Target frame size (width, height)
            transform: Optional torchvision transforms
            normalize: Whether to normalize frames to [0, 1]
        """
        self.data_dir = Path(data_dir)
        self.sequence_length = sequence_length
        self.prediction_horizon = prediction_horizon
        self.frame_size = frame_size
        self.normalize = normalize

        # Default transforms
        if transform is None:
            self.transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize(frame_size),
                transforms.ToTensor(),
            ])
        else:
            self.transform = transform

        # Load all frame paths
        self.frame_paths = sorted(self.data_dir.glob("*.png"))
        self.frame_paths.extend(sorted(self.data_dir.glob("*.jpg")))
        self.frame_paths = sorted(self.frame_paths)

        if len(self.frame_paths) == 0:
            raise ValueError(f"No frames found in {data_dir}")

        # Calculate valid indices (accounting for sequence and prediction)
        self.valid_indices = len(self.frame_paths) - sequence_length - prediction_horizon + 1

        if self.valid_indices <= 0:
            raise ValueError(
                f"Not enough frames for sequence_length={sequence_length} "
                f"and prediction_horizon={prediction_horizon}"
            )

        print(f"📊 Loaded {len(self.frame_paths)} frames")
        print(f"   Valid sequences: {self.valid_indices}")
        print(f"   Sequence length: {sequence_length}")
        print(f"   Prediction horizon: {prediction_horizon}")

    def __len__(self) -> int:
        return self.valid_indices

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get a training sample.

        Returns:
            Tuple of (input_frames, target_frame)
            - input_frames: [sequence_length, C, H, W]
            - target_frame: [C, H, W]
        """
        # Load input sequence
        input_frames = []
        for i in range(self.sequence_length):
            frame_idx = idx + i
            frame = self._load_frame(self.frame_paths[frame_idx])
            input_frames.append(frame)

        # Load target frame
        target_idx = idx + self.sequence_length + self.prediction_horizon - 1
        target_frame = self._load_frame(self.frame_paths[target_idx])

        # Stack input frames
        if self.sequence_length == 1:
            input_tensor = input_frames[0]  # [C, H, W]
        else:
            input_tensor = torch.stack(input_frames)  # [T, C, H, W]

        return input_tensor, target_frame

    def _load_frame(self, path: Path) -> torch.Tensor:
        """Load and preprocess a single frame."""
        # Load image
        frame = cv2.imread(str(path))
        if frame is None:
            raise RuntimeError(f"Failed to load frame: {path}")

        # Convert BGR to RGB
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Apply transforms
        frame_tensor = self.transform(frame)

        # Normalize if requested
        if self.normalize and frame_tensor.max() > 1.0:
            frame_tensor = frame_tensor / 255.0

        return frame_tensor


class VideoDataset(Dataset):
    """
    Dataset that loads frames directly from video file.

    More memory efficient for large videos.
    """

    def __init__(
        self,
        video_path: str,
        sequence_length: int = 1,
        prediction_horizon: int = 1,
        frame_size: Tuple[int, int] = (128, 128),
        frame_skip: int = 1,
    ):
        """
        Initialize video dataset.

        Args:
            video_path: Path to video file
            sequence_length: Number of input frames
            prediction_horizon: How many frames ahead to predict
            frame_size: Target frame size (width, height)
            frame_skip: Skip every N frames (for temporal downsampling)
        """
        self.video_path = video_path
        self.sequence_length = sequence_length
        self.prediction_horizon = prediction_horizon
        self.frame_size = frame_size
        self.frame_skip = frame_skip

        # Get video info
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise RuntimeError(f"Failed to open video: {video_path}")

        self.total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.fps = cap.get(cv2.CAP_PROP_FPS)
        cap.release()

        # Calculate valid indices
        effective_frames = self.total_frames // frame_skip
        self.valid_indices = effective_frames - sequence_length - prediction_horizon + 1

        print(f"🎥 Loaded video: {video_path}")
        print(f"   Total frames: {self.total_frames}")
        print(f"   FPS: {self.fps}")
        print(f"   Valid sequences: {self.valid_indices}")

    def __len__(self) -> int:
        return self.valid_indices

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get a training sample from video."""
        cap = cv2.VideoCapture(self.video_path)

        input_frames = []
        # Load input sequence
        for i in range(self.sequence_length):
            frame_idx = (idx + i) * self.frame_skip
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            if not ret:
                raise RuntimeError(f"Failed to read frame {frame_idx}")

            frame_tensor = self._preprocess_frame(frame)
            input_frames.append(frame_tensor)

        # Load target frame
        target_idx = (idx + self.sequence_length + self.prediction_horizon - 1) * self.frame_skip
        cap.set(cv2.CAP_PROP_POS_FRAMES, target_idx)
        ret, target = cap.read()
        if not ret:
            raise RuntimeError(f"Failed to read target frame {target_idx}")

        target_tensor = self._preprocess_frame(target)
        cap.release()

        # Stack input frames
        if self.sequence_length == 1:
            input_tensor = input_frames[0]
        else:
            input_tensor = torch.stack(input_frames)

        return input_tensor, target_tensor

    def _preprocess_frame(self, frame: np.ndarray) -> torch.Tensor:
        """Preprocess a frame."""
        # Convert BGR to RGB
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Resize
        frame = cv2.resize(frame, self.frame_size)

        # Convert to tensor and normalize
        frame_tensor = torch.from_numpy(frame).permute(2, 0, 1).float() / 255.0

        return frame_tensor


def create_dataloaders(
    data_dir: str,
    batch_size: int = 4,
    train_split: float = 0.8,
    sequence_length: int = 1,
    prediction_horizon: int = 1,
    frame_size: Tuple[int, int] = (128, 128),
    num_workers: int = 0,
) -> Tuple[DataLoader, DataLoader]:
    """
    Create train and validation dataloaders.

    Args:
        data_dir: Directory containing frames
        batch_size: Batch size
        train_split: Fraction of data for training
        sequence_length: Number of input frames
        prediction_horizon: Frames ahead to predict
        frame_size: Target frame size
        num_workers: Number of worker processes

    Returns:
        Tuple of (train_loader, val_loader)
    """
    # Create full dataset
    full_dataset = LavaLampDataset(
        data_dir=data_dir,
        sequence_length=sequence_length,
        prediction_horizon=prediction_horizon,
        frame_size=frame_size,
    )

    # Split into train and validation
    train_size = int(train_split * len(full_dataset))
    val_size = len(full_dataset) - train_size

    train_dataset, val_dataset = torch.utils.data.random_split(
        full_dataset, [train_size, val_size]
    )

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    print(f"📦 Created dataloaders:")
    print(f"   Train samples: {train_size}")
    print(f"   Val samples: {val_size}")
    print(f"   Batch size: {batch_size}")

    return train_loader, val_loader
