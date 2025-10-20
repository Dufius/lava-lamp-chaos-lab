"""Inference module for making predictions with trained models."""

import torch
import cv2
import numpy as np
from pathlib import Path
from typing import List, Optional, Tuple
import torchvision.transforms as transforms

from .models import SimpleCNN, UNet, ConvLSTM
from .evaluation import compute_metrics, visualize_predictions, visualize_prediction_sequence


class Predictor:
    """Inference class for lava lamp prediction."""

    def __init__(
        self,
        model_path: str,
        model_type: str = "simple_cnn",
        device: str = "auto",
        frame_size: Tuple[int, int] = (128, 128),
    ):
        """
        Initialize predictor.

        Args:
            model_path: Path to trained model checkpoint
            model_type: Model architecture type
            device: Device to use
            frame_size: Frame size the model was trained on
        """
        # Set device
        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        self.frame_size = frame_size

        # Create model
        if model_type == "simple_cnn":
            self.model = SimpleCNN()
        elif model_type == "unet":
            self.model = UNet()
        elif model_type == "conv_lstm":
            self.model = ConvLSTM()
        else:
            raise ValueError(f"Unknown model type: {model_type}")

        # Load checkpoint
        checkpoint = torch.load(model_path, map_location=self.device)

        if "model_state_dict" in checkpoint:
            self.model.load_state_dict(checkpoint["model_state_dict"])
        else:
            self.model.load_state_dict(checkpoint)

        self.model.to(self.device)
        self.model.eval()

        print(f"✅ Loaded model from {model_path}")
        print(f"   Device: {self.device}")

    def preprocess_frame(self, frame: np.ndarray) -> torch.Tensor:
        """
        Preprocess a frame for prediction.

        Args:
            frame: Input frame as numpy array [H, W, C]

        Returns:
            Preprocessed tensor [1, C, H, W]
        """
        # Convert BGR to RGB if needed
        if frame.shape[2] == 3:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Resize
        frame = cv2.resize(frame, self.frame_size)

        # Convert to tensor and normalize
        transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.ToTensor(),
        ])

        frame_tensor = transform(frame)

        # Add batch dimension
        frame_tensor = frame_tensor.unsqueeze(0)

        return frame_tensor

    def postprocess_frame(self, tensor: torch.Tensor) -> np.ndarray:
        """
        Convert model output back to numpy image.

        Args:
            tensor: Output tensor [1, C, H, W]

        Returns:
            Image as numpy array [H, W, C]
        """
        # Remove batch dimension and move to CPU
        frame = tensor.squeeze(0).cpu().numpy()

        # Transpose from [C, H, W] to [H, W, C]
        frame = np.transpose(frame, (1, 2, 0))

        # Clip to valid range
        frame = np.clip(frame, 0, 1)

        # Convert to uint8
        frame = (frame * 255).astype(np.uint8)

        return frame

    def predict_next_frame(self, frame: np.ndarray) -> np.ndarray:
        """
        Predict the next frame given current frame.

        Args:
            frame: Current frame [H, W, C]

        Returns:
            Predicted next frame [H, W, C]
        """
        # Preprocess
        input_tensor = self.preprocess_frame(frame).to(self.device)

        # Predict
        with torch.no_grad():
            output_tensor = self.model(input_tensor)

        # Postprocess
        prediction = self.postprocess_frame(output_tensor)

        return prediction

    def predict_sequence(
        self,
        initial_frame: np.ndarray,
        num_steps: int,
    ) -> List[np.ndarray]:
        """
        Predict a sequence of frames autoregressively.

        This is key for testing prediction horizon limits!

        Args:
            initial_frame: Starting frame [H, W, C]
            num_steps: Number of future frames to predict

        Returns:
            List of predicted frames
        """
        predictions = []
        current_frame = initial_frame

        print(f"🔮 Predicting {num_steps} frames autoregressively...")

        for step in range(num_steps):
            # Predict next frame
            next_frame = self.predict_next_frame(current_frame)
            predictions.append(next_frame)

            # Use prediction as input for next step
            current_frame = next_frame

            if (step + 1) % 10 == 0:
                print(f"   Predicted {step + 1}/{num_steps} frames")

        return predictions

    def predict_from_video(
        self,
        video_path: str,
        output_path: str,
        max_frames: Optional[int] = None,
    ):
        """
        Process a video and create predictions.

        Args:
            video_path: Input video path
            output_path: Output video path
            max_frames: Maximum frames to process
        """
        cap = cv2.VideoCapture(video_path)

        if not cap.isOpened():
            raise RuntimeError(f"Failed to open video: {video_path}")

        # Get video properties
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        print(f"📹 Processing video: {video_path}")
        print(f"   Resolution: {width}x{height}")
        print(f"   FPS: {fps}")
        print(f"   Total frames: {total_frames}")

        # Create video writer
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(output_path, fourcc, fps, (width * 2, height))

        frame_idx = 0

        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                # Predict next frame
                prediction = self.predict_next_frame(frame)

                # Resize prediction to match input
                prediction = cv2.resize(prediction, (width, height))

                # Create side-by-side comparison
                comparison = np.hstack([frame, prediction])

                # Write to output
                out.write(comparison)

                frame_idx += 1

                if max_frames and frame_idx >= max_frames:
                    break

                if frame_idx % 30 == 0:
                    print(f"   Processed {frame_idx} frames")

        finally:
            cap.release()
            out.release()

        print(f"✅ Processed {frame_idx} frames")
        print(f"💾 Saved to {output_path}")


def main():
    """Command-line entry point."""
    import argparse

    parser = argparse.ArgumentParser(description="Make predictions with trained model")
    parser.add_argument("--model-path", type=str, required=True,
                        help="Path to model checkpoint")
    parser.add_argument("--model-type", type=str, default="simple_cnn",
                        choices=["simple_cnn", "unet", "conv_lstm"],
                        help="Model architecture")
    parser.add_argument("--input", type=str, required=True,
                        help="Input image or video path")
    parser.add_argument("--output", type=str, default="prediction.png",
                        help="Output path")
    parser.add_argument("--steps", type=int, default=1,
                        help="Number of steps to predict (for sequence)")
    parser.add_argument("--device", type=str, default="auto",
                        help="Device to use")

    args = parser.parse_args()

    # Create predictor
    predictor = Predictor(
        model_path=args.model_path,
        model_type=args.model_type,
        device=args.device,
    )

    # Check if input is video or image
    input_path = Path(args.input)
    if input_path.suffix.lower() in [".mp4", ".avi", ".mov"]:
        # Process video
        predictor.predict_from_video(
            video_path=args.input,
            output_path=args.output,
        )
    else:
        # Process single image
        frame = cv2.imread(args.input)
        if frame is None:
            raise RuntimeError(f"Failed to load image: {args.input}")

        if args.steps == 1:
            # Single prediction
            prediction = predictor.predict_next_frame(frame)
            cv2.imwrite(args.output, prediction)
            print(f"💾 Saved prediction to {args.output}")
        else:
            # Sequence prediction
            predictions = predictor.predict_sequence(frame, args.steps)

            # Save as video or image grid
            if args.output.endswith(".mp4"):
                # Save as video
                height, width = predictions[0].shape[:2]
                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                out = cv2.VideoWriter(args.output, fourcc, 10, (width, height))

                for pred in predictions:
                    out.write(pred)

                out.release()
                print(f"💾 Saved prediction sequence to {args.output}")
            else:
                # Save first prediction
                cv2.imwrite(args.output, predictions[-1])
                print(f"💾 Saved final prediction to {args.output}")


if __name__ == "__main__":
    main()
