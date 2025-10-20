"""Training module for the Lava Lamp Chaos Lab."""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from pathlib import Path
from typing import Optional, Dict, Tuple
import json
from tqdm import tqdm
import time

from .dataset import create_dataloaders
from .models import SimpleCNN, UNet, ConvLSTM


class Trainer:
    """Training manager for lava lamp prediction models."""

    def __init__(
        self,
        model: nn.Module,
        device: str = "auto",
        checkpoint_dir: str = "checkpoints",
    ):
        """
        Initialize trainer.

        Args:
            model: PyTorch model to train
            device: Device to use ('cuda', 'cpu', or 'auto')
            checkpoint_dir: Directory to save checkpoints
        """
        # Set device
        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        self.model = model.to(self.device)
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # Training state
        self.current_epoch = 0
        self.best_val_loss = float("inf")
        self.history = {"train_loss": [], "val_loss": []}

        print(f"🖥️  Using device: {self.device}")
        print(f"📁 Checkpoint directory: {self.checkpoint_dir}")

    def train_epoch(
        self,
        train_loader: DataLoader,
        criterion: nn.Module,
        optimizer: optim.Optimizer,
    ) -> float:
        """
        Train for one epoch.

        Args:
            train_loader: Training data loader
            criterion: Loss function
            optimizer: Optimizer

        Returns:
            Average training loss
        """
        self.model.train()
        total_loss = 0.0
        num_batches = 0

        pbar = tqdm(train_loader, desc=f"Epoch {self.current_epoch + 1} [Train]")

        for batch_idx, (inputs, targets) in enumerate(pbar):
            inputs = inputs.to(self.device)
            targets = targets.to(self.device)

            # Forward pass
            optimizer.zero_grad()
            outputs = self.model(inputs)
            loss = criterion(outputs, targets)

            # Backward pass
            loss.backward()
            optimizer.step()

            # Update metrics
            total_loss += loss.item()
            num_batches += 1

            # Update progress bar
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})

        avg_loss = total_loss / num_batches
        return avg_loss

    def validate(
        self,
        val_loader: DataLoader,
        criterion: nn.Module,
    ) -> float:
        """
        Validate the model.

        Args:
            val_loader: Validation data loader
            criterion: Loss function

        Returns:
            Average validation loss
        """
        self.model.eval()
        total_loss = 0.0
        num_batches = 0

        with torch.no_grad():
            pbar = tqdm(val_loader, desc=f"Epoch {self.current_epoch + 1} [Val]  ")

            for inputs, targets in pbar:
                inputs = inputs.to(self.device)
                targets = targets.to(self.device)

                # Forward pass
                outputs = self.model(inputs)
                loss = criterion(outputs, targets)

                # Update metrics
                total_loss += loss.item()
                num_batches += 1

                # Update progress bar
                pbar.set_postfix({"loss": f"{loss.item():.4f}"})

        avg_loss = total_loss / num_batches
        return avg_loss

    def train(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        epochs: int,
        learning_rate: float = 1e-3,
        criterion: Optional[nn.Module] = None,
        optimizer: Optional[optim.Optimizer] = None,
        save_every: int = 5,
    ) -> Dict:
        """
        Full training loop.

        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
            epochs: Number of epochs
            learning_rate: Learning rate
            criterion: Loss function (default: MSE)
            optimizer: Optimizer (default: Adam)
            save_every: Save checkpoint every N epochs

        Returns:
            Training history
        """
        # Setup loss and optimizer
        if criterion is None:
            criterion = nn.MSELoss()

        if optimizer is None:
            optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)

        print("\n🚀 Starting training...")
        print(f"   Epochs: {epochs}")
        print(f"   Learning rate: {learning_rate}")
        print(f"   Criterion: {criterion.__class__.__name__}")
        print(f"   Optimizer: {optimizer.__class__.__name__}")
        print()

        start_time = time.time()

        for epoch in range(epochs):
            self.current_epoch = epoch

            # Train
            train_loss = self.train_epoch(train_loader, criterion, optimizer)
            self.history["train_loss"].append(train_loss)

            # Validate
            val_loss = self.validate(val_loader, criterion)
            self.history["val_loss"].append(val_loss)

            # Print epoch summary
            print(f"Epoch {epoch + 1}/{epochs} - "
                  f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

            # Save checkpoint
            if (epoch + 1) % save_every == 0:
                self.save_checkpoint(f"checkpoint_epoch_{epoch + 1}.pt")

            # Save best model
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.save_checkpoint("best_model.pt")
                print(f"   💾 New best model! Val loss: {val_loss:.4f}")

        # Training complete
        elapsed_time = time.time() - start_time
        print(f"\n✅ Training complete! Time: {elapsed_time / 60:.2f} minutes")
        print(f"   Best val loss: {self.best_val_loss:.4f}")

        # Save final checkpoint
        self.save_checkpoint("final_model.pt")

        return self.history

    def save_checkpoint(self, filename: str):
        """Save model checkpoint."""
        checkpoint = {
            "epoch": self.current_epoch,
            "model_state_dict": self.model.state_dict(),
            "best_val_loss": self.best_val_loss,
            "history": self.history,
        }

        path = self.checkpoint_dir / filename
        torch.save(checkpoint, path)

    def load_checkpoint(self, filename: str):
        """Load model checkpoint."""
        path = self.checkpoint_dir / filename

        if not path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {path}")

        checkpoint = torch.load(path, map_location=self.device)

        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.current_epoch = checkpoint["epoch"]
        self.best_val_loss = checkpoint["best_val_loss"]
        self.history = checkpoint["history"]

        print(f"✅ Loaded checkpoint from epoch {self.current_epoch}")


def train(
    data_path: str,
    epochs: int = 10,
    batch_size: int = 4,
    learning_rate: float = 1e-3,
    model_name: str = "simple_cnn",
    frame_size: Tuple[int, int] = (128, 128),
    device: str = "auto",
    checkpoint_dir: str = "checkpoints",
) -> None:
    """
    Train a model to predict chaotic lava lamp motion.

    Args:
        data_path: Path to the training data
        epochs: Number of training epochs
        batch_size: Batch size for training
        learning_rate: Learning rate
        model_name: Model architecture ('simple_cnn', 'unet', 'conv_lstm')
        frame_size: Frame size for training
        device: Device to use
        checkpoint_dir: Directory to save checkpoints
    """
    data_dir = Path(data_path)

    if not data_dir.exists():
        print(f"⚠️  Warning: Data directory '{data_path}' does not exist")
        print("   Creating placeholder directory...")
        data_dir.mkdir(parents=True, exist_ok=True)
        print("   ⚠️  No data available. Please add training data first.")
        return

    # Check if there are any frames
    frame_files = list(data_dir.glob("*.png")) + list(data_dir.glob("*.jpg"))
    if len(frame_files) == 0:
        print(f"⚠️  No frames found in {data_path}")
        print("   Please add training data first.")
        return

    print("📊 Training configuration:")
    print(f"   • Data path: {data_path}")
    print(f"   • Frames found: {len(frame_files)}")
    print(f"   • Epochs: {epochs}")
    print(f"   • Batch size: {batch_size}")
    print(f"   • Learning rate: {learning_rate}")
    print(f"   • Model: {model_name}")
    print(f"   • Frame size: {frame_size}")
    print()

    # Create dataloaders
    try:
        train_loader, val_loader = create_dataloaders(
            data_dir=data_path,
            batch_size=batch_size,
            frame_size=frame_size,
        )
    except Exception as e:
        print(f"❌ Error creating dataloaders: {e}")
        return

    # Create model
    if model_name == "simple_cnn":
        model = SimpleCNN()
    elif model_name == "unet":
        model = UNet()
    elif model_name == "conv_lstm":
        model = ConvLSTM()
    else:
        raise ValueError(f"Unknown model: {model_name}")

    print(f"🧠 Model: {model.__class__.__name__}")
    print(f"   Parameters: {model.count_parameters():,}")
    print()

    # Create trainer and train
    trainer = Trainer(model, device=device, checkpoint_dir=checkpoint_dir)
    history = trainer.train(
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=epochs,
        learning_rate=learning_rate,
    )

    # Save training history
    history_path = Path(checkpoint_dir) / "training_history.json"
    with open(history_path, "w") as f:
        json.dump(history, f, indent=2)

    print(f"\n💾 Training history saved to {history_path}")


def main():
    """Command-line entry point."""
    import argparse

    parser = argparse.ArgumentParser(description="Train lava lamp prediction model")
    parser.add_argument("--data-path", type=str, default="data/samples",
                        help="Path to training data")
    parser.add_argument("--epochs", type=int, default=10,
                        help="Number of epochs")
    parser.add_argument("--batch-size", type=int, default=4,
                        help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-3,
                        help="Learning rate")
    parser.add_argument("--model", type=str, default="simple_cnn",
                        choices=["simple_cnn", "unet", "conv_lstm"],
                        help="Model architecture")
    parser.add_argument("--device", type=str, default="auto",
                        help="Device to use")

    args = parser.parse_args()

    train(
        data_path=args.data_path,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        model_name=args.model,
        device=args.device,
    )


if __name__ == "__main__":
    main()
