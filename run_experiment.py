import os
import torch

from src import train

if __name__ == "__main__":
    print("🧠 Lava Lamp Chaos Lab: Running baseline training...")
    train.train(data_path="data/samples", epochs=1, batch_size=2)
    print("✅ Done.")
