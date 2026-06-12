import argparse

from src import train

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run baseline training experiment")
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cpu", "cuda", "hybrid"],
        help="Device to use: cpu, cuda, hybrid (model split across GPU+CPU), "
        "or auto (cuda if available)",
    )
    args = parser.parse_args()

    print("🧠 Lava Lamp Chaos Lab: Running baseline training...")
    train.train(data_path="data/samples", epochs=1, batch_size=2, device=args.device)
    print("✅ Done.")
