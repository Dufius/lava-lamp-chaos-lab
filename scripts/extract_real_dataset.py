"""Extract a training dataset from real lava lamp footage.

Reads the Creative Commons clips in data/raw/ and writes downsampled frames
to data/frames/. Lava lamps evolve slowly, so frames are sampled every
FRAME_INTERVAL source frames (~0.5s apart at 25fps) to give meaningful
motion between consecutive samples. Each frame is prefixed with its source
clip id so the dataset loader's sorted order keeps clips contiguous and only
a handful of cross-clip pairs ever occur.

Usage:
    python scripts/extract_real_dataset.py
"""

import sys
from pathlib import Path

import cv2
from tqdm import tqdm

RAW_DIR = Path("data/raw")
OUT_DIR = Path("data/frames")
FRAME_INTERVAL = 12  # sample every Nth source frame
OUT_SIZE = (256, 256)  # (width, height) stored on disk; loader resizes further
JPEG_QUALITY = 92


def extract_clip(video_path: Path, clip_id: int, out_dir: Path) -> int:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"⚠️  Could not open {video_path}, skipping")
        return 0

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    kept = 0
    src_idx = 0
    pbar = tqdm(total=total, desc=f"clip {clip_id:02d}", unit="f", leave=False)
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if src_idx % FRAME_INTERVAL == 0:
                frame = cv2.resize(frame, OUT_SIZE, interpolation=cv2.INTER_AREA)
                name = out_dir / f"lamp{clip_id:02d}_{kept:06d}.jpg"
                cv2.imwrite(
                    str(name), frame, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY]
                )
                kept += 1
            src_idx += 1
            pbar.update(1)
    finally:
        pbar.close()
        cap.release()
    return kept


def main() -> int:
    clips = sorted(RAW_DIR.glob("lavalamp_*.ogv"))
    if not clips:
        print(f"❌ No clips found in {RAW_DIR}/ (expected lavalamp_*.ogv)")
        return 1

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"🎬 Extracting from {len(clips)} clips → {OUT_DIR}/")
    print(f"   interval={FRAME_INTERVAL}  size={OUT_SIZE}  quality={JPEG_QUALITY}")

    total_kept = 0
    for clip_id, clip in enumerate(clips, start=1):
        kept = extract_clip(clip, clip_id, OUT_DIR)
        total_kept += kept
        print(f"   • {clip.name}: {kept} frames")

    print(f"\n✅ Wrote {total_kept} frames to {OUT_DIR}/")
    return 0


if __name__ == "__main__":
    sys.exit(main())
