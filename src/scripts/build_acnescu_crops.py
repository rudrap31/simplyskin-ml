"""Generate AcneSCU lesion crops + manifest CSV + source-image-level
stratified train/val/test splits for the classifier pipeline.

Usage:
    python3 src/scripts/build_acnescu_crops.py
    python3 src/scripts/build_acnescu_crops.py --padding 0.15 --seed 42
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.data.acnescu_crops import BROAD_CLASSES, MANIFEST_PATH, SPLITS_PATH, build_crops_and_manifest


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--padding", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-ratio", type=float, default=0.70)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--test-ratio", type=float, default=0.15)
    args = parser.parse_args()

    ratios = (args.train_ratio, args.val_ratio, args.test_ratio)
    assert abs(sum(ratios) - 1.0) < 1e-6, f"ratios must sum to 1.0, got {ratios}"

    summary = build_crops_and_manifest(padding=args.padding, seed=args.seed, ratios=ratios)

    print(f"Generated {summary['num_crops']} crops.")
    print(f"Rejected: {summary['rejected']}")
    print(f"\nImages per split: {summary['num_images_by_split']}")

    print("\nCrop (annotation) counts per class per split:")
    header = f"{'class':26s}" + "".join(f"{s:>10s}" for s in ("train", "val", "test"))
    print(header)
    for cls in BROAD_CLASSES:
        row = f"{cls:26s}"
        for split in ("train", "val", "test"):
            row += f"{summary['class_counts_per_split'][split].get(cls, 0):>10d}"
        print(row)

    print(f"\nManifest: {MANIFEST_PATH}")
    print(f"Splits:   {SPLITS_PATH}")


if __name__ == "__main__":
    main()
