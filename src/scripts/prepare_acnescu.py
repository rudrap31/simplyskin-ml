"""Build remapped AcneSCU annotations + train/val/test splits."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from collections import Counter

from src.data.acnescu import build_remapped_annotations, build_splits


def main():
    remapped = build_remapped_annotations()
    counts = Counter(a["broad_category"] for a in remapped["annotations"])
    print("Remapped annotation counts:")
    for name, n in counts.most_common():
        print(f"  {name}: {n}")
    print(f"  dropped (other): {remapped['dropped_other_count']}")

    splits = build_splits()
    print("\nSplit sizes (images):")
    for name, ids in splits.items():
        print(f"  {name}: {len(ids)}")


if __name__ == "__main__":
    main()
