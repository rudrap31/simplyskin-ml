"""Pre-training audit for classifier_v2's crop-scale augmentation: does
expanding a training crop (1.5x / 2x the manifest box) frequently pull in
a DIFFERENT, already-labeled lesion from the same image? If so, that
annotation's label (the original lesion's class) becomes wrong for the
expanded crop — heavy label noise, silently, unless measured.

This is a ground-truth-based check, not a classifier-prediction-based
one: for every train-split annotation, it deterministically expands the
crop (same math as training, but WITHOUT the random center shift — pure
symmetric expansion — so this is a conservative/lower-bound contamination
estimate; shifting only makes contamination more likely) and checks
whether any OTHER annotation's box in the same image, of a DIFFERENT
mapped_class, has its center fall inside the expanded region.

Also saves a visual grid per class (tight / 1.5x / 2x) for manual
inspection, with extra emphasis on comedonal_like per request.

Usage:
    python3 src/scripts/audit_crop_scale_contamination.py
"""
import csv
import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

from src.data.acnescu import BROAD_CLASSES
from src.data.acnescu_crops import DATASET_ROOT, MANIFEST_PATH, expanded_crop_box

REPO_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = REPO_ROOT / "runs" / "crop_scale_contamination_audit"
SCALES = [1.0, 1.5, 2.0]
NUM_EXAMPLES_PER_CLASS = {"comedonal_like": 8}
DEFAULT_EXAMPLES_PER_CLASS = 4
VIS_SEED = 42


def load_manifest_rows():
    with open(MANIFEST_PATH, newline="") as f:
        return list(csv.DictReader(f))


def box_center(box):
    return (box[0] + box[2]) / 2, (box[1] + box[3]) / 2


def point_in_box(point, box):
    x, y = point
    return box[0] <= x <= box[2] and box[1] <= y <= box[3]


def run_contamination_audit(rows):
    train_rows = [r for r in rows if r["split"] == "train"]

    rows_by_image = defaultdict(list)
    for r in train_rows:
        rows_by_image[r["source_image_id"]].append(r)

    # image dims: read once per image from the manifest's own crop
    # geometry isn't enough (we don't store full-image size in the
    # manifest), so open each source image once
    image_sizes = {}

    contamination_counts = {cls: {s: 0 for s in SCALES} for cls in BROAD_CLASSES}
    total_counts = {cls: 0 for cls in BROAD_CLASSES}
    contaminating_class_breakdown = {cls: {s: Counter() for s in SCALES} for cls in BROAD_CLASSES}

    for image_id, image_rows in rows_by_image.items():
        image_path = REPO_ROOT / image_rows[0]["source_image_path"]
        if image_path not in image_sizes:
            with Image.open(image_path) as im:
                image_sizes[image_path] = im.size
        img_w, img_h = image_sizes[image_path]

        other_boxes = [
            (
                [float(r["bbox_x1"]), float(r["bbox_y1"]), float(r["bbox_x2"]), float(r["bbox_y2"])],
                r["mapped_class"],
                r["annotation_id"],
            )
            for r in image_rows
        ]

        for row in image_rows:
            cls = row["mapped_class"]
            total_counts[cls] += 1
            padded_box = [float(row["bbox_x1"]), float(row["bbox_y1"]), float(row["bbox_x2"]), float(row["bbox_y2"])]

            for scale in SCALES:
                expanded = expanded_crop_box(padded_box, img_w, img_h, scale)
                found_different_class = False
                for other_box, other_cls, other_ann_id in other_boxes:
                    if other_ann_id == row["annotation_id"] or other_cls == cls:
                        continue
                    if point_in_box(box_center(other_box), expanded):
                        found_different_class = True
                        contaminating_class_breakdown[cls][scale][other_cls] += 1
                if found_different_class:
                    contamination_counts[cls][scale] += 1

    return total_counts, contamination_counts, contaminating_class_breakdown


def save_visualization_grid(rows, class_name: str, num_examples: int, output_dir: Path):
    rng = random.Random(VIS_SEED)
    class_rows = [r for r in rows if r["split"] == "train" and r["mapped_class"] == class_name]
    if not class_rows:
        return
    sample = rng.sample(class_rows, min(num_examples, len(class_rows)))

    fig, axes = plt.subplots(len(sample), len(SCALES), figsize=(len(SCALES) * 3, len(sample) * 3.2))
    if len(sample) == 1:
        axes = axes.reshape(1, -1)

    for row_idx, row in enumerate(sample):
        image_path = REPO_ROOT / row["source_image_path"]
        with Image.open(image_path) as im:
            image = im.convert("RGB")
            img_w, img_h = image.size
            padded_box = [float(row["bbox_x1"]), float(row["bbox_y1"]), float(row["bbox_x2"]), float(row["bbox_y2"])]

            for col, scale in enumerate(SCALES):
                box = expanded_crop_box(padded_box, img_w, img_h, scale)
                crop = image.crop((box[0], box[1], box[2], box[3]))
                axes[row_idx, col].imshow(crop)
                axes[row_idx, col].set_title(f"{scale}x  (ann {row['annotation_id']})", fontsize=9)
                axes[row_idx, col].axis("off")

    fig.suptitle(f"class={class_name}", fontsize=12)
    fig.tight_layout()
    fig.savefig(output_dir / f"{class_name}_scale_grid.jpg", dpi=120)
    plt.close(fig)


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    rows = load_manifest_rows()

    print("Generating per-class visualization grids (tight / 1.5x / 2x)...")
    for cls in BROAD_CLASSES:
        n = NUM_EXAMPLES_PER_CLASS.get(cls, DEFAULT_EXAMPLES_PER_CLASS)
        save_visualization_grid(rows, cls, n, OUTPUT_DIR)
        print(f"  saved {cls}_scale_grid.jpg ({n} examples)")

    print("\nRunning ground-truth contamination audit (train split)...")
    total_counts, contamination_counts, breakdown = run_contamination_audit(rows)

    print("\n" + "=" * 80)
    print("Fraction of expanded crops whose region contains a DIFFERENT-class lesion's center")
    print("(1.0x is the manifest's own tight box; contamination there would indicate")
    print(" annotation boxes for different classes already overlap even without expansion)")
    print("=" * 80)
    header = f"{'class':26s}" + "".join(f"{s:>10.1f}x" for s in SCALES)
    print(header)
    summary_rows = {}
    for cls in BROAD_CLASSES:
        total = total_counts[cls]
        row_str = f"{cls:26s}"
        summary_rows[cls] = {}
        for scale in SCALES:
            frac = contamination_counts[cls][scale] / total if total else 0.0
            summary_rows[cls][scale] = frac
            row_str += f"{frac:>10.1%}"
        print(row_str)

    print("\nContaminating-class breakdown (which OTHER class shows up), at 2.0x scale:")
    for cls in BROAD_CLASSES:
        top = breakdown[cls][2.0].most_common(3)
        if top:
            top_str = ", ".join(f"{name}={n}" for name, n in top)
            print(f"  {cls:26s} <- {top_str}")

    summary = {
        "total_counts": total_counts,
        "contamination_fraction": summary_rows,
        "contaminating_class_breakdown_at_2x": {
            cls: dict(breakdown[cls][2.0].most_common()) for cls in BROAD_CLASSES
        },
    }
    with open(OUTPUT_DIR / "contamination_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nSaved: {OUTPUT_DIR / 'contamination_summary.json'}")
    print(f"Saved visualization grids -> {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
