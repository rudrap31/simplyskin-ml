"""Generate sample annotation grids for both datasets to sanity-check loaders."""
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.data.acne04 import Acne04Detection
from src.data.acnescu import BROAD_CLASSES, AcneSCUDetection
from src.viz.visualize import draw_boxes, save_sample_grid

OUT_DIR = Path(__file__).resolve().parents[2] / "outputs" / "viz_samples"

# reverse map: broad_category_id -> name, ids start at 1
ID_TO_BROAD_NAME = {i + 1: name for i, name in enumerate(BROAD_CLASSES)}


def visualize_acne04(n: int = 12, seed: int = 0):
    ds = Acne04Detection(fold=0, split="trainval")
    rng = random.Random(seed)
    indices = rng.sample(range(len(ds)), n)

    annotated = []
    for idx in indices:
        image, target = ds[idx]
        boxes = target["boxes"].tolist()
        annotated.append(draw_boxes(image, boxes, labels=["lesion"] * len(boxes)))

    out_path = OUT_DIR / "acne04_samples.jpg"
    save_sample_grid(annotated, out_path, cell_size=450)
    print(f"ACNE04: {len(ds)} images in fold. Saved {n} samples -> {out_path}")


def visualize_acnescu(n: int = 12, seed: int = 0):
    ds = AcneSCUDetection(split="train")
    rng = random.Random(seed)
    indices = rng.sample(range(len(ds)), min(n, len(ds)))

    annotated = []
    for idx in indices:
        image, target = ds[idx]
        boxes = target["boxes"].tolist()
        labels = [ID_TO_BROAD_NAME[l] for l in target["labels"].tolist()]
        annotated.append(draw_boxes(image, boxes, labels=labels))

    out_path = OUT_DIR / "acnescu_samples.jpg"
    save_sample_grid(annotated, out_path, cell_size=450)
    print(f"AcneSCU: {len(ds)} images in train split. Saved {n} samples -> {out_path}")


if __name__ == "__main__":
    visualize_acne04()
    visualize_acnescu()
