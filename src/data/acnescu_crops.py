"""AcneSCU lesion-crop classifier dataset: builds per-annotation crops
(padded, clamped bounding boxes) from the raw COCO annotations, a CSV
manifest describing every crop, and a source-image-level train/val/test
split (stratified by each image's dominant class where practical).

Kept separate from src/data/acnescu.py (which serves the broad-class
*detection* dataset / splits.json) so this classifier work doesn't
disturb that existing artifact or its consumers.

Class mapping (locked v1 scope, same as src/data/acnescu.py):
  - comedonal_like           <- closed_comedo, open_comedo
  - inflammatory_like        <- papule, pustule
  - deeper_inflammatory_like <- nodule
  - non_active_acne          <- atrophic_scar, hypertrophic_scar, melasma, nevus
  - (dropped)                <- other
"""
import csv
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import Dataset

from src.data.acnescu import BROAD_CLASSES, RAW_ANNOTATIONS, RAW_TO_BROAD

DATASET_ROOT = Path(__file__).resolve().parents[2] / "datasets" / "acnescu"
IMAGES_DIR = DATASET_ROOT / "train"
CLASSIFIER_DIR = DATASET_ROOT / "classifier"
CROPS_DIR = CLASSIFIER_DIR / "crops"
MANIFEST_PATH = CLASSIFIER_DIR / "manifest.csv"
SPLITS_PATH = CLASSIFIER_DIR / "splits.json"

MIN_CROP_SIZE = 8  # pixels, post-padding/clamping; smaller than this is unusable


def load_raw_annotations() -> dict:
    with open(RAW_ANNOTATIONS) as f:
        return json.load(f)


def pad_and_clamp_box(bbox_xywh, img_width: int, img_height: int, padding: float):
    """COCO xywh -> padded, clamped xyxy. Returns None if the resulting
    box is degenerate (non-positive area or below MIN_CROP_SIZE)."""
    x, y, w, h = bbox_xywh
    if w <= 0 or h <= 0:
        return None

    pad_x = w * padding
    pad_y = h * padding
    x1 = x - pad_x
    y1 = y - pad_y
    x2 = x + w + pad_x
    y2 = y + h + pad_y

    x1 = max(0.0, x1)
    y1 = max(0.0, y1)
    x2 = min(float(img_width), x2)
    y2 = min(float(img_height), y2)

    if (x2 - x1) < MIN_CROP_SIZE or (y2 - y1) < MIN_CROP_SIZE:
        return None

    return [x1, y1, x2, y2]


def assign_splits(images_by_dominant_class: dict, seed: int, ratios=(0.7, 0.15, 0.15)) -> dict:
    """Stratified-by-dominant-class split at the image level. Within each
    dominant-class group, shuffle deterministically and cut at the ratio
    boundaries, then merge groups — keeps class balance close to uniform
    across splits without letting any source image appear in more than
    one split."""
    rng = random.Random(seed)
    splits = {"train": [], "val": [], "test": []}

    for class_name, image_ids in sorted(images_by_dominant_class.items()):
        ids = list(image_ids)
        rng.shuffle(ids)
        n = len(ids)
        n_train = int(round(n * ratios[0]))
        n_val = int(round(n * ratios[1]))
        splits["train"].extend(ids[:n_train])
        splits["val"].extend(ids[n_train : n_train + n_val])
        splits["test"].extend(ids[n_train + n_val :])

    for k in splits:
        splits[k] = sorted(splits[k])
    return splits


def build_crops_and_manifest(padding: float = 0.15, seed: int = 42, ratios=(0.7, 0.15, 0.15)) -> dict:
    """Reads raw COCO annotations, crops+pads+clamps every annotation in
    the 4 kept broad classes, saves crop images to disk, writes the CSV
    manifest, and writes a source-image-level stratified split. Returns a
    summary dict (also useful for the CLI script / audit report)."""
    raw = load_raw_annotations()
    raw_id_to_name = {c["id"]: c["name"] for c in raw["categories"]}
    images_by_id = {img["id"]: img for img in raw["images"]}

    CROPS_DIR.mkdir(parents=True, exist_ok=True)
    for cls in BROAD_CLASSES:
        (CROPS_DIR / cls).mkdir(parents=True, exist_ok=True)

    manifest_rows = []
    rejected = {"other_dropped": 0, "degenerate_box": 0, "missing_image": 0}
    anns_by_image = defaultdict(list)  # for dominant-class computation

    # Group annotations by source image first, so each image is opened and
    # decoded exactly once (annotation-order iteration re-opened/re-decoded
    # the same JPEG once per annotation on it — 275 images / 31.5k
    # annotations means ~114 redundant re-decodes per image on average,
    # which is what made this slow, especially over Drive-mounted I/O).
    anns_grouped_by_image = defaultdict(list)
    for ann in raw["annotations"]:
        anns_grouped_by_image[ann["image_id"]].append(ann)

    for image_id, anns in anns_grouped_by_image.items():
        img_info = images_by_id[image_id]
        image_path = IMAGES_DIR / img_info["file_name"]

        opened_image = None
        if image_path.exists():
            with Image.open(image_path) as im:
                opened_image = im.convert("RGB")

        for ann in anns:
            raw_name = raw_id_to_name[ann["category_id"]]
            mapped_class = RAW_TO_BROAD.get(raw_name)
            if mapped_class is None:
                rejected["other_dropped"] += 1
                continue

            if opened_image is None:
                rejected["missing_image"] += 1
                continue

            box = pad_and_clamp_box(ann["bbox"], img_info["width"], img_info["height"], padding)
            if box is None:
                rejected["degenerate_box"] += 1
                continue

            crop_filename = f"{ann['id']}.jpg"
            crop_path = CROPS_DIR / mapped_class / crop_filename

            crop = opened_image.crop((box[0], box[1], box[2], box[3]))
            crop.save(crop_path, quality=95)

            row = {
                "annotation_id": ann["id"],
                "source_image_id": ann["image_id"],
                "source_image_path": str(image_path.relative_to(DATASET_ROOT.parents[1])),
                "original_category": raw_name,
                "mapped_class": mapped_class,
                "bbox_x1": box[0],
                "bbox_y1": box[1],
                "bbox_x2": box[2],
                "bbox_y2": box[3],
                "crop_path": str(crop_path.relative_to(DATASET_ROOT.parents[1])),
                "split": None,  # filled in below
                "seed": seed,
            }
            manifest_rows.append(row)
            anns_by_image[ann["image_id"]].append(mapped_class)

    # dominant class per source image (for stratified splitting)
    images_by_dominant_class = defaultdict(list)
    for image_id, classes in anns_by_image.items():
        dominant = Counter(classes).most_common(1)[0][0]
        images_by_dominant_class[dominant].append(image_id)

    splits = assign_splits(images_by_dominant_class, seed=seed, ratios=ratios)
    image_id_to_split = {img_id: split for split, ids in splits.items() for img_id in ids}

    for row in manifest_rows:
        row["split"] = image_id_to_split[row["source_image_id"]]

    CLASSIFIER_DIR.mkdir(parents=True, exist_ok=True)
    with open(MANIFEST_PATH, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(manifest_rows[0].keys()))
        writer.writeheader()
        writer.writerows(manifest_rows)

    with open(SPLITS_PATH, "w") as f:
        json.dump({"seed": seed, "ratios": ratios, "padding": padding, "splits": splits}, f, indent=2)

    class_counts_per_split = {
        split: dict(Counter(r["mapped_class"] for r in manifest_rows if r["split"] == split))
        for split in ("train", "val", "test")
    }

    return {
        "num_crops": len(manifest_rows),
        "rejected": rejected,
        "num_images_by_split": {k: len(v) for k, v in splits.items()},
        "class_counts_per_split": class_counts_per_split,
    }


def compute_class_weights(class_counts_train: dict) -> torch.Tensor:
    """Square-root inverse-frequency weights (normalized to mean 1) over
    BROAD_CLASSES order, computed from the training split only.
    weight[c] = 1 / sqrt(train_count[c]), then rescaled so the four
    weights average to 1. Softer than plain inverse frequency — keeps the
    rarest class (deeper_inflammatory_like) upweighted without letting it
    dominate the loss as much as full 1/count would."""
    counts = torch.tensor([max(class_counts_train.get(c, 0), 1) for c in BROAD_CLASSES], dtype=torch.float32)
    weights = 1.0 / torch.sqrt(counts)
    weights = weights * (len(weights) / weights.sum())
    return weights


class AcneSCUCropDataset(Dataset):
    """One cropped lesion image -> broad-class label. Crops are read from
    disk at their saved (padded/clamped) resolution; resizing to the
    model's input size happens in `transform`, not permanently on disk."""

    CLASS_TO_ID = {name: i for i, name in enumerate(BROAD_CLASSES)}

    def __init__(self, split: str, transform=None, manifest_path: Path = MANIFEST_PATH, repo_root: Path = None):
        assert split in ("train", "val", "test")
        self.repo_root = repo_root or DATASET_ROOT.parents[1]
        self.transform = transform

        with open(manifest_path, newline="") as f:
            rows = list(csv.DictReader(f))
        self.rows = [r for r in rows if r["split"] == split]

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx: int):
        row = self.rows[idx]
        crop_path = self.repo_root / row["crop_path"]
        image = Image.open(crop_path).convert("RGB")
        label = self.CLASS_TO_ID[row["mapped_class"]]

        if self.transform is not None:
            image = self.transform(image)

        return image, label, row["annotation_id"], row["source_image_id"]
