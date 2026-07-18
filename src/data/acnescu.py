"""AcneSCU dataset: COCO-format lesion annotations, remapped to broad classes.

Raw AcneSCU has 10 fine-grained categories mixing active acne lesions with
scars/pigmentation/moles. We remap down to 5 classes per the locked v1 scope:
  - comedonal_like          <- closed_comedo, open_comedo
  - inflammatory_like       <- papule, pustule
  - deeper_inflammatory_like <- nodule
  - non_active_acne         <- atrophic_scar, hypertrophic_scar, melasma, nevus
  - (dropped)               <- other (too ambiguous to keep as a class)
"""
import json
import random
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import Dataset

DATASET_ROOT = Path(__file__).resolve().parents[2] / "datasets" / "acnescu"
RAW_ANNOTATIONS = DATASET_ROOT / "train" / "_annotations.coco.json"
IMAGES_DIR = DATASET_ROOT / "train"
REMAPPED_ANNOTATIONS = DATASET_ROOT / "remapped_annotations.json"
SPLITS_FILE = DATASET_ROOT / "splits.json"

RAW_TO_BROAD = {
    "closed_comedo": "comedonal_like",
    "open_comedo": "comedonal_like",
    "papule": "inflammatory_like",
    "pustule": "inflammatory_like",
    "nodule": "deeper_inflammatory_like",
    "atrophic_scar": "non_active_acne",
    "hypertrophic_scar": "non_active_acne",
    "melasma": "non_active_acne",
    "nevus": "non_active_acne",
    # "other" is intentionally omitted -> dropped
}

BROAD_CLASSES = [
    "comedonal_like",
    "inflammatory_like",
    "deeper_inflammatory_like",
    "non_active_acne",
]
# class 0 is background/unused; detection-style class ids start at 1
BROAD_CLASS_TO_ID = {name: i + 1 for i, name in enumerate(BROAD_CLASSES)}


def build_remapped_annotations() -> dict:
    """Load the raw COCO json, drop 'other', remap categories to broad
    classes, and write the result to REMAPPED_ANNOTATIONS."""
    with open(RAW_ANNOTATIONS) as f:
        raw = json.load(f)

    raw_id_to_name = {c["id"]: c["name"] for c in raw["categories"]}

    kept_annotations = []
    dropped = 0
    for ann in raw["annotations"]:
        raw_name = raw_id_to_name[ann["category_id"]]
        broad_name = RAW_TO_BROAD.get(raw_name)
        if broad_name is None:
            dropped += 1
            continue
        new_ann = dict(ann)
        new_ann["broad_category"] = broad_name
        new_ann["broad_category_id"] = BROAD_CLASS_TO_ID[broad_name]
        kept_annotations.append(new_ann)

    remapped = {
        "images": raw["images"],
        "annotations": kept_annotations,
        "categories": [
            {"id": BROAD_CLASS_TO_ID[name], "name": name} for name in BROAD_CLASSES
        ],
        "dropped_other_count": dropped,
    }

    with open(REMAPPED_ANNOTATIONS, "w") as f:
        json.dump(remapped, f, indent=2)

    return remapped


def build_splits(seed: int = 42, ratios=(0.7, 0.15, 0.15)) -> dict:
    """Split by image_id (each AcneSCU image is one distinct photo, no
    stated multi-photo-per-subject relationship) into train/val/test."""
    with open(REMAPPED_ANNOTATIONS) as f:
        data = json.load(f)

    image_ids = [img["id"] for img in data["images"]]
    rng = random.Random(seed)
    rng.shuffle(image_ids)

    n = len(image_ids)
    n_train = int(n * ratios[0])
    n_val = int(n * ratios[1])

    splits = {
        "train": sorted(image_ids[:n_train]),
        "val": sorted(image_ids[n_train : n_train + n_val]),
        "test": sorted(image_ids[n_train + n_val :]),
    }

    with open(SPLITS_FILE, "w") as f:
        json.dump(splits, f, indent=2)

    return splits


class AcneSCUDetection(Dataset):
    """Lesion detection/classification dataset with broad-class labels.

    Returns (image, target) where target['boxes'] is (N,4) xyxy and
    target['labels'] are broad-class ids (see BROAD_CLASS_TO_ID).
    """

    def __init__(self, split: str = "train", transforms=None):
        assert split in ("train", "val", "test")
        with open(REMAPPED_ANNOTATIONS) as f:
            data = json.load(f)
        with open(SPLITS_FILE) as f:
            splits = json.load(f)

        split_ids = set(splits[split])
        self.images_by_id = {
            img["id"]: img for img in data["images"] if img["id"] in split_ids
        }
        self.image_ids = sorted(self.images_by_id.keys())

        self.anns_by_image = {img_id: [] for img_id in self.image_ids}
        for ann in data["annotations"]:
            if ann["image_id"] in self.anns_by_image:
                self.anns_by_image[ann["image_id"]].append(ann)

        self.transforms = transforms

    def __len__(self) -> int:
        return len(self.image_ids)

    def __getitem__(self, idx: int):
        image_id = self.image_ids[idx]
        img_info = self.images_by_id[image_id]
        image_path = IMAGES_DIR / img_info["file_name"]
        image = Image.open(image_path).convert("RGB")

        boxes = []
        labels = []
        for ann in self.anns_by_image[image_id]:
            x, y, w, h = ann["bbox"]  # COCO format: xywh
            boxes.append([x, y, x + w, y + h])
            labels.append(ann["broad_category_id"])

        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        if boxes.numel() == 0:
            boxes = boxes.reshape(0, 4)
        labels = torch.as_tensor(labels, dtype=torch.int64)

        target = {
            "boxes": boxes,
            "labels": labels,
            "image_id": image_id,
            "file_name": img_info["file_name"],
        }

        if self.transforms is not None:
            image, target = self.transforms(image, target)

        return image, target
