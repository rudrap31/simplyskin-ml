"""ACNE04 dataset: Pascal VOC lesion detection annotations."""
import random
import xml.etree.ElementTree as ET
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import Dataset

DATASET_ROOT = Path(__file__).resolve().parents[2] / "datasets" / "acne04"
IMAGES_DIR = DATASET_ROOT / "Classification" / "JPEGImages"
ANNOTATIONS_DIR = DATASET_ROOT / "Detection" / "VOC2007" / "Annotations"
SPLITS_DIR = DATASET_ROOT / "Detection" / "VOC2007" / "ImageSets" / "Main"

# ACNE04 boxes are a single generic lesion class ("fore"). Background is 0
# so the detector's foreground class id is 1.
LESION_CLASS_ID = 1


def parse_voc_annotation(xml_path: Path) -> dict:
    """Parse one VOC XML file into image size + a list of lesion boxes."""
    tree = ET.parse(xml_path)
    root = tree.getroot()

    size = root.find("size")
    width = int(size.find("width").text)
    height = int(size.find("height").text)

    boxes = []
    for obj in root.findall("object"):
        bnd = obj.find("bndbox")
        boxes.append(
            [
                float(bnd.find("xmin").text),
                float(bnd.find("ymin").text),
                float(bnd.find("xmax").text),
                float(bnd.find("ymax").text),
            ]
        )

    return {"width": width, "height": height, "boxes": boxes}


def load_split_ids(fold: int, split: str) -> list[str]:
    """Load image ids (no extension) for a given fold (0-4) and split.

    split is 'trainval' or 'test' — these are ACNE04's official files.
    """
    assert split in ("trainval", "test")
    split_file = SPLITS_DIR / f"NNEW_{split}_{fold}.txt"
    ids = []
    with open(split_file) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            # lines look like: "levle0_0.jpg  0  3"
            ids.append(line.split()[0].removesuffix(".jpg"))
    return ids


def split_trainval(fold: int, val_ratio: float = 0.15, seed: int = 42) -> tuple[list[str], list[str]]:
    """Deterministically split the official trainval ids into train/val.

    The official test fold is never touched here — this only carves an
    internal validation set out of trainval for model selection.
    """
    ids = load_split_ids(fold, "trainval")
    ids = sorted(ids)  # sort first so shuffle result is independent of filesystem order
    rng = random.Random(seed)
    rng.shuffle(ids)

    n_val = int(len(ids) * val_ratio)
    val_ids = sorted(ids[:n_val])
    train_ids = sorted(ids[n_val:])
    return train_ids, val_ids


class Acne04Detection(Dataset):
    """Lesion detection dataset. Returns (image, target) pairs.

    target is a dict with 'boxes' (N,4 xyxy float tensor), 'labels' (N,
    all LESION_CLASS_ID), and 'image_id'/'severity'/'lesion_count' for
    bookkeeping — matches the format torchvision detection models expect.

    split: 'train' and 'val' are carved out of the official trainval file
    (see split_trainval); 'test' is the official, untouched held-out fold.
    """

    def __init__(
        self,
        fold: int = 0,
        split: str = "train",
        transforms=None,
        val_ratio: float = 0.15,
        seed: int = 42,
    ):
        assert split in ("train", "val", "test")
        if split == "test":
            self.ids = load_split_ids(fold, "test")
        else:
            train_ids, val_ids = split_trainval(fold, val_ratio=val_ratio, seed=seed)
            self.ids = train_ids if split == "train" else val_ids
        self.transforms = transforms

    def __len__(self) -> int:
        return len(self.ids)

    def __getitem__(self, idx: int):
        image_id = self.ids[idx]
        image_path = IMAGES_DIR / f"{image_id}.jpg"
        ann_path = ANNOTATIONS_DIR / f"{image_id}.xml"

        image = Image.open(image_path).convert("RGB")
        ann = parse_voc_annotation(ann_path)

        boxes = torch.as_tensor(ann["boxes"], dtype=torch.float32)
        if boxes.numel() == 0:
            boxes = boxes.reshape(0, 4)
        labels = torch.full((boxes.shape[0],), LESION_CLASS_ID, dtype=torch.int64)

        severity = int(image_id.split("_")[0].removeprefix("levle"))

        target = {
            "boxes": boxes,
            "labels": labels,
            "image_id": image_id,
            "severity": severity,
            "lesion_count": boxes.shape[0],
        }

        if self.transforms is not None:
            image, target = self.transforms(image, target)

        return image, target
