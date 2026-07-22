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
VAL_CONTEXT_CROPS_PATH = CLASSIFIER_DIR / "val_context_crops.json"

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


# classifier_v2: crop-scale augmentation. See src/scripts/crop_scale_diagnostic.py
# for the diagnostic that motivated this — the production pipeline crops
# a detector box (typically wider/looser than AcneSCU's own tight
# annotation boxes), but v1 only ever trained on the tight 15%-padded
# crop. Randomly widening the training crop exposes the classifier to
# the kind of framing a real detector box actually produces.
#
# Scales are relative to the manifest's stored box, which is ALREADY
# 15%-padded relative to the raw annotation (pad_and_clamp_box pads each
# side by w*0.15, so the manifest box is (1 + 2*0.15) = 1.3x the raw
# annotation, not 1.15x — a 1.0x draw here is "tight" only relative to
# the manifest box, not the original annotation). So relative to the raw
# annotation, the three scale choices are approximately:
#   1.0x manifest box -> ~1.3x  raw annotation
#   1.5x manifest box -> ~1.95x raw annotation
#   2.0x manifest box -> ~2.6x  raw annotation
MULTI_SCALE_CHOICES = [1.0, 1.5, 2.0]
MULTI_SCALE_WEIGHTS = [0.25, 0.35, 0.40]  # ~25% tight, ~35% 1.5x, ~40% 2x — default policy, most classes
MULTI_SCALE_SHIFT_FRACTION = 0.15  # max random center shift, as a fraction of box width/height, applied to expanded scales only

# Per-class scale-sampling policy. See
# runs/crop_scale_contamination_audit/contamination_summary.json (ground-
# truth-based: does an expanded crop's region contain a DIFFERENT-class
# lesion's center): at 2.0x, deeper_inflammatory_like crops are
# contaminated 72.3% of the time (vs <8% for every other class), so 2.0x
# is excluded entirely for it and 1.0x is upweighted instead. This is a
# fix to the crop-scale POLICY (contaminated inputs), not a loss-weight
# fix — compute_class_weights is deliberately left untouched.
CLASS_SCALE_POLICY = {
    "comedonal_like": {"scale_choices": MULTI_SCALE_CHOICES, "scale_weights": [0.25, 0.35, 0.40]},
    "inflammatory_like": {"scale_choices": MULTI_SCALE_CHOICES, "scale_weights": [0.25, 0.35, 0.40]},
    "non_active_acne": {"scale_choices": MULTI_SCALE_CHOICES, "scale_weights": [0.25, 0.35, 0.40]},
    "deeper_inflammatory_like": {"scale_choices": [1.0, 1.5, 2.0], "scale_weights": [0.70, 0.30, 0.0]},
}


def get_scale_policy(mapped_class: str):
    policy = CLASS_SCALE_POLICY[mapped_class]
    return policy["scale_choices"], policy["scale_weights"]


def expanded_crop_box(padded_box, img_w: int, img_h: int, scale: float, shift_x_frac: float = 0.0, shift_y_frac: float = 0.0) -> list:
    """Pure, deterministic scale+shift of the manifest's stored box —
    given explicit scale/shift values (no randomness of its own), so both
    the live training augmentation and the fixed validation set can share
    this one implementation while drawing their scale/shift differently
    (global `random` for training, a seeded `random.Random` for the
    reproducible validation crops). Falls back to the original box if the
    result degenerates after clamping to the image (extremely rare)."""
    x1, y1, x2, y2 = padded_box
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    w, h = x2 - x1, y2 - y1

    cx += shift_x_frac * w
    cy += shift_y_frac * h

    new_w, new_h = w * scale, h * scale
    nx1, ny1 = max(0.0, cx - new_w / 2), max(0.0, cy - new_h / 2)
    nx2, ny2 = min(float(img_w), cx + new_w / 2), min(float(img_h), cy + new_h / 2)

    if (nx2 - nx1) < MIN_CROP_SIZE or (ny2 - ny1) < MIN_CROP_SIZE:
        return [x1, y1, x2, y2]
    return [nx1, ny1, nx2, ny2]


def multi_scale_crop_box(padded_box, img_w: int, img_h: int, mapped_class: str) -> list:
    """Training-time augmentation: randomly pick a scale according to
    CLASS_SCALE_POLICY[mapped_class] (default classes: 1.0x p=0.25 / 1.5x
    p=0.35 / 2.0x p=0.40; deeper_inflammatory_like: 1.0x p=0.70 / 1.5x
    p=0.30 / 2.0x p=0.0 — see CLASS_SCALE_POLICY's docstring), with a
    random center shift on expanded scales only (a detector box is rarely
    perfectly centered on the lesion the way a hand-annotated one is).
    Uses the global `random` module — same convention as
    src/data/transforms.py's augmentations — so this is process-level
    random, not seeded per-instance; only the train/val/test split
    assignment and the fixed validation crops need to be reproducible,
    not this draw."""
    scale_choices, scale_weights = get_scale_policy(mapped_class)
    scale = random.choices(scale_choices, weights=scale_weights, k=1)[0]
    shift_x_frac = shift_y_frac = 0.0
    if scale > 1.0:
        shift_x_frac = random.uniform(-MULTI_SCALE_SHIFT_FRACTION, MULTI_SCALE_SHIFT_FRACTION)
        shift_y_frac = random.uniform(-MULTI_SCALE_SHIFT_FRACTION, MULTI_SCALE_SHIFT_FRACTION)
    return expanded_crop_box(padded_box, img_w, img_h, scale, shift_x_frac, shift_y_frac)


def deterministic_context_crop_box(padded_box, img_w: int, img_h: int, rng: random.Random, mapped_class: str) -> tuple:
    """Same class-specific scale/shift distribution as multi_scale_crop_box
    (CLASS_SCALE_POLICY), but driven by a caller-supplied random.Random
    instance instead of the global `random` module, so a given rng
    (seeded per-annotation) always reproduces the identical crop — used to
    build the fixed, deterministic context validation set
    (val_context_crops.json), which must NOT be re-randomized on every
    evaluation. Returns (box, scale) so the chosen scale can be recorded
    for auditing."""
    scale_choices, scale_weights = get_scale_policy(mapped_class)
    scale = rng.choices(scale_choices, weights=scale_weights, k=1)[0]
    shift_x_frac = shift_y_frac = 0.0
    if scale > 1.0:
        shift_x_frac = rng.uniform(-MULTI_SCALE_SHIFT_FRACTION, MULTI_SCALE_SHIFT_FRACTION)
        shift_y_frac = rng.uniform(-MULTI_SCALE_SHIFT_FRACTION, MULTI_SCALE_SHIFT_FRACTION)
    box = expanded_crop_box(padded_box, img_w, img_h, scale, shift_x_frac, shift_y_frac)
    return box, scale


class AcneSCUMultiScaleCropDataset(Dataset):
    """Train-only dataset for classifier_v2: re-crops from the full source
    image at a randomly chosen scale (see multi_scale_crop_box) on every
    __getitem__ call, instead of reading a fixed pre-cropped image from
    disk. val/test must keep using the fixed AcneSCUCropDataset (same
    protocol as v1) so v1-vs-v2 numbers stay comparable."""

    CLASS_TO_ID = AcneSCUCropDataset.CLASS_TO_ID

    def __init__(self, split: str, transform=None, manifest_path: Path = MANIFEST_PATH, repo_root: Path = None):
        assert split == "train", "multi-scale crop augmentation is train-only; use AcneSCUCropDataset for val/test"
        self.repo_root = repo_root or DATASET_ROOT.parents[1]
        self.transform = transform

        with open(manifest_path, newline="") as f:
            rows = list(csv.DictReader(f))
        self.rows = [r for r in rows if r["split"] == split]

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx: int):
        row = self.rows[idx]
        image_path = self.repo_root / row["source_image_path"]
        image = Image.open(image_path).convert("RGB")
        img_w, img_h = image.size

        padded_box = [float(row["bbox_x1"]), float(row["bbox_y1"]), float(row["bbox_x2"]), float(row["bbox_y2"])]
        box = multi_scale_crop_box(padded_box, img_w, img_h, row["mapped_class"])
        crop = image.crop((box[0], box[1], box[2], box[3]))
        label = self.CLASS_TO_ID[row["mapped_class"]]

        if self.transform is not None:
            crop = self.transform(crop)

        return crop, label, row["annotation_id"], row["source_image_id"]


def build_val_context_crops(seed: int = 42, manifest_path: Path = MANIFEST_PATH) -> dict:
    """Precompute ONE fixed, deterministic detector-style ("context") crop
    box per val-split annotation and write it to VAL_CONTEXT_CROPS_PATH.
    This must be generated once and reused — a validation set that gets
    re-randomized on every evaluation call can't be used to track whether
    a model is actually improving epoch to epoch.

    Each annotation gets its own random.Random seeded from `seed` combined
    with its annotation_id, so the result is reproducible independent of
    dict/CSV row ordering and independent of anything drawn for any other
    annotation."""
    with open(manifest_path, newline="") as f:
        rows = list(csv.DictReader(f))
    val_rows = [r for r in rows if r["split"] == "val"]

    # cache image dimensions per source image (opened once each, not once per annotation)
    image_sizes = {}
    entries = []
    scale_counts_per_class = defaultdict(Counter)

    for row in val_rows:
        image_path = DATASET_ROOT.parents[1] / row["source_image_path"]
        if image_path not in image_sizes:
            with Image.open(image_path) as im:
                image_sizes[image_path] = im.size
        img_w, img_h = image_sizes[image_path]

        padded_box = [float(row["bbox_x1"]), float(row["bbox_y1"]), float(row["bbox_x2"]), float(row["bbox_y2"])]
        ann_seed = seed * 1_000_003 + int(row["annotation_id"])  # arithmetic combination, not hash() (str/bytes hashing is randomized per-process by default; int arithmetic is not)
        rng = random.Random(ann_seed)
        box, scale = deterministic_context_crop_box(padded_box, img_w, img_h, rng, row["mapped_class"])

        scale_counts_per_class[row["mapped_class"]][scale] += 1
        entries.append(
            {
                "annotation_id": row["annotation_id"],
                "source_image_id": row["source_image_id"],
                "source_image_path": row["source_image_path"],
                "mapped_class": row["mapped_class"],
                "scale": scale,
                "box": box,
            }
        )

    payload = {
        "seed": seed,
        "class_scale_policy": CLASS_SCALE_POLICY,
        "shift_fraction": MULTI_SCALE_SHIFT_FRACTION,
        "entries": entries,
    }
    with open(VAL_CONTEXT_CROPS_PATH, "w") as f:
        json.dump(payload, f, indent=2)

    achieved_distribution_per_class = {}
    for cls, counts in scale_counts_per_class.items():
        total_cls = sum(counts.values())
        achieved_distribution_per_class[cls] = {str(s): n / total_cls for s, n in counts.items()}

    return {
        "num_entries": len(entries),
        "scale_counts_per_class": {cls: dict(counts) for cls, counts in scale_counts_per_class.items()},
        "achieved_distribution_per_class": achieved_distribution_per_class,
    }


class AcneSCUContextValDataset(Dataset):
    """Fixed, deterministic detector-style validation set: one precomputed
    expanded crop per val annotation, loaded from VAL_CONTEXT_CROPS_PATH
    (see build_val_context_crops). NOT randomized per evaluation — every
    call to __getitem__(i) across every epoch returns the exact same crop,
    so val_context_macro_f1 is comparable epoch to epoch."""

    CLASS_TO_ID = AcneSCUCropDataset.CLASS_TO_ID

    def __init__(self, transform=None, crops_path: Path = VAL_CONTEXT_CROPS_PATH, repo_root: Path = None):
        self.repo_root = repo_root or DATASET_ROOT.parents[1]
        self.transform = transform

        with open(crops_path) as f:
            payload = json.load(f)
        self.entries = payload["entries"]

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, idx: int):
        entry = self.entries[idx]
        image_path = self.repo_root / entry["source_image_path"]
        image = Image.open(image_path).convert("RGB")
        box = entry["box"]
        crop = image.crop((box[0], box[1], box[2], box[3]))
        label = self.CLASS_TO_ID[entry["mapped_class"]]

        if self.transform is not None:
            crop = self.transform(crop)

        return crop, label, entry["annotation_id"], entry["source_image_id"]
