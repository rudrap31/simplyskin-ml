"""Audit the raw AcneSCU COCO annotations before building the classifier
crop pipeline: paths, counts per category, usable-crop counts per kept
class, corrupted/missing images, tiny/invalid boxes, duplicate filenames
or annotation ids.

Usage:
    python3 src/scripts/audit_acnescu.py
"""
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from PIL import Image

from src.data.acnescu import RAW_ANNOTATIONS, RAW_TO_BROAD
from src.data.acnescu_crops import IMAGES_DIR, MIN_CROP_SIZE, pad_and_clamp_box

TINY_BOX_THRESHOLD = 10  # raw (unpadded) box width/height in pixels, flagged as suspicious


def main():
    print(f"Annotations file: {RAW_ANNOTATIONS}")
    print(f"Images dir:       {IMAGES_DIR}")

    with open(RAW_ANNOTATIONS) as f:
        raw = json.load(f)

    images = raw["images"]
    annotations = raw["annotations"]
    categories = {c["id"]: c["name"] for c in raw["categories"]}

    print(f"\nImage count: {len(images)}")
    print(f"Annotation count: {len(annotations)}")

    print("\nAnnotation count per raw category:")
    raw_counts = Counter(categories[a["category_id"]] for a in annotations)
    for name, n in raw_counts.most_common():
        mapped = RAW_TO_BROAD.get(name, "(dropped: other)")
        print(f"  {name:20s} {n:6d}  -> {mapped}")

    # duplicate filenames / annotation ids
    filenames = [img["file_name"] for img in images]
    filename_dupes = {name: n for name, n in Counter(filenames).items() if n > 1}
    ann_ids = [a["id"] for a in annotations]
    ann_id_dupes = {aid: n for aid, n in Counter(ann_ids).items() if n > 1}
    print(f"\nDuplicate image filenames: {len(filename_dupes)}")
    if filename_dupes:
        for name, n in list(filename_dupes.items())[:10]:
            print(f"  {name}: {n} occurrences")
    print(f"Duplicate annotation ids: {len(ann_id_dupes)}")
    if ann_id_dupes:
        for aid, n in list(ann_id_dupes.items())[:10]:
            print(f"  id {aid}: {n} occurrences")

    # missing / corrupted images
    missing = []
    corrupted = []
    dims_by_image_id = {}
    for img in images:
        path = IMAGES_DIR / img["file_name"]
        if not path.exists():
            missing.append(img["file_name"])
            continue
        try:
            with Image.open(path) as im:
                im.verify()
            with Image.open(path) as im:
                dims_by_image_id[img["id"]] = im.size  # (w, h), re-open after verify()
        except Exception as e:
            corrupted.append((img["file_name"], str(e)))

    print(f"\nMissing images: {len(missing)}")
    for name in missing[:10]:
        print(f"  {name}")
    print(f"Corrupted images: {len(corrupted)}")
    for name, err in corrupted[:10]:
        print(f"  {name}: {err}")

    # tiny / invalid raw boxes, and cross-check against the actual
    # COCO-declared image dims vs. what PIL reads (mismatches would
    # silently corrupt padding math downstream)
    tiny_boxes = 0
    invalid_boxes = 0
    dim_mismatches = 0
    usable_after_padding = Counter()
    rejected_after_padding = Counter()

    images_by_id = {img["id"]: img for img in images}
    for ann in annotations:
        raw_name = categories[ann["category_id"]]
        mapped = RAW_TO_BROAD.get(raw_name)
        img_info = images_by_id[ann["image_id"]]
        x, y, w, h = ann["bbox"]

        if w <= 0 or h <= 0:
            invalid_boxes += 1
            continue
        if w < TINY_BOX_THRESHOLD or h < TINY_BOX_THRESHOLD:
            tiny_boxes += 1

        actual_dims = dims_by_image_id.get(ann["image_id"])
        if actual_dims is not None and (actual_dims[0] != img_info["width"] or actual_dims[1] != img_info["height"]):
            dim_mismatches += 1

        if mapped is None:
            continue  # 'other', not part of classifier v1

        box = pad_and_clamp_box(ann["bbox"], img_info["width"], img_info["height"], padding=0.15)
        if box is None:
            rejected_after_padding[mapped] += 1
        else:
            usable_after_padding[mapped] += 1

    print(f"\nInvalid raw boxes (w<=0 or h<=0): {invalid_boxes}")
    print(f"Tiny raw boxes (<{TINY_BOX_THRESHOLD}px on a side): {tiny_boxes}")
    print(f"Image dimension mismatches (COCO json vs. actual file): {dim_mismatches}")

    print(f"\nUsable crops per selected class (padding=0.15, min_crop_size={MIN_CROP_SIZE}px):")
    for cls in sorted(set(usable_after_padding) | set(rejected_after_padding)):
        print(f"  {cls:26s} usable={usable_after_padding[cls]:6d}  rejected={rejected_after_padding[cls]:4d}")

    total_usable = sum(usable_after_padding.values())
    total_rejected = sum(rejected_after_padding.values())
    print(f"\nTotal usable crops (kept classes only): {total_usable}")
    print(f"Total rejected (degenerate after padding/clamping): {total_rejected}")
    print(f"Total 'other' annotations dropped entirely: {raw_counts.get('other', 0)}")


if __name__ == "__main__":
    main()
