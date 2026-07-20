"""Lightweight sanity checks for the AcneSCU classifier pipeline. Not a
full test suite — just enough to catch a broken wiring before a real
training run (mirrors the plain assert/print style of
smoke_test_detector.py; no pytest dependency in this repo).

Checks:
  1. COCO annotation parsing (raw categories load, remap is well-formed)
  2. Crop extraction (pad_and_clamp_box degenerate/normal cases)
  3. Split leakage (no source image appears in more than one split)
  4. Model forward pass (correct output shape for all classes)
  5. Checkpoint save/load round-trip

Usage:
    python3 src/scripts/test_classifier_pipeline.py
"""
import csv
import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import torch

from src.data.acnescu import RAW_TO_BROAD, BROAD_CLASSES
from src.data.acnescu_crops import MANIFEST_PATH, SPLITS_PATH, pad_and_clamp_box
from src.models.classifier import build_model


def check(name, condition):
    status = "PASS" if condition else "FAIL"
    print(f"[{status}] {name}")
    if not condition:
        raise AssertionError(name)


def test_coco_parsing():
    print("\n1. COCO annotation parsing")
    from src.data.acnescu import RAW_ANNOTATIONS

    with open(RAW_ANNOTATIONS) as f:
        raw = json.load(f)
    check("raw annotations file has images/annotations/categories keys", all(k in raw for k in ("images", "annotations", "categories")))
    raw_names = {c["name"] for c in raw["categories"]}
    check("all RAW_TO_BROAD keys exist in raw categories", set(RAW_TO_BROAD.keys()).issubset(raw_names))
    check("remap targets are exactly BROAD_CLASSES", set(RAW_TO_BROAD.values()) == set(BROAD_CLASSES))
    check("'other' is intentionally excluded from RAW_TO_BROAD", "other" not in RAW_TO_BROAD)


def test_crop_extraction():
    print("\n2. Crop extraction (pad_and_clamp_box)")
    # normal box, well within image bounds
    box = pad_and_clamp_box([100, 100, 50, 50], img_width=1000, img_height=1000, padding=0.15)
    check("normal box padded correctly", box is not None and abs((box[2] - box[0]) - 50 * 1.3) < 1e-6)

    # box touching the image edge must clamp, not go negative/out-of-bounds
    box = pad_and_clamp_box([0, 0, 20, 20], img_width=1000, img_height=1000, padding=0.15)
    check("edge box clamps to >= 0", box is not None and box[0] == 0.0 and box[1] == 0.0)

    # degenerate (zero-area) box must be rejected
    box = pad_and_clamp_box([10, 10, 0, 5], img_width=1000, img_height=1000, padding=0.15)
    check("zero-width box rejected", box is None)

    # tiny box that pads to still-too-small must be rejected
    box = pad_and_clamp_box([5, 5, 1, 1], img_width=1000, img_height=1000, padding=0.15)
    check("sub-MIN_CROP_SIZE box rejected", box is None)


def test_split_leakage():
    print("\n3. Split leakage")
    if not SPLITS_PATH.exists() or not MANIFEST_PATH.exists():
        print("  SKIPPED: manifest/splits not generated yet (run build_acnescu_crops.py first)")
        return

    with open(SPLITS_PATH) as f:
        splits_data = json.load(f)
    splits = splits_data["splits"]

    train_ids = set(splits["train"])
    val_ids = set(splits["val"])
    test_ids = set(splits["test"])
    check("train/val disjoint", train_ids.isdisjoint(val_ids))
    check("train/test disjoint", train_ids.isdisjoint(test_ids))
    check("val/test disjoint", val_ids.isdisjoint(test_ids))

    with open(MANIFEST_PATH, newline="") as f:
        rows = list(csv.DictReader(f))
    image_id_to_splits = {}
    for row in rows:
        image_id_to_splits.setdefault(row["source_image_id"], set()).add(row["split"])
    multi_split_images = {img_id: s for img_id, s in image_id_to_splits.items() if len(s) > 1}
    check("every source image's crops all land in exactly one split", len(multi_split_images) == 0)


def test_model_forward_pass():
    print("\n4. Model forward pass")
    model = build_model("efficientnet_b0", num_classes=4, pretrained=False)
    model.eval()
    dummy = torch.randn(2, 3, 224, 224)
    with torch.no_grad():
        out = model(dummy)
    check("output shape is (batch, num_classes)", tuple(out.shape) == (2, 4))


def test_checkpoint_roundtrip():
    print("\n5. Checkpoint save/load round-trip")
    model = build_model("efficientnet_b0", num_classes=4, pretrained=False)
    with tempfile.TemporaryDirectory() as tmpdir:
        ckpt_path = Path(tmpdir) / "test.pth"
        torch.save({"model_state_dict": model.state_dict(), "epoch": 1, "class_names": BROAD_CLASSES}, ckpt_path)

        model2 = build_model("efficientnet_b0", num_classes=4, pretrained=False)
        checkpoint = torch.load(ckpt_path, map_location="cpu")
        model2.load_state_dict(checkpoint["model_state_dict"])

        dummy = torch.randn(1, 3, 224, 224)
        model.eval()
        model2.eval()
        with torch.no_grad():
            out1 = model(dummy)
            out2 = model2(dummy)
        check("loaded model produces identical output", torch.allclose(out1, out2))
        check("checkpoint preserves class_names", checkpoint["class_names"] == BROAD_CLASSES)


def main():
    test_coco_parsing()
    test_crop_extraction()
    test_split_leakage()
    test_model_forward_pass()
    test_checkpoint_roundtrip()
    print("\nAll checks passed.")


if __name__ == "__main__":
    main()
