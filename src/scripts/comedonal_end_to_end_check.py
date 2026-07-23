"""Targeted comedonal_like end-to-end check: does the detector -> classifier
pipeline actually work for open/closed comedones specifically, not just in
aggregate?

Restricted to the classifier's held-out TEST split images only (unseen by
either model in training) so this isn't inflated by memorization.

A "genuine comedonal detector box" is a detector box (score >= threshold,
same as production) whose region CONTAINS the center of at least one
AcneSCU ground-truth comedonal_like annotation. Containment, not strict
IoU>=0.5, because detector boxes are typically much larger than AcneSCU's
tiny comedone boxes — an IoU>=0.5 requirement would reject genuine hits
by construction (see matched_detections.csv from the earlier pipeline
check: 0 comedonal_like boxes passed strict IoU matching in the whole
41-image test set, which is a matching-protocol artifact, not proof the
detector never finds comedones).

For every such box: classify at 100%/75%/50% crop scale with BOTH
classifier_v1 and classifier_v2, save a visualization grid, and report
aggregate stats. Does not retrain or modify anything.

Usage:
    python3 src/scripts/comedonal_end_to_end_check.py
"""
import argparse
import csv
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from PIL import Image
from torchvision.transforms.functional import to_tensor

from src.data.acnescu_crops import IMAGES_DIR, MIN_CROP_SIZE, pad_and_clamp_box
from src.data.classifier_transforms import get_classifier_transform
from src.inference.pipeline import load_classifier, load_detector, resolve_device

REPO_ROOT = Path(__file__).resolve().parents[2]
DATASET_ROOT = REPO_ROOT / "datasets" / "acnescu"
REMAPPED_ANNOTATIONS = DATASET_ROOT / "remapped_annotations.json"
CLASSIFIER_SPLITS = DATASET_ROOT / "classifier" / "splits.json"

SCALES = [1.0, 0.75, 0.5]


def scaled_crop_box(padded_box, scale, img_w, img_h):
    x1, y1, x2, y2 = padded_box
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    w, h = (x2 - x1) * scale, (y2 - y1) * scale
    nx1, ny1, nx2, ny2 = cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2
    nx1, ny1 = max(0.0, nx1), max(0.0, ny1)
    nx2, ny2 = min(float(img_w), nx2), min(float(img_h), ny2)
    if (nx2 - nx1) < MIN_CROP_SIZE or (ny2 - ny1) < MIN_CROP_SIZE:
        return None
    return [nx1, ny1, nx2, ny2]


def box_center(box):
    return (box[0] + box[2]) / 2, (box[1] + box[3]) / 2


def point_in_box(point, box):
    x, y = point
    return box[0] <= x <= box[2] and box[1] <= y <= box[3]


@torch.no_grad()
def classify(model, class_names, transform, image, box, device):
    crop = image.crop((box[0], box[1], box[2], box[3]))
    tensor = transform(crop).unsqueeze(0).to(device)
    probs = torch.softmax(model(tensor), dim=1)[0]
    pred_idx = int(probs.argmax().item())
    return class_names[pred_idx], float(probs[pred_idx].item()), crop


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--detector-checkpoint", type=str, default="artifacts/detector_v1/best.pth")
    parser.add_argument("--classifier-v1-checkpoint", type=str, default="artifacts/classifier_v1/best.pth")
    parser.add_argument("--classifier-v2-checkpoint", type=str, default="artifacts/classifier_v2/best.pth")
    parser.add_argument("--score-threshold", type=float, default=0.5)
    parser.add_argument("--crop-padding", type=float, default=0.15)
    parser.add_argument("--output-dir", type=str, default="runs/comedonal_end_to_end_check")
    parser.add_argument("--device", type=str, default="auto")
    args = parser.parse_args()

    device = resolve_device(args.device)
    print(f"device: {device}")

    detector, detector_cfg = load_detector(Path(args.detector_checkpoint), device)
    clf_v1, cfg_v1, class_names_v1 = load_classifier(Path(args.classifier_v1_checkpoint), device)
    clf_v2, cfg_v2, class_names_v2 = load_classifier(Path(args.classifier_v2_checkpoint), device)
    assert class_names_v1 == class_names_v2
    class_names = class_names_v1
    transform_v1 = get_classifier_transform(train=False, input_size=cfg_v1.get("input_size", 224))
    transform_v2 = get_classifier_transform(train=False, input_size=cfg_v2.get("input_size", 224))

    with open(REMAPPED_ANNOTATIONS) as f:
        gt_data = json.load(f)
    images_by_id = {img["id"]: img for img in gt_data["images"]}
    comedonal_gt_by_image = {}
    for ann in gt_data["annotations"]:
        if ann["broad_category"] != "comedonal_like":
            continue
        x, y, w, h = ann["bbox"]
        comedonal_gt_by_image.setdefault(ann["image_id"], []).append([x, y, x + w, y + h])

    with open(CLASSIFIER_SPLITS) as f:
        test_image_ids = json.load(f)["splits"]["test"]

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    genuine_boxes = []  # list of dicts: image info, det box/score, num comedonal gt hit

    print(f"\nScanning {len(test_image_ids)} held-out test images for detector boxes over comedonal_like lesions...")
    for image_id in test_image_ids:
        if image_id not in comedonal_gt_by_image:
            continue
        img_info = images_by_id[image_id]
        image_path = IMAGES_DIR / img_info["file_name"]
        image = Image.open(image_path).convert("RGB")
        img_w, img_h = image.size

        with torch.no_grad():
            output = detector([to_tensor(image).to(device)])[0]
        keep = output["scores"] >= args.score_threshold
        boxes = output["boxes"][keep].cpu().tolist()
        scores = output["scores"][keep].cpu().tolist()

        gt_boxes = comedonal_gt_by_image[image_id]

        for box, score in zip(boxes, scores):
            n_hit = sum(1 for gt in gt_boxes if point_in_box(box_center(gt), box))
            if n_hit > 0:
                genuine_boxes.append(
                    {
                        "image_id": image_id,
                        "file_name": img_info["file_name"],
                        "box": box,
                        "score": score,
                        "num_comedonal_gt_hit": n_hit,
                        "img_w": img_w,
                        "img_h": img_h,
                    }
                )

    print(f"Found {len(genuine_boxes)} genuine comedonal detector boxes across the test split.")

    if not genuine_boxes:
        print("No genuine comedonal detector boxes found at this threshold — nothing further to evaluate.")
        return

    # classify each at 100/75/50% with both models
    results = []
    image_cache = {}
    for gb in genuine_boxes:
        if gb["file_name"] not in image_cache:
            image_cache[gb["file_name"]] = Image.open(IMAGES_DIR / gb["file_name"]).convert("RGB")
        image = image_cache[gb["file_name"]]

        x1, y1, x2, y2 = gb["box"]
        bbox_xywh = [x1, y1, x2 - x1, y2 - y1]
        padded = pad_and_clamp_box(bbox_xywh, gb["img_w"], gb["img_h"], args.crop_padding)
        if padded is None:
            continue

        row = dict(gb)
        for scale in SCALES:
            scaled = scaled_crop_box(padded, scale, gb["img_w"], gb["img_h"])
            if scaled is None:
                row[f"v1_{int(scale*100)}"] = (None, None)
                row[f"v2_{int(scale*100)}"] = (None, None)
                continue
            cls1, conf1, _ = classify(clf_v1, class_names, transform_v1, image, scaled, device)
            cls2, conf2, _ = classify(clf_v2, class_names, transform_v2, image, scaled, device)
            row[f"v1_{int(scale*100)}"] = (cls1, conf1)
            row[f"v2_{int(scale*100)}"] = (cls2, conf2)
        results.append(row)

    # visualization grid: one row per box, cols = [context, v1@100, v2@100, v2@75, v2@50]
    n = len(results)
    fig, axes = plt.subplots(n, 5, figsize=(5 * 3, n * 3.2))
    if n == 1:
        axes = axes.reshape(1, -1)

    col_titles = ["context (det box)", "v1 @ 100%", "v2 @ 100%", "v2 @ 75%", "v2 @ 50%"]

    for i, row in enumerate(results):
        image = image_cache[row["file_name"]]
        x1, y1, x2, y2 = row["box"]
        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
        cw, ch = (x2 - x1) * 2.5, (y2 - y1) * 2.5
        ctx_box = [max(0, cx - cw / 2), max(0, cy - ch / 2), min(row["img_w"], cx + cw / 2), min(row["img_h"], cy + ch / 2)]
        ctx_crop = image.crop(tuple(ctx_box))
        axes[i, 0].imshow(ctx_crop)
        axes[i, 0].set_title(f"det score={row['score']:.2f}", fontsize=8)
        axes[i, 0].axis("off")

        bbox_xywh = [x1, y1, x2 - x1, y2 - y1]
        padded = pad_and_clamp_box(bbox_xywh, row["img_w"], row["img_h"], args.crop_padding)

        col_configs = [("v1_100", 1.0), ("v2_100", 1.0), ("v2_75", 0.75), ("v2_50", 0.5)]
        for col_idx, (key, scale) in enumerate(col_configs, start=1):
            cls_name, conf = row[key]
            scaled = scaled_crop_box(padded, scale, row["img_w"], row["img_h"]) if padded else None
            if scaled is None or cls_name is None:
                axes[i, col_idx].axis("off")
                continue
            crop = image.crop((scaled[0], scaled[1], scaled[2], scaled[3]))
            axes[i, col_idx].imshow(crop)
            correct_marker = "OK" if cls_name == "comedonal_like" else "X"
            axes[i, col_idx].set_title(f"{cls_name} ({conf:.2f}) [{correct_marker}]", fontsize=8)
            axes[i, col_idx].axis("off")

    for col_idx, title in enumerate(col_titles):
        axes[0, col_idx].set_title(f"{title}\n" + axes[0, col_idx].get_title(), fontsize=8)

    fig.tight_layout()
    fig.savefig(output_dir / "comedonal_e2e_grid.jpg", dpi=110)
    plt.close(fig)

    # save csv
    csv_rows = []
    for row in results:
        csv_rows.append(
            {
                "image_id": row["image_id"],
                "file_name": row["file_name"],
                "detector_score": row["score"],
                "num_comedonal_gt_hit": row["num_comedonal_gt_hit"],
                "v1_100_class": row["v1_100"][0], "v1_100_conf": row["v1_100"][1],
                "v2_100_class": row["v2_100"][0], "v2_100_conf": row["v2_100"][1],
                "v2_75_class": row["v2_75"][0], "v2_75_conf": row["v2_75"][1],
                "v2_50_class": row["v2_50"][0], "v2_50_conf": row["v2_50"][1],
            }
        )
    with open(output_dir / "comedonal_e2e_results.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(csv_rows[0].keys()))
        writer.writeheader()
        writer.writerows(csv_rows)

    # summary stats
    def pct_correct(key):
        vals = [r[key][0] for r in results if r[key][0] is not None]
        if not vals:
            return 0.0, Counter()
        correct = sum(1 for v in vals if v == "comedonal_like")
        wrong = Counter(v for v in vals if v != "comedonal_like")
        return correct / len(vals), wrong

    print("\n" + "=" * 70)
    print(f"Total genuine comedonal detector boxes evaluated: {len(results)}")
    for key, label in [("v1_100", "v1 @ 100%"), ("v2_100", "v2 @ 100%"), ("v2_75", "v2 @ 75%"), ("v2_50", "v2 @ 50%")]:
        pct, wrong = pct_correct(key)
        wrong_str = ", ".join(f"{c}={n}" for c, n in wrong.most_common())
        print(f"  {label:12s}  comedonal_like predicted: {pct:.1%}   wrong classes: {wrong_str if wrong_str else '(none)'}")
    print("=" * 70)

    summary = {
        "num_genuine_comedonal_boxes": len(results),
        "score_threshold": args.score_threshold,
        "results_by_config": {
            key: {"pct_correct": pct_correct(key)[0], "wrong_class_counts": dict(pct_correct(key)[1])}
            for key in ["v1_100", "v2_100", "v2_75", "v2_50"]
        },
    }
    with open(output_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nSaved: {output_dir / 'comedonal_e2e_grid.jpg'}")
    print(f"Saved: {output_dir / 'comedonal_e2e_results.csv'}")
    print(f"Saved: {output_dir / 'summary.json'}")


if __name__ == "__main__":
    main()
