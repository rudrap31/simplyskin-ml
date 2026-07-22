"""Part 2 diagnostic: is the classifier's prediction on detector-generated
crops sensitive to crop scale? For every kept detector box on a sample of
AcneSCU face images, generate 3 center-crops at 100% / 75% / 50% of the
box actually used by the production pipeline (the 15%-padded box —
"100%" here means exactly what the live pipeline crops today, not the
raw unpadded detector box), classify each, and visualize.

Question this answers: does tightening the crop toward the lesion's
center recover comedonal_like/inflammatory_like predictions, or does the
classifier keep predicting deeper_inflammatory_like/non_active_acne
regardless of scale? If predictions shift meaningfully as scale shrinks,
that's strong evidence crop scale (not a code bug, not the classifier
itself) is the primary driver of the pipeline's end-to-end mismatch.

Usage:
    python3 src/scripts/crop_scale_diagnostic.py --num-images 25
"""
import argparse
import csv
import json
import random
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
CLASSIFIER_SPLITS = DATASET_ROOT / "classifier" / "splits.json"

SCALES = [1.0, 0.75, 0.5]  # relative to the production (15%-padded) box


def scaled_crop_box(padded_box, scale, img_w, img_h):
    """Center-crop the padded box down to `scale` of its size, same
    center. Shrinking toward the center of an already in-bounds box stays
    in-bounds by construction, but we still clamp/reject defensively."""
    x1, y1, x2, y2 = padded_box
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    w, h = (x2 - x1) * scale, (y2 - y1) * scale
    nx1, ny1, nx2, ny2 = cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2
    nx1, ny1 = max(0.0, nx1), max(0.0, ny1)
    nx2, ny2 = min(float(img_w), nx2), min(float(img_h), ny2)
    if (nx2 - nx1) < MIN_CROP_SIZE or (ny2 - ny1) < MIN_CROP_SIZE:
        return None
    return [nx1, ny1, nx2, ny2]


@torch.no_grad()
def classify_crop(image, box, classifier, class_names, transform, device):
    crop = image.crop((box[0], box[1], box[2], box[3]))
    tensor = transform(crop).unsqueeze(0).to(device)
    probs = torch.softmax(classifier(tensor), dim=1)[0]
    pred_idx = int(probs.argmax().item())
    return class_names[pred_idx], float(probs[pred_idx].item()), crop


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--detector-checkpoint", type=str, default="artifacts/detector_v1/best.pth")
    parser.add_argument("--classifier-checkpoint", type=str, default="artifacts/classifier_v1/best.pth")
    parser.add_argument("--score-threshold", type=float, default=0.5)
    parser.add_argument("--crop-padding", type=float, default=0.15)
    parser.add_argument("--num-images", type=int, default=25)
    parser.add_argument("--max-boxes-per-image", type=int, default=6, help="highest-confidence boxes visualized per image")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=str, default="runs/crop_scale_diagnostic")
    parser.add_argument("--device", type=str, default="auto")
    args = parser.parse_args()

    device = resolve_device(args.device)
    print(f"device: {device}")

    detector, detector_cfg = load_detector(Path(args.detector_checkpoint), device)
    classifier, classifier_cfg, class_names = load_classifier(Path(args.classifier_checkpoint), device)
    transform = get_classifier_transform(train=False, input_size=classifier_cfg.get("input_size", 224))
    print(f"classes: {class_names}")

    with open(CLASSIFIER_SPLITS) as f:
        splits = json.load(f)["splits"]
    test_image_ids = splits["test"]

    with open(DATASET_ROOT / "remapped_annotations.json") as f:
        gt_data = json.load(f)
    images_by_id = {img["id"]: img for img in gt_data["images"]}

    rng = random.Random(args.seed)
    n = min(args.num_images, len(test_image_ids))
    sample_ids = sorted(rng.sample(test_image_ids, n))
    print(f"Running on {n} images (seed={args.seed}) from the classifier's test split")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    csv_rows = []
    class_counts_by_scale = {s: Counter() for s in SCALES}

    for count, image_id in enumerate(sample_ids):
        img_info = images_by_id[image_id]
        image_path = IMAGES_DIR / img_info["file_name"]
        image = Image.open(image_path).convert("RGB")
        img_w, img_h = image.size

        with torch.no_grad():
            output = detector([to_tensor(image).to(device)])[0]
        keep = output["scores"] >= args.score_threshold
        boxes = output["boxes"][keep].cpu().tolist()
        scores = output["scores"][keep].cpu().tolist()

        order = sorted(range(len(scores)), key=lambda i: -scores[i])[: args.max_boxes_per_image]

        n_boxes = len(order)
        if n_boxes == 0:
            print(f"  [{count + 1}/{n}] {img_info['file_name']}: no detections above threshold, skipping")
            continue

        fig, axes = plt.subplots(n_boxes, 1 + len(SCALES), figsize=((1 + len(SCALES)) * 3, n_boxes * 3.2))
        if n_boxes == 1:
            axes = axes.reshape(1, -1)

        for row, box_idx in enumerate(order):
            box = boxes[box_idx]
            score = scores[box_idx]
            bbox_xywh = [box[0], box[1], box[2] - box[0], box[3] - box[1]]
            padded = pad_and_clamp_box(bbox_xywh, img_w, img_h, args.crop_padding)

            # context panel: a wider region around the box for orientation
            ctx_pad = 2.5
            cx, cy = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
            cw, ch = (box[2] - box[0]) * ctx_pad, (box[3] - box[1]) * ctx_pad
            ctx_box = [max(0, cx - cw / 2), max(0, cy - ch / 2), min(img_w, cx + cw / 2), min(img_h, cy + ch / 2)]
            context_crop = image.crop(tuple(ctx_box))
            axes[row, 0].imshow(context_crop)
            axes[row, 0].set_title(f"det score={score:.2f}", fontsize=9)
            axes[row, 0].axis("off")

            if padded is None:
                for col in range(1, 1 + len(SCALES)):
                    axes[row, col].axis("off")
                continue

            for col, scale in enumerate(SCALES, start=1):
                scaled_box = scaled_crop_box(padded, scale, img_w, img_h)
                if scaled_box is None:
                    axes[row, col].axis("off")
                    continue
                pred_class, conf, crop_img = classify_crop(image, scaled_box, classifier, class_names, transform, device)
                axes[row, col].imshow(crop_img)
                axes[row, col].set_title(f"{int(scale * 100)}%: {pred_class}\n({conf:.2f})", fontsize=8)
                axes[row, col].axis("off")

                class_counts_by_scale[scale][pred_class] += 1
                csv_rows.append(
                    {
                        "image_id": image_id,
                        "file_name": img_info["file_name"],
                        "box_index": box_idx,
                        "detector_score": score,
                        "scale": scale,
                        "predicted_class": pred_class,
                        "confidence": conf,
                    }
                )

        fig.suptitle(img_info["file_name"], fontsize=10)
        fig.tight_layout()
        fig.savefig(output_dir / f"{Path(img_info['file_name']).stem}_scales.jpg", dpi=110)
        plt.close(fig)

        print(f"  [{count + 1}/{n}] {img_info['file_name']}: {n_boxes} boxes visualized")

    with open(output_dir / "crop_scale_results.csv", "w", newline="") as f:
        if csv_rows:
            writer = csv.DictWriter(f, fieldnames=list(csv_rows[0].keys()))
            writer.writeheader()
            writer.writerows(csv_rows)

    print("\n" + "=" * 60)
    print("Predicted-class distribution by crop scale:")
    for scale in SCALES:
        total = sum(class_counts_by_scale[scale].values())
        print(f"\n  scale={int(scale * 100)}%  (n={total})")
        for cls, n_cls in class_counts_by_scale[scale].most_common():
            print(f"    {cls:26s} {n_cls:5d}  ({n_cls / total:.1%})" if total else f"    {cls}: 0")
    print("=" * 60)

    summary = {
        "num_images": n,
        "score_threshold": args.score_threshold,
        "crop_padding": args.crop_padding,
        "scales": SCALES,
        "class_counts_by_scale": {str(s): dict(c) for s, c in class_counts_by_scale.items()},
    }
    with open(output_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nSaved: {output_dir / 'crop_scale_results.csv'}")
    print(f"Saved: {output_dir / 'summary.json'}")
    print(f"Saved per-image visualizations -> {output_dir}")


if __name__ == "__main__":
    main()
