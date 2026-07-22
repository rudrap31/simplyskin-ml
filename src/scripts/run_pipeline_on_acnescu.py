"""End-to-end sanity check: run the full frozen pipeline (detector v1 ->
crop -> classifier v1) on real AcneSCU face images and compare against
AcneSCU's own ground-truth boxes/broad-classes.

This is NOT a new eval benchmark for either model individually — the
detector already has its ACNE04 test-set numbers, the classifier already
has its AcneSCU test-set numbers on pre-cropped lesions. This checks
something neither of those measures: does the *chained* pipeline (feeding
the detector's own imperfect boxes into the classifier, on faces the
detector has never seen a face-photo distribution like) still work
sensibly end to end.

Restricted to the classifier's held-out test-split images (41 images) so
neither model has seen these AcneSCU images during its own training.

Usage:
    python3 src/scripts/run_pipeline_on_acnescu.py \
        --detector-checkpoint artifacts/detector_v1/best.pth \
        --classifier-checkpoint artifacts/classifier_v1/best.pth
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from PIL import Image, ImageDraw, ImageFont
from torchvision.ops import box_iou

from src.data.acnescu import BROAD_CLASSES
from src.data.acnescu_crops import IMAGES_DIR
from src.inference.pipeline import CLASS_NAMES, load_classifier, load_detector, resolve_device, run_pipeline_on_image
from src.train.classifier_metrics import classification_report_str, compute_classification_metrics

REPO_ROOT = Path(__file__).resolve().parents[2]
DATASET_ROOT = REPO_ROOT / "datasets" / "acnescu"
REMAPPED_ANNOTATIONS = DATASET_ROOT / "remapped_annotations.json"
CLASSIFIER_SPLITS = DATASET_ROOT / "classifier" / "splits.json"

GT_COLOR = (0, 220, 0)
PRED_COLOR = (255, 140, 0)

CLASS_LETTER = {
    "comedonal_like": "C",
    "inflammatory_like": "I",
    "deeper_inflammatory_like": "D",
    "non_active_acne": "N",
}

_FONT_CANDIDATES = [
    "/System/Library/Fonts/Supplemental/Arial Bold.ttf",
    "/System/Library/Fonts/Helvetica.ttc",
    "/Library/Fonts/Arial Bold.ttf",
]


def get_font(size: int):
    for path in _FONT_CANDIDATES:
        if Path(path).exists():
            try:
                return ImageFont.truetype(path, size)
            except Exception:
                continue
    return ImageFont.load_default()


def greedy_match(pred_boxes, pred_scores, gt_boxes, iou_threshold=0.5):
    """Single-class greedy one-to-one IoU matching, descending confidence
    order (same convention as src.train.metrics.compute_precision_recall,
    reimplemented here because we also need to know *which* gt each pred
    matched, to compare predicted class vs. gt class)."""
    order = sorted(range(len(pred_scores)), key=lambda i: -pred_scores[i])
    matched_gt = set()
    matches = []  # (pred_idx, gt_idx)
    unmatched_pred_idx = []

    ious = None
    if len(pred_boxes) > 0 and len(gt_boxes) > 0:
        ious = box_iou(torch.tensor(pred_boxes), torch.tensor(gt_boxes))

    for i in order:
        if ious is None:
            unmatched_pred_idx.append(i)
            continue
        row = ious[i].clone()
        for gt_idx in matched_gt:
            row[gt_idx] = 0.0
        best_iou, best_j = row.max(dim=0)
        if best_iou.item() >= iou_threshold:
            matched_gt.add(best_j.item())
            matches.append((i, best_j.item()))
        else:
            unmatched_pred_idx.append(i)

    unmatched_gt_idx = [j for j in range(len(gt_boxes)) if j not in matched_gt]
    return matches, unmatched_pred_idx, unmatched_gt_idx


def draw_pipeline_result(image, gt_boxes, gt_classes, detections, max_size=1200):
    """Returns (clean_resized_image, annotated_image) — same scale, so
    they can be placed side by side for comparison.

    GT boxes (green) are drawn without per-box text — AcneSCU's boxes are
    often tiny and packed densely, so per-box labels there are unreadable
    noise. Predicted boxes (orange) get a single large bold letter code
    instead of the full class name, with a legend explaining the codes."""
    clean = image.copy()
    scale = min(1.0, max_size / max(clean.size))
    if scale < 1.0:
        clean = clean.resize((round(clean.width * scale), round(clean.height * scale)), Image.BILINEAR)

    annotated = clean.copy()
    draw = ImageDraw.Draw(annotated)
    for box in gt_boxes:
        x1, y1, x2, y2 = [c * scale for c in box]
        draw.rectangle([x1, y1, x2, y2], outline=GT_COLOR, width=2)

    letter_font = get_font(28)
    for det in detections:
        x1, y1, x2, y2 = [c * scale for c in det["box"]]
        draw.rectangle([x1, y1, x2, y2], outline=PRED_COLOR, width=3)
        if det.get("classified"):
            letter = CLASS_LETTER[det["predicted_class"]]
            text_pos = (x1 + 2, max(0, y1 - 32))
            text_bbox = draw.textbbox(text_pos, letter, font=letter_font)
            draw.rectangle(text_bbox, fill=PRED_COLOR)
            draw.text(text_pos, letter, fill=(0, 0, 0), font=letter_font)

    legend_font = get_font(22)
    legend_lines = [
        "GREEN = AcneSCU ground truth box (no label, too dense to caption)",
        "ORANGE = detector + classifier prediction:",
        "  C = comedonal_like   I = inflammatory_like",
        "  D = deeper_inflammatory_like   N = non_active_acne",
    ]
    pad = 6
    line_heights = [draw.textbbox((0, 0), line, font=legend_font)[3] for line in legend_lines]
    box_h = sum(line_heights) + pad * (len(legend_lines) + 1)
    box_w = max(draw.textbbox((0, 0), line, font=legend_font)[2] for line in legend_lines) + pad * 2
    draw.rectangle([0, 0, box_w, box_h], fill=(255, 255, 255))
    y = pad
    for line, lh in zip(legend_lines, line_heights):
        draw.text((pad, y), line, fill=(0, 0, 0), font=legend_font)
        y += lh + pad

    return clean, annotated


def make_side_by_side(clean_image, annotated_image, gap: int = 8):
    w, h = clean_image.size
    combined = Image.new("RGB", (w * 2 + gap, h), color=(255, 255, 255))
    combined.paste(clean_image, (0, 0))
    combined.paste(annotated_image, (w + gap, 0))
    return combined


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--detector-checkpoint", type=str, default="artifacts/detector_v1/best.pth")
    parser.add_argument("--classifier-checkpoint", type=str, default="artifacts/classifier_v1/best.pth")
    parser.add_argument("--score-threshold", type=float, default=0.5, help="detector confidence threshold")
    parser.add_argument("--iou-threshold", type=float, default=0.5)
    parser.add_argument("--num-visualizations", type=int, default=20)
    parser.add_argument("--output-dir", type=str, default="runs/pipeline_v1_acnescu_check")
    parser.add_argument("--device", type=str, default="auto")
    args = parser.parse_args()

    device = resolve_device(args.device)
    print(f"device: {device}")

    detector, detector_cfg = load_detector(Path(args.detector_checkpoint), device)
    classifier, classifier_cfg, classifier_class_names = load_classifier(Path(args.classifier_checkpoint), device)
    print(f"Loaded detector: {args.detector_checkpoint}  (score_threshold={args.score_threshold})")
    print(f"Loaded classifier: {args.classifier_checkpoint}  classes={classifier_class_names}")

    with open(REMAPPED_ANNOTATIONS) as f:
        gt_data = json.load(f)
    images_by_id = {img["id"]: img for img in gt_data["images"]}
    gt_anns_by_image = {}
    for ann in gt_data["annotations"]:
        gt_anns_by_image.setdefault(ann["image_id"], []).append(ann)

    with open(CLASSIFIER_SPLITS) as f:
        splits_data = json.load(f)
    splits = splits_data["splits"]
    crop_padding = splits_data["padding"]  # the exact padding used when the classifier's training crops were generated
    test_image_ids = splits["test"]
    print(f"Running on {len(test_image_ids)} held-out test images (classifier's test split)")
    print(f"crop_padding={crop_padding} (read from classifier/splits.json, matches classifier training crops)")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    total_tp, total_fp, total_fn = 0, 0, 0
    classified_preds, classified_gt = [], []  # for classification-conditional metrics
    per_image_rows = []
    class_to_id = {name: i for i, name in enumerate(CLASS_NAMES)}

    for count, image_id in enumerate(test_image_ids):
        img_info = images_by_id[image_id]
        image_path = IMAGES_DIR / img_info["file_name"]
        image = Image.open(image_path).convert("RGB")

        gt_anns = gt_anns_by_image.get(image_id, [])
        gt_boxes = [[a["bbox"][0], a["bbox"][1], a["bbox"][0] + a["bbox"][2], a["bbox"][1] + a["bbox"][3]] for a in gt_anns]
        gt_classes = [a["broad_category"] for a in gt_anns]

        detections = run_pipeline_on_image(
            image, detector, classifier, classifier_class_names, device,
            detector_score_threshold=args.score_threshold,
            crop_padding=crop_padding,
            classifier_input_size=classifier_cfg.get("input_size", 224),
        )

        pred_boxes = [d["box"] for d in detections]
        pred_scores = [d["detector_score"] for d in detections]
        matches, unmatched_pred_idx, unmatched_gt_idx = greedy_match(pred_boxes, pred_scores, gt_boxes, args.iou_threshold)

        total_tp += len(matches)
        total_fp += len(unmatched_pred_idx)
        total_fn += len(unmatched_gt_idx)

        for pred_idx, gt_idx in matches:
            det = detections[pred_idx]
            if not det["classified"]:
                continue  # degenerate crop, no class prediction to compare
            classified_preds.append(class_to_id[det["predicted_class"]])
            classified_gt.append(class_to_id[gt_classes[gt_idx]])
            per_image_rows.append(
                {
                    "image_id": image_id,
                    "file_name": img_info["file_name"],
                    "gt_class": gt_classes[gt_idx],
                    "predicted_class": det["predicted_class"],
                    "confidence": det["class_confidence"],
                    "detector_score": det["detector_score"],
                    "correct": det["predicted_class"] == gt_classes[gt_idx],
                }
            )

        if count < args.num_visualizations:
            clean, annotated = draw_pipeline_result(image, gt_boxes, gt_classes, detections)
            side_by_side = make_side_by_side(clean, annotated)
            side_by_side.save(output_dir / f"{img_info['file_name']}")

        print(f"  [{count + 1}/{len(test_image_ids)}] {img_info['file_name']}: gt={len(gt_boxes)} det={len(detections)} matched={len(matches)}")

    detection_precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
    detection_recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
    detection_f1 = (
        2 * detection_precision * detection_recall / (detection_precision + detection_recall)
        if (detection_precision + detection_recall) > 0
        else 0.0
    )

    print("\n" + "=" * 60)
    print("DETECTION STAGE (class-agnostic, detector v1 on AcneSCU faces — OOD for the detector)")
    print(f"  tp={total_tp}  fp={total_fp}  fn={total_fn}")
    print(f"  precision={detection_precision:.4f}  recall={detection_recall:.4f}  f1={detection_f1:.4f}")

    classification_metrics = None
    if classified_preds:
        preds_tensor = torch.tensor(classified_preds)
        labels_tensor = torch.tensor(classified_gt)
        classification_metrics = compute_classification_metrics(preds_tensor, labels_tensor, CLASS_NAMES)

        print("\nCLASSIFICATION STAGE (conditioned on matched detections only, i.e. tp boxes)")
        print(f"  n_matched_and_classified={len(classified_preds)}")
        print(classification_report_str(classification_metrics))
    else:
        print("\nNo matched detections were classified — cannot compute classification-conditional metrics.")
    print("=" * 60)

    summary = {
        "detector_checkpoint": args.detector_checkpoint,
        "classifier_checkpoint": args.classifier_checkpoint,
        "score_threshold": args.score_threshold,
        "iou_threshold": args.iou_threshold,
        "num_test_images": len(test_image_ids),
        "detection": {
            "tp": total_tp,
            "fp": total_fp,
            "fn": total_fn,
            "precision": detection_precision,
            "recall": detection_recall,
            "f1": detection_f1,
        },
        "classification_conditioned_on_tp": classification_metrics,
    }
    with open(output_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    import csv

    with open(output_dir / "matched_detections.csv", "w", newline="") as f:
        if per_image_rows:
            writer = csv.DictWriter(f, fieldnames=list(per_image_rows[0].keys()))
            writer.writeheader()
            writer.writerows(per_image_rows)

    print(f"\nSaved: {output_dir / 'summary.json'}")
    print(f"Saved: {output_dir / 'matched_detections.csv'}")
    print(f"Saved {min(args.num_visualizations, len(test_image_ids))} annotated images -> {output_dir}")


if __name__ == "__main__":
    main()
