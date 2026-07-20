"""Generate fixed-seed visual validation results for a trained ACNE04 detector.

Reconstructs the exact validation split used during training (fold/seed/
val_ratio come from the checkpoint's embedded config, not a possibly-drifted
YAML file), runs the model, and saves ground-truth vs. predicted boxes
(with confidence scores) for a fixed sample of validation images.

Usage:
    python3 src/scripts/visualize_val_predictions.py \
        --checkpoint best.pth --num-images 25 --score-threshold 0.50
"""
import argparse
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import torch
from PIL import Image, ImageDraw
from torchvision.transforms.functional import to_pil_image

from src.data.acne04 import Acne04Detection
from src.data.transforms import get_transform
from src.models.detector import build_model
from src.train.metrics import compute_map, compute_precision_recall

GT_COLOR = (0, 220, 0)
PRED_COLOR = (255, 140, 0)


def draw_validation_image(image_pil, gt_boxes, pred_boxes, pred_scores, max_size: int = 1200) -> Image.Image:
    """Ground truth (green) + predictions with score labels (orange), drawn
    in a single pass so gt/pred boxes share one consistent scale factor."""
    image = image_pil.copy()
    scale = min(1.0, max_size / max(image.size))
    if scale < 1.0:
        new_size = (round(image.width * scale), round(image.height * scale))
        image = image.resize(new_size, Image.BILINEAR)

    draw = ImageDraw.Draw(image)
    for box in gt_boxes:
        x1, y1, x2, y2 = [c * scale for c in box]
        draw.rectangle([x1, y1, x2, y2], outline=GT_COLOR, width=3)

    for box, score in zip(pred_boxes, pred_scores):
        x1, y1, x2, y2 = [c * scale for c in box]
        draw.rectangle([x1, y1, x2, y2], outline=PRED_COLOR, width=3)
        draw.text((x1, max(0, y1 - 12)), f"{score:.2f}", fill=PRED_COLOR)

    return image


def resolve_device(requested: str) -> torch.device:
    if requested != "auto":
        return torch.device(requested)
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default="best.pth")
    parser.add_argument("--num-images", type=int, default=25)
    parser.add_argument("--seed", type=int, default=42, help="fixed seed for which val images get sampled")
    parser.add_argument("--score-threshold", type=float, default=0.50)
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--device", type=str, default="auto")
    args = parser.parse_args()

    ckpt_path = Path(args.checkpoint)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found at {ckpt_path}")

    checkpoint = torch.load(ckpt_path, map_location="cpu")
    cfg = checkpoint["config"]
    print(f"Loaded checkpoint: {ckpt_path}")
    print(f"  epoch={checkpoint.get('epoch')}  mAP@0.5={checkpoint.get('mAP_50'):.4f} (as recorded during training)")
    print(f"  training config (embedded in checkpoint): {cfg}")

    device = resolve_device(args.device)
    print(f"device: {device}")

    # data_root in the checkpoint's config is wherever training actually ran
    # (e.g. a Colab-local path) — fall back to the repo-default dataset
    # location if that path doesn't exist on this machine.
    data_root = cfg.get("data_root")
    if data_root and not Path(data_root).exists():
        print(f"Note: data_root '{data_root}' from checkpoint doesn't exist here; using repo-default datasets/acne04 instead.")
        data_root = None

    val_ds = Acne04Detection(
        fold=cfg["fold"],
        split="val",
        transforms=get_transform(train=False, min_size=cfg.get("min_size"), max_size=cfg.get("max_size")),
        val_ratio=cfg["val_ratio"],
        seed=cfg["seed"],
        data_root=data_root,
    )
    print(f"validation split (reconstructed): {len(val_ds)} images")

    n = min(args.num_images, len(val_ds))
    rng = random.Random(args.seed)
    indices = sorted(rng.sample(range(len(val_ds)), n))

    model = build_model(
        cfg["model_name"],
        num_classes=cfg["num_classes"],
        pretrained=False,  # we're loading trained weights next; no need to fetch COCO weights first
        min_size=cfg.get("min_size"),
        max_size=cfg.get("max_size"),
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    output_dir = (
        Path(args.output_dir)
        if args.output_dir
        else Path("runs/detector_colab_baseline/visualizations") / f"threshold_{args.score_threshold:.2f}"
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"output_dir: {output_dir}")

    per_image_results = []
    all_predictions, all_targets = [], []

    with torch.no_grad():
        for idx in indices:
            image, target = val_ds[idx]
            output = model([image.to(device)])[0]

            keep = output["scores"] >= args.score_threshold
            pred_boxes = output["boxes"][keep].cpu().tolist()
            pred_scores = output["scores"][keep].cpu().tolist()
            gt_boxes = target["boxes"].tolist()

            pil_image = to_pil_image(image)
            annotated = draw_validation_image(pil_image, gt_boxes, pred_boxes, pred_scores)
            out_path = output_dir / f"{target['image_id']}.jpg"
            annotated.save(out_path)

            per_image_results.append(
                {"image_id": target["image_id"], "gt_count": len(gt_boxes), "pred_count": len(pred_boxes)}
            )
            print(f"  {target['image_id']}: gt={len(gt_boxes)}  pred={len(pred_boxes)}  -> {out_path.name}")

            all_predictions.append(
                {"boxes": output["boxes"].cpu(), "labels": output["labels"].cpu(), "scores": output["scores"].cpu()}
            )
            all_targets.append({"boxes": target["boxes"], "labels": target["labels"]})

    map_metrics = compute_map(all_predictions, all_targets)
    pr_metrics = compute_precision_recall(all_predictions, all_targets, score_threshold=args.score_threshold)

    summary = {
        "checkpoint": str(ckpt_path),
        "checkpoint_epoch": checkpoint.get("epoch"),
        "num_images": n,
        "seed": args.seed,
        "score_threshold": args.score_threshold,
        **map_metrics,
        **pr_metrics,
        "per_image": per_image_results,
    }
    with open(output_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nMetrics over these {n} sampled images (NOT the full val set):")
    print(
        f"  mAP@0.5={map_metrics['mAP_50']:.4f}  mAP@0.5:0.95={map_metrics['mAP']:.4f}  "
        f"precision={pr_metrics['precision']:.4f}  recall={pr_metrics['recall']:.4f}"
    )
    print(f"\nSaved {n} annotated images + summary.json -> {output_dir}")


if __name__ == "__main__":
    main()
