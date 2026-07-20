"""Sweep confidence thresholds on the full validation set to find the
precision/recall/F1 operating point, reusing the exact matching logic
training uses (src.train.metrics.compute_precision_recall: IoU>=0.5,
greedy one-to-one matching in descending confidence order).

Runs inference once (unfiltered predictions), then re-scores at each
threshold without re-running the model.

Usage:
    python3 src/scripts/threshold_analysis.py --checkpoint best.pth
"""
import argparse
import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

from src.data.acne04 import Acne04Detection
from src.data.transforms import get_transform
from src.models.detector import build_model
from src.train.metrics import compute_precision_recall

THRESHOLDS = [0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.70, 0.80, 0.90]


def resolve_device(requested: str) -> torch.device:
    if requested != "auto":
        return torch.device(requested)
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def run_inference(model, val_ds, device):
    """One unfiltered pass over the val set — every predicted box at every
    score, so we can re-threshold in-memory instead of re-running the model
    14 times."""
    all_predictions, all_targets = [], []
    with torch.no_grad():
        for i in range(len(val_ds)):
            image, target = val_ds[i]
            output = model([image.to(device)])[0]
            all_predictions.append(
                {"boxes": output["boxes"].cpu(), "labels": output["labels"].cpu(), "scores": output["scores"].cpu()}
            )
            all_targets.append({"boxes": target["boxes"], "labels": target["labels"]})
            if (i + 1) % 50 == 0 or i == len(val_ds) - 1:
                print(f"  inference {i + 1}/{len(val_ds)}")
    return all_predictions, all_targets


def sweep_thresholds(all_predictions, all_targets, thresholds):
    rows = []
    n_images = len(all_predictions)
    for t in thresholds:
        metrics = compute_precision_recall(all_predictions, all_targets, iou_threshold=0.5, score_threshold=t)
        precision, recall = metrics["precision"], metrics["recall"]
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

        n_preds = sum((pred["scores"] >= t).sum().item() for pred in all_predictions)
        avg_preds_per_image = n_preds / n_images if n_images > 0 else 0.0

        rows.append(
            {
                "threshold": t,
                "tp": metrics["tp"],
                "fp": metrics["fp"],
                "fn": metrics["fn"],
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "avg_preds_per_image": avg_preds_per_image,
            }
        )
    return rows


def save_csv(rows, path: Path):
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def plot_precision_recall(rows, path: Path):
    rows_sorted = sorted(rows, key=lambda r: r["threshold"])
    precisions = [r["precision"] for r in rows_sorted]
    recalls = [r["recall"] for r in rows_sorted]

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot(recalls, precisions, marker="o", color="#1f77b4")
    for r in rows_sorted:
        ax.annotate(f"{r['threshold']:.2f}", (r["recall"], r["precision"]), fontsize=7, xytext=(4, 4), textcoords="offset points")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision vs. Recall across confidence thresholds")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_threshold_curves(rows, path: Path):
    rows_sorted = sorted(rows, key=lambda r: r["threshold"])
    thresholds = [r["threshold"] for r in rows_sorted]
    precisions = [r["precision"] for r in rows_sorted]
    recalls = [r["recall"] for r in rows_sorted]
    f1s = [r["f1"] for r in rows_sorted]

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(thresholds, precisions, marker="o", label="Precision", color="#1f77b4")
    ax.plot(thresholds, recalls, marker="o", label="Recall", color="#d62728")
    ax.plot(thresholds, f1s, marker="o", label="F1", color="#2ca02c")
    ax.set_xlabel("Confidence threshold")
    ax.set_ylabel("Score")
    ax.set_title("Precision / Recall / F1 vs. confidence threshold")
    ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default="best.pth")
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
    print(f"  training config's score_threshold (used for training-time precision/recall logging): {cfg.get('score_threshold')}")

    device = resolve_device(args.device)
    print(f"device: {device}")

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
    print(f"validation split (reconstructed, FULL set): {len(val_ds)} images")

    model = build_model(
        cfg["model_name"],
        num_classes=cfg["num_classes"],
        pretrained=False,
        min_size=cfg.get("min_size"),
        max_size=cfg.get("max_size"),
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    output_dir = Path(args.output_dir) if args.output_dir else Path("runs/detector_colab_baseline/threshold_analysis")
    output_dir.mkdir(parents=True, exist_ok=True)

    print("\nRunning inference once over the full validation set...")
    all_predictions, all_targets = run_inference(model, val_ds, device)

    print("\nSweeping thresholds (re-scoring in-memory, no re-inference)...")
    rows = sweep_thresholds(all_predictions, all_targets, THRESHOLDS)

    csv_path = output_dir / "threshold_sweep.csv"
    save_csv(rows, csv_path)

    pr_plot_path = output_dir / "precision_recall_plot.png"
    plot_precision_recall(rows, pr_plot_path)

    curves_plot_path = output_dir / "threshold_vs_metrics_plot.png"
    plot_threshold_curves(rows, curves_plot_path)

    best_f1_row = max(rows, key=lambda r: r["f1"])
    recall_80_candidates = [r for r in rows if r["recall"] >= 0.80]
    best_recall_80_row = max(recall_80_candidates, key=lambda r: r["threshold"]) if recall_80_candidates else None

    training_threshold = cfg.get("score_threshold")
    training_row = next((r for r in rows if abs(r["threshold"] - training_threshold) < 1e-9), None)

    print("\n" + "=" * 90)
    print(f"{'threshold':>9} {'tp':>5} {'fp':>5} {'fn':>5} {'precision':>10} {'recall':>8} {'f1':>7} {'avg_preds/img':>14}")
    for r in sorted(rows, key=lambda r: r["threshold"]):
        marker = " <- training score_threshold" if training_row is not None and r is training_row else ""
        print(
            f"{r['threshold']:>9.2f} {r['tp']:>5} {r['fp']:>5} {r['fn']:>5} "
            f"{r['precision']:>10.4f} {r['recall']:>8.4f} {r['f1']:>7.4f} {r['avg_preds_per_image']:>14.2f}{marker}"
        )
    print("=" * 90)

    print(f"\nBest F1: threshold={best_f1_row['threshold']:.2f}  "
          f"(precision={best_f1_row['precision']:.4f}, recall={best_f1_row['recall']:.4f}, f1={best_f1_row['f1']:.4f})")

    if best_recall_80_row is not None:
        print(f"Highest threshold retaining >=80% recall: threshold={best_recall_80_row['threshold']:.2f}  "
              f"(recall={best_recall_80_row['recall']:.4f}, precision={best_recall_80_row['precision']:.4f})")
    else:
        max_recall_row = max(rows, key=lambda r: r["recall"])
        print(f"No swept threshold retains >=80% recall. "
              f"Highest recall achieved is {max_recall_row['recall']:.4f} at threshold={max_recall_row['threshold']:.2f}.")

    if training_row is not None:
        print(f"\nAt threshold={training_threshold:.2f} (matches training's score_threshold), this full-val-set sweep gives:")
        print(f"  precision={training_row['precision']:.4f}  recall={training_row['recall']:.4f}")
        print("This uses the identical compute_precision_recall() function training's evaluate() calls each epoch,")
        print("on the identical reconstructed validation split — so if this doesn't match the number you recall")
        print("from the Colab training log, the likely causes are: (a) that recalled number was from an earlier/")
        print("different epoch than this checkpoint (best.pth is epoch 9), not a different threshold or IoU rule;")
        print("or (b) it's simply a different threshold than 0.5 that was being read off at the time.")
    else:
        print(f"\n(training config's score_threshold={training_threshold} isn't in the swept threshold list, "
              f"so no direct row to compare against above.)")

    summary = {
        "checkpoint": str(ckpt_path),
        "checkpoint_epoch": checkpoint.get("epoch"),
        "val_set_size": len(val_ds),
        "training_score_threshold": training_threshold,
        "training_score_threshold_row": training_row,
        "best_f1": best_f1_row,
        "best_threshold_with_recall_at_least_0.80": best_recall_80_row,
        "all_rows": rows,
    }
    with open(output_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nSaved:\n  {csv_path}\n  {pr_plot_path}\n  {curves_plot_path}\n  {output_dir / 'summary.json'}")


if __name__ == "__main__":
    main()
