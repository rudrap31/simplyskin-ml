"""Test-set evaluation for the AcneSCU lesion-crop classifier. Loads the
best checkpoint (selected by validation macro F1 during training) and
reports full test metrics + a visualization of misclassified crops.

Usage:
    python3 src/scripts/evaluate_classifier.py --checkpoint runs/classifier_baseline/best.pth
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

from src.data.acnescu_crops import AcneSCUContextValDataset, AcneSCUCropDataset, DATASET_ROOT
from src.data.classifier_transforms import get_classifier_transform
from src.models.classifier import build_model
from src.train.classifier_metrics import classification_report_str, compute_classification_metrics

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MANIFEST = REPO_ROOT / "datasets" / "acnescu" / "classifier" / "manifest.csv"


def resolve_device(requested: str) -> torch.device:
    if requested != "auto":
        return torch.device(requested)
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def plot_confusion_matrix(cm, class_names, path: Path):
    import numpy as np

    cm = np.array(cm)
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(class_names)))
    ax.set_yticks(range(len(class_names)))
    ax.set_xticklabels(class_names, rotation=45, ha="right", fontsize=8)
    ax.set_yticklabels(class_names, fontsize=8)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Confusion matrix (test set)")
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center", fontsize=8, color="black")
    fig.colorbar(im)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def save_misclassified_grid(misclassified, output_dir: Path, max_images: int = 25):
    """misclassified: list of (crop_path, true_class, pred_class, confidence, source_image_id)."""
    from PIL import Image

    n = min(len(misclassified), max_images)
    if n == 0:
        return
    cols = 5
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3, rows * 3.4))
    axes = axes.flatten() if n > 1 else [axes]

    for i in range(n):
        crop_path, true_cls, pred_cls, conf, source_img_id = misclassified[i]
        img = Image.open(REPO_ROOT / crop_path).convert("RGB")
        axes[i].imshow(img)
        axes[i].set_title(
            f"true: {true_cls}\npred: {pred_cls} ({conf:.2f})\nsrc: {source_img_id}", fontsize=7
        )
        axes[i].axis("off")

    for i in range(n, len(axes)):
        axes[i].axis("off")

    fig.tight_layout()
    fig.savefig(output_dir / "misclassified_examples.jpg", dpi=150)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default="runs/classifier_baseline/best.pth")
    parser.add_argument("--manifest-path", type=str, default=str(DEFAULT_MANIFEST))
    parser.add_argument(
        "--eval-set", type=str, default="test", choices=["test", "context_val"],
        help="'test': the original held-out tight test split (AcneSCUCropDataset). "
             "'context_val': the fixed, deterministic detector-style crops (AcneSCUContextValDataset) — NOT the held-out test set.",
    )
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--max-misclassified", type=int, default=25)
    args = parser.parse_args()

    ckpt_path = Path(args.checkpoint)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found at {ckpt_path}")

    checkpoint = torch.load(ckpt_path, map_location="cpu")
    cfg = checkpoint["config"]
    class_names = checkpoint["class_names"]
    print(f"Loaded checkpoint: {ckpt_path}")
    print(f"  epoch={checkpoint.get('epoch')}  val_macro_f1={checkpoint.get('val_macro_f1')}")
    print(f"  classes: {class_names}")
    print(f"  eval_set: {args.eval_set}")

    device = resolve_device(args.device)
    print(f"device: {device}")

    if args.eval_set == "test":
        test_ds = AcneSCUCropDataset(
            split="test",
            transform=get_classifier_transform(train=False, input_size=cfg.get("input_size", 224)),
            manifest_path=Path(args.manifest_path),
        )
    else:
        test_ds = AcneSCUContextValDataset(
            transform=get_classifier_transform(train=False, input_size=cfg.get("input_size", 224)),
        )
    print(f"{args.eval_set} crops: {len(test_ds)}")

    test_loader = torch.utils.data.DataLoader(test_ds, batch_size=cfg["batch_size"], shuffle=False, num_workers=0)

    model = build_model(cfg["model_name"], num_classes=cfg["num_classes"], pretrained=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    default_dirname = "test_evaluation" if args.eval_set == "test" else "context_val_evaluation"
    output_dir = Path(args.output_dir) if args.output_dir else ckpt_path.parent / default_dirname
    output_dir.mkdir(parents=True, exist_ok=True)

    all_preds, all_labels, all_confidences = [], [], []
    misclassified = []

    with torch.no_grad():
        for images, labels, ann_ids, source_img_ids in test_loader:
            images = images.to(device)
            outputs = model(images)
            probs = torch.softmax(outputs, dim=1)
            confs, preds = probs.max(dim=1)

            all_preds.append(preds.cpu())
            all_labels.append(labels)
            all_confidences.append(confs.cpu())

            for i in range(len(labels)):
                if preds[i].item() != labels[i].item() and args.eval_set == "test":
                    # context_val has no static crop_path (its crops are
                    # computed on the fly from source_image + box), so the
                    # misclassified-crop grid is test-set-only
                    ann_id = ann_ids[i]
                    row = next(r for r in test_ds.rows if r["annotation_id"] == ann_id)
                    misclassified.append(
                        (
                            row["crop_path"],
                            class_names[labels[i].item()],
                            class_names[preds[i].item()],
                            confs[i].item(),
                            source_img_ids[i],
                        )
                    )

    all_preds = torch.cat(all_preds)
    all_labels = torch.cat(all_labels)
    num_misclassified = (all_preds != all_labels).sum().item()

    metrics = compute_classification_metrics(all_preds, all_labels, class_names)

    print("\n" + "=" * 60)
    print(f"{args.eval_set} accuracy:       {metrics['accuracy']:.4f}")
    print(f"{args.eval_set} macro precision: {metrics['macro_precision']:.4f}")
    print(f"{args.eval_set} macro recall:    {metrics['macro_recall']:.4f}")
    print(f"{args.eval_set} macro F1:        {metrics['macro_f1']:.4f}")
    print("=" * 60)
    report = classification_report_str(metrics)
    print(report)

    with open(output_dir / "classification_report.txt", "w") as f:
        f.write(report)

    with open(output_dir / "test_metrics.json", "w") as f:
        json.dump(
            {
                "checkpoint": str(ckpt_path),
                "eval_set": args.eval_set,
                "num_crops": len(test_ds),
                "num_misclassified": num_misclassified,
                **{k: v for k, v in metrics.items() if k != "confusion_matrix"},
                "confusion_matrix": metrics["confusion_matrix"],
            },
            f,
            indent=2,
        )

    plot_confusion_matrix(metrics["confusion_matrix"], class_names, output_dir / "confusion_matrix.png")
    if args.eval_set == "test":
        save_misclassified_grid(misclassified, output_dir, max_images=args.max_misclassified)

    print(f"\nMisclassified: {num_misclassified}/{len(test_ds)}")
    print(f"Saved: {output_dir / 'classification_report.txt'}")
    print(f"Saved: {output_dir / 'test_metrics.json'}")
    print(f"Saved: {output_dir / 'confusion_matrix.png'}")
    if args.eval_set == "test" and misclassified:
        print(f"Saved: {output_dir / 'misclassified_examples.jpg'}")


if __name__ == "__main__":
    main()
