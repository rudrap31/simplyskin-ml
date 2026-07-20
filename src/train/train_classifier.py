"""Main training entrypoint for the AcneSCU lesion-crop classifier
(classifier v1: comedonal_like / inflammatory_like / deeper_inflammatory_like
/ non_active_acne). Detector v1 is frozen — this trains an independent
downstream model on cropped lesion images.

Usage:
    python3 src/train/train_classifier.py --config configs/classifier_baseline.yaml
    python3 src/train/train_classifier.py --config configs/classifier_baseline.yaml --resume
"""
import argparse
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.data.acnescu_crops import AcneSCUCropDataset, compute_class_weights
from src.data.classifier_transforms import get_classifier_transform
from src.models.classifier import build_model
from src.train.classifier_engine import evaluate, train_one_epoch
from src.train.classifier_metrics import classification_report_str

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MANIFEST = REPO_ROOT / "datasets" / "acnescu" / "classifier" / "manifest.csv"


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


class Tee:
    def __init__(self, *streams):
        self.streams = streams

    def write(self, data):
        for s in self.streams:
            s.write(data)
            s.flush()

    def flush(self):
        for s in self.streams:
            s.flush()


def resolve_device(requested: str) -> torch.device:
    if requested != "auto":
        return torch.device(requested)
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def build_dataloaders(cfg: dict):
    manifest_path = Path(cfg.get("manifest_path", DEFAULT_MANIFEST))
    input_size = cfg.get("input_size", 224)

    train_ds = AcneSCUCropDataset(
        split="train", transform=get_classifier_transform(train=True, input_size=input_size), manifest_path=manifest_path
    )
    val_ds = AcneSCUCropDataset(
        split="val", transform=get_classifier_transform(train=False, input_size=input_size), manifest_path=manifest_path
    )

    train_loader = torch.utils.data.DataLoader(
        train_ds, batch_size=cfg["batch_size"], shuffle=True, num_workers=cfg["num_workers"]
    )
    val_loader = torch.utils.data.DataLoader(
        val_ds, batch_size=cfg["batch_size"], shuffle=False, num_workers=cfg["num_workers"]
    )
    return train_loader, val_loader


def train(cfg: dict, output_dir: Path, resume: bool = False):
    output_dir.mkdir(parents=True, exist_ok=True)
    log_file = open(output_dir / "train.log", "a")
    sys.stdout = Tee(sys.__stdout__, log_file)
    try:
        return _train(cfg, output_dir, resume=resume)
    finally:
        sys.stdout = sys.__stdout__
        log_file.close()


def _train(cfg: dict, output_dir: Path, resume: bool = False):
    set_seed(cfg["seed"])
    device = resolve_device(cfg["device"])
    print(f"Using device: {device}")

    with open(output_dir / "config.yaml", "w") as f:
        yaml.safe_dump(cfg, f)

    train_loader, val_loader = build_dataloaders(cfg)
    class_names = sorted(AcneSCUCropDataset.CLASS_TO_ID, key=AcneSCUCropDataset.CLASS_TO_ID.get)
    print(f"train crops: {len(train_loader.dataset)}  val crops: {len(val_loader.dataset)}")
    print(f"classes: {class_names}")

    model = build_model(cfg["model_name"], num_classes=cfg["num_classes"], pretrained=cfg["pretrained"])
    model.to(device)

    if cfg.get("class_weighting", True):
        from collections import Counter
        import csv as csv_module

        with open(cfg.get("manifest_path", DEFAULT_MANIFEST), newline="") as f:
            rows = list(csv_module.DictReader(f))
        train_counts = dict(Counter(r["mapped_class"] for r in rows if r["split"] == "train"))
        weights = compute_class_weights(train_counts).to(device)
        print(f"Class weights (train split, inverse-frequency): {dict(zip(class_names, weights.tolist()))}")
        criterion = torch.nn.CrossEntropyLoss(weight=weights)
    else:
        criterion = torch.nn.CrossEntropyLoss()

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=cfg["lr_step_size"], gamma=cfg["lr_gamma"])

    history = []
    best_macro_f1 = -1.0
    best_epoch = -1
    epochs_since_improvement = 0
    start_epoch = 0

    last_ckpt_path = output_dir / "last.pth"
    if resume:
        if not last_ckpt_path.exists():
            raise FileNotFoundError(
                f"--resume was set but no checkpoint found at {last_ckpt_path}. "
                f"Run without --resume to start fresh, or check output_dir."
            )
        print(f"Resuming from {last_ckpt_path}...")
        checkpoint = torch.load(last_ckpt_path, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        start_epoch = checkpoint["epoch"]
        best_macro_f1 = checkpoint["best_macro_f1"]
        best_epoch = checkpoint["best_epoch"]
        epochs_since_improvement = checkpoint.get("epochs_since_improvement", 0)

        history_path = output_dir / "history.json"
        if history_path.exists():
            with open(history_path) as f:
                history = json.load(f)

        print(f"Resumed at epoch {start_epoch} (best so far: epoch {best_epoch}, macro_f1={best_macro_f1:.4f})")

    early_stopping_patience = cfg.get("early_stopping_patience", 8)

    for epoch in range(start_epoch, cfg["epochs"]):
        start = time.time()
        print(f"\nEpoch {epoch + 1}/{cfg['epochs']}")

        train_metrics = train_one_epoch(
            model, optimizer, train_loader, device, criterion, log_interval=cfg["log_interval"]
        )
        scheduler.step()

        val_metrics = evaluate(model, val_loader, device, criterion, class_names)

        elapsed = time.time() - start
        record = {
            "epoch": epoch + 1,
            "train_loss": train_metrics["loss"],
            "val_loss": val_metrics["loss"],
            "val_accuracy": val_metrics["accuracy"],
            "val_macro_precision": val_metrics["macro_precision"],
            "val_macro_recall": val_metrics["macro_recall"],
            "val_macro_f1": val_metrics["macro_f1"],
            "val_per_class_f1": val_metrics["per_class_f1"],
            "lr": scheduler.get_last_lr()[0],
            "elapsed_sec": elapsed,
        }
        history.append(record)

        print(
            f"  train_loss={record['train_loss']:.4f}  val_loss={record['val_loss']:.4f}  "
            f"val_acc={record['val_accuracy']:.4f}  val_macro_f1={record['val_macro_f1']:.4f}  "
            f"({elapsed:.1f}s)"
        )

        with open(output_dir / "history.json", "w") as f:
            json.dump(history, f, indent=2)

        if record["val_macro_f1"] > best_macro_f1:
            best_macro_f1 = record["val_macro_f1"]
            best_epoch = epoch + 1
            epochs_since_improvement = 0
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "epoch": epoch + 1,
                    "val_macro_f1": best_macro_f1,
                    "config": cfg,
                    "class_names": class_names,
                },
                output_dir / "best.pth",
            )
            print(f"  -> new best (val_macro_f1={best_macro_f1:.4f}), saved best.pth")
        else:
            epochs_since_improvement += 1

        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "epoch": epoch + 1,
                "best_macro_f1": best_macro_f1,
                "best_epoch": best_epoch,
                "epochs_since_improvement": epochs_since_improvement,
                "config": cfg,
                "class_names": class_names,
            },
            last_ckpt_path,
        )

        if epochs_since_improvement >= early_stopping_patience:
            print(
                f"\nEarly stopping: no val_macro_f1 improvement for {early_stopping_patience} epochs "
                f"(best epoch {best_epoch}, val_macro_f1={best_macro_f1:.4f})."
            )
            break

    torch.save(
        {"model_state_dict": model.state_dict(), "epoch": epoch + 1, "config": cfg, "class_names": class_names},
        output_dir / "final.pth",
    )

    summary = {"best_epoch": best_epoch, "best_val_macro_f1": best_macro_f1, "total_epochs_run": epoch + 1}
    with open(output_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nDone. Best epoch {best_epoch} with val_macro_f1={best_macro_f1:.4f}")
    print("\nFinal-epoch validation report:")
    print(classification_report_str(val_metrics))
    return history, summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/classifier_baseline.yaml")
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--epochs", type=int, default=None, help="override epochs from config")
    parser.add_argument("--manifest-path", type=str, default=None, help="override manifest_path from config")
    parser.add_argument("--resume", action="store_true", help="resume from last.pth in output_dir if present")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    if args.epochs is not None:
        cfg["epochs"] = args.epochs
    if args.manifest_path is not None:
        cfg["manifest_path"] = args.manifest_path

    output_dir = Path(args.output_dir) if args.output_dir else Path(cfg["output_dir"])
    train(cfg, output_dir, resume=args.resume)


if __name__ == "__main__":
    main()
