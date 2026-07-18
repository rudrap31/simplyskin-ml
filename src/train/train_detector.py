"""Main training entrypoint for the ACNE04 lesion detector baseline.

Usage:
    python3 src/train/train_detector.py --config configs/detector_baseline.yaml
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

from src.data.acne04 import Acne04Detection
from src.data.transforms import collate_fn, get_transform
from src.models.detector import build_model
from src.train.engine import evaluate, train_one_epoch


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def resolve_device(requested: str) -> torch.device:
    if requested != "auto":
        return torch.device(requested)
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def build_dataloaders(cfg: dict):
    train_ds = Acne04Detection(
        fold=cfg["fold"], split="train", transforms=get_transform(train=True),
        val_ratio=cfg["val_ratio"], seed=cfg["seed"],
    )
    val_ds = Acne04Detection(
        fold=cfg["fold"], split="val", transforms=get_transform(train=False),
        val_ratio=cfg["val_ratio"], seed=cfg["seed"],
    )

    train_loader = torch.utils.data.DataLoader(
        train_ds, batch_size=cfg["batch_size"], shuffle=True,
        num_workers=cfg["num_workers"], collate_fn=collate_fn,
    )
    val_loader = torch.utils.data.DataLoader(
        val_ds, batch_size=cfg["batch_size"], shuffle=False,
        num_workers=cfg["num_workers"], collate_fn=collate_fn,
    )
    return train_loader, val_loader


def train(cfg: dict, output_dir: Path):
    set_seed(cfg["seed"])
    device = resolve_device(cfg["device"])
    print(f"Using device: {device}")

    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / "config.yaml", "w") as f:
        yaml.safe_dump(cfg, f)

    train_loader, val_loader = build_dataloaders(cfg)
    print(f"train images: {len(train_loader.dataset)}  val images: {len(val_loader.dataset)}")

    model = build_model(cfg["model_name"], num_classes=cfg["num_classes"], pretrained=cfg["pretrained"])
    model.to(device)

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=cfg["lr"], momentum=cfg["momentum"], weight_decay=cfg["weight_decay"])
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=cfg["lr_step_size"], gamma=cfg["lr_gamma"])

    history = []
    best_map50 = -1.0
    best_epoch = -1

    for epoch in range(cfg["epochs"]):
        start = time.time()
        print(f"\nEpoch {epoch + 1}/{cfg['epochs']}")

        train_losses = train_one_epoch(model, optimizer, train_loader, device, log_interval=cfg["log_interval"])
        scheduler.step()

        val_metrics = evaluate(model, val_loader, device, score_threshold=cfg["score_threshold"])

        elapsed = time.time() - start
        record = {
            "epoch": epoch + 1,
            "train_loss": train_losses["total_loss"],
            "train_loss_components": {k: v for k, v in train_losses.items() if k != "total_loss"},
            "val_loss": val_metrics["val_loss"],
            "val_loss_components": val_metrics["val_loss_components"],
            "mAP_50": val_metrics["mAP_50"],
            "mAP_50_95": val_metrics["mAP"],
            "precision": val_metrics["precision"],
            "recall": val_metrics["recall"],
            "lr": scheduler.get_last_lr()[0],
            "elapsed_sec": elapsed,
        }
        history.append(record)

        print(
            f"  train_loss={record['train_loss']:.4f}  val_loss={record['val_loss']:.4f}  "
            f"mAP@0.5={record['mAP_50']:.4f}  mAP@0.5:0.95={record['mAP_50_95']:.4f}  "
            f"precision={record['precision']:.4f}  recall={record['recall']:.4f}  "
            f"({elapsed:.1f}s)"
        )

        with open(output_dir / "history.json", "w") as f:
            json.dump(history, f, indent=2)

        if record["mAP_50"] > best_map50:
            best_map50 = record["mAP_50"]
            best_epoch = epoch + 1
            torch.save(
                {"model_state_dict": model.state_dict(), "epoch": epoch + 1, "mAP_50": best_map50, "config": cfg},
                output_dir / "best.pth",
            )
            print(f"  -> new best (mAP@0.5={best_map50:.4f}), saved best.pth")

    torch.save(
        {"model_state_dict": model.state_dict(), "epoch": cfg["epochs"], "config": cfg},
        output_dir / "final.pth",
    )

    summary = {"best_epoch": best_epoch, "best_mAP_50": best_map50, "total_epochs": cfg["epochs"]}
    with open(output_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nDone. Best epoch {best_epoch} with mAP@0.5={best_map50:.4f}")
    return history, summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/detector_baseline.yaml")
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--epochs", type=int, default=None, help="override epochs from config")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    if args.epochs is not None:
        cfg["epochs"] = args.epochs

    output_dir = Path(args.output_dir) if args.output_dir else Path(cfg["output_dir"])
    train(cfg, output_dir)


if __name__ == "__main__":
    main()
