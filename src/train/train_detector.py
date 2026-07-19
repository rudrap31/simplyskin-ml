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


class Tee:
    """Mirrors writes to multiple streams — used so training progress
    printed to stdout also lands in a log file under output_dir (Colab
    doesn't persist cell output to Drive on its own)."""

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
    min_size = cfg.get("min_size")
    max_size = cfg.get("max_size")
    data_root = cfg.get("data_root")

    train_ds = Acne04Detection(
        fold=cfg["fold"], split="train", transforms=get_transform(train=True, min_size=min_size, max_size=max_size),
        val_ratio=cfg["val_ratio"], seed=cfg["seed"], data_root=data_root,
    )
    val_ds = Acne04Detection(
        fold=cfg["fold"], split="val", transforms=get_transform(train=False, min_size=min_size, max_size=max_size),
        val_ratio=cfg["val_ratio"], seed=cfg["seed"], data_root=data_root,
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
    if cfg.get("mixed_precision", False) and device.type != "cuda":
        print("Note: mixed_precision=true has no effect on this device (only CUDA is supported); running in full precision.")

    with open(output_dir / "config.yaml", "w") as f:
        yaml.safe_dump(cfg, f)

    train_loader, val_loader = build_dataloaders(cfg)
    print(f"train images: {len(train_loader.dataset)}  val images: {len(val_loader.dataset)}")

    model = build_model(
        cfg["model_name"],
        num_classes=cfg["num_classes"],
        pretrained=cfg["pretrained"],
        min_size=cfg.get("min_size"),
        max_size=cfg.get("max_size"),
    )
    model.to(device)

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=cfg["lr"], momentum=cfg["momentum"], weight_decay=cfg["weight_decay"])
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=cfg["lr_step_size"], gamma=cfg["lr_gamma"])

    use_amp = cfg.get("mixed_precision", False) and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    history = []
    best_map50 = -1.0
    best_epoch = -1
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
        scaler.load_state_dict(checkpoint["scaler_state_dict"])
        start_epoch = checkpoint["epoch"]  # epoch was already completed, so resume after it
        best_map50 = checkpoint["best_map50"]
        best_epoch = checkpoint["best_epoch"]

        history_path = output_dir / "history.json"
        if history_path.exists():
            with open(history_path) as f:
                history = json.load(f)

        print(f"Resumed at epoch {start_epoch} (best so far: epoch {best_epoch}, mAP@0.5={best_map50:.4f})")

    for epoch in range(start_epoch, cfg["epochs"]):
        start = time.time()
        print(f"\nEpoch {epoch + 1}/{cfg['epochs']}")

        train_losses = train_one_epoch(
            model, optimizer, train_loader, device, scaler,
            log_interval=cfg["log_interval"], mixed_precision=cfg.get("mixed_precision", False),
        )
        scheduler.step()

        val_metrics = evaluate(
            model, val_loader, device,
            score_threshold=cfg["score_threshold"], mixed_precision=cfg.get("mixed_precision", False),
        )

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

        # full state (not just weights) so a disconnected Colab session can
        # pick back up mid-run instead of restarting from pretrained weights
        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "scaler_state_dict": scaler.state_dict(),
                "epoch": epoch + 1,
                "best_map50": best_map50,
                "best_epoch": best_epoch,
                "config": cfg,
            },
            last_ckpt_path,
        )

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
    parser.add_argument("--resume", action="store_true", help="resume from last.pth in output_dir if present")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    if args.epochs is not None:
        cfg["epochs"] = args.epochs

    output_dir = Path(args.output_dir) if args.output_dir else Path(cfg["output_dir"])
    train(cfg, output_dir, resume=args.resume)


if __name__ == "__main__":
    main()
