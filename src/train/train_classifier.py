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

from src.data.acnescu_crops import (
    AcneSCUContextValDataset,
    AcneSCUCropDataset,
    AcneSCUMultiScaleCropDataset,
    CLASS_SCALE_POLICY,
    VAL_CONTEXT_CROPS_PATH,
    compute_class_weights,
)
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

    # "multi_scale" (classifier_v2): train crops are randomly re-cropped
    # from the full source image at tight/1.5x/2x scale each epoch, to
    # resemble real detector-box framing. val/test always stay on the
    # fixed AcneSCUCropDataset crop, matching v1's protocol exactly, so
    # v1-vs-v2 numbers on the held-out set stay directly comparable.
    train_augmentation = cfg.get("train_crop_augmentation", "fixed")
    if train_augmentation == "multi_scale":
        train_ds = AcneSCUMultiScaleCropDataset(
            split="train", transform=get_classifier_transform(train=True, input_size=input_size), manifest_path=manifest_path
        )
    elif train_augmentation == "fixed":
        train_ds = AcneSCUCropDataset(
            split="train", transform=get_classifier_transform(train=True, input_size=input_size), manifest_path=manifest_path
        )
    else:
        raise ValueError(f"Unknown train_crop_augmentation '{train_augmentation}', expected 'fixed' or 'multi_scale'")

    val_ds = AcneSCUCropDataset(
        split="val", transform=get_classifier_transform(train=False, input_size=input_size), manifest_path=manifest_path
    )

    train_loader = torch.utils.data.DataLoader(
        train_ds, batch_size=cfg["batch_size"], shuffle=True, num_workers=cfg["num_workers"]
    )
    val_loader = torch.utils.data.DataLoader(
        val_ds, batch_size=cfg["batch_size"], shuffle=False, num_workers=cfg["num_workers"]
    )

    # classifier_v2: a second, FIXED validation set of detector-style
    # ("context") crops — precomputed once by build_val_context_crops.py,
    # never re-randomized per evaluation. Opt-in via use_context_validation
    # so v1's config/behavior is completely unaffected.
    val_context_loader = None
    if cfg.get("use_context_validation", False):
        if not VAL_CONTEXT_CROPS_PATH.exists():
            raise FileNotFoundError(
                f"use_context_validation=true but {VAL_CONTEXT_CROPS_PATH} doesn't exist. "
                f"Run `python3 src/scripts/build_val_context_crops.py` first."
            )
        val_context_ds = AcneSCUContextValDataset(transform=get_classifier_transform(train=False, input_size=input_size))
        val_context_loader = torch.utils.data.DataLoader(
            val_context_ds, batch_size=cfg["batch_size"], shuffle=False, num_workers=cfg["num_workers"]
        )

    return train_loader, val_loader, val_context_loader


def compute_checkpoint_score(cfg: dict, tight_f1: float, context_f1) -> float:
    """context_f1 is None when use_context_validation=False, in which case
    this reduces exactly to v1's original single-metric behavior (select
    on tight macro F1 alone)."""
    if context_f1 is None:
        return tight_f1

    metric = cfg.get("checkpoint_selection_metric", "combined_avg")
    if metric == "combined_avg":
        return (tight_f1 + context_f1) / 2
    if metric == "context_with_floor":
        floor = cfg.get("tight_f1_floor", 0.0)
        # disqualify (never selected as best) if the regression guard trips
        return context_f1 if tight_f1 >= floor else float("-inf")
    raise ValueError(f"Unknown checkpoint_selection_metric '{metric}', expected 'combined_avg' or 'context_with_floor'")


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

    # snapshot the crop-scale policy actually in effect (CLASS_SCALE_POLICY
    # is a code constant, not config-driven, so this is the only record of
    # exactly what was used for this specific run) plus the contamination
    # audit that motivated it, if available, so the run directory is
    # self-contained/reproducible without depending on the (gitignored)
    # runs/ directory still containing the original audit later.
    run_metadata = {"class_scale_policy": CLASS_SCALE_POLICY}
    contamination_audit_path = REPO_ROOT / "runs" / "crop_scale_contamination_audit" / "contamination_summary.json"
    if contamination_audit_path.exists():
        with open(contamination_audit_path) as f:
            run_metadata["contamination_audit"] = json.load(f)
    else:
        run_metadata["contamination_audit"] = None
        print(f"Note: no contamination audit found at {contamination_audit_path} — run_metadata.json will not include it.")
    with open(output_dir / "run_metadata.json", "w") as f:
        json.dump(run_metadata, f, indent=2)

    train_loader, val_loader, val_context_loader = build_dataloaders(cfg)
    class_names = sorted(AcneSCUCropDataset.CLASS_TO_ID, key=AcneSCUCropDataset.CLASS_TO_ID.get)
    print(f"train crops: {len(train_loader.dataset)}  val crops: {len(val_loader.dataset)}")
    if val_context_loader is not None:
        print(f"val context crops (fixed, detector-style): {len(val_context_loader.dataset)}")
        print(f"checkpoint_selection_metric: {cfg.get('checkpoint_selection_metric', 'combined_avg')}")
    print(f"classes: {class_names}")

    model = build_model(cfg["model_name"], num_classes=cfg["num_classes"], pretrained=cfg["pretrained"])

    # regression guard baseline: the val macro F1 the starting checkpoint
    # already had (v1 only ever had a "tight" val set, so this is directly
    # comparable to this run's val_tight_macro_f1)
    tight_f1_baseline = None
    finetune_from = cfg.get("finetune_from")
    if finetune_from and not resume:
        # resume takes priority — if resuming a v2 run, last.pth already
        # holds the (possibly partially fine-tuned) weights, so loading
        # finetune_from again here would discard that progress.
        print(f"Initializing weights from {finetune_from} (fine-tuning, not training from scratch)")
        base_checkpoint = torch.load(finetune_from, map_location="cpu")
        model.load_state_dict(base_checkpoint["model_state_dict"])
        tight_f1_baseline = base_checkpoint.get("val_macro_f1")

    model.to(device)

    class_weights_dict = None
    if cfg.get("class_weighting", True):
        from collections import Counter
        import csv as csv_module

        with open(cfg.get("manifest_path", DEFAULT_MANIFEST), newline="") as f:
            rows = list(csv_module.DictReader(f))
        train_counts = dict(Counter(r["mapped_class"] for r in rows if r["split"] == "train"))
        weights = compute_class_weights(train_counts).to(device)
        class_weights_dict = dict(zip(class_names, weights.tolist()))
        print(f"Class weights (train split, sqrt-inverse-frequency, normalized to mean 1): {class_weights_dict}")
        with open(output_dir / "class_weights.json", "w") as f:
            json.dump(class_weights_dict, f, indent=2)
        criterion = torch.nn.CrossEntropyLoss(weight=weights)
    else:
        criterion = torch.nn.CrossEntropyLoss()

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=cfg["lr_step_size"], gamma=cfg["lr_gamma"])

    history = []
    best_checkpoint_score = float("-inf")
    best_val_tight_macro_f1 = -1.0
    best_val_context_macro_f1 = None
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
        best_checkpoint_score = checkpoint.get("best_checkpoint_score", checkpoint.get("best_macro_f1", float("-inf")))
        best_val_tight_macro_f1 = checkpoint.get("best_val_tight_macro_f1", checkpoint.get("best_macro_f1", -1.0))
        best_val_context_macro_f1 = checkpoint.get("best_val_context_macro_f1")
        best_epoch = checkpoint["best_epoch"]
        epochs_since_improvement = checkpoint.get("epochs_since_improvement", 0)
        tight_f1_baseline = checkpoint.get("tight_f1_baseline", tight_f1_baseline)

        history_path = output_dir / "history.json"
        if history_path.exists():
            with open(history_path) as f:
                history = json.load(f)

        print(f"Resumed at epoch {start_epoch} (best so far: epoch {best_epoch}, checkpoint_score={best_checkpoint_score:.4f})")

    early_stopping_patience = cfg.get("early_stopping_patience", 8)

    epoch = start_epoch - 1  # so `epoch + 1` below is still correct if resuming with nothing left to train
    if start_epoch >= cfg["epochs"]:
        print(f"\nAlready at epoch {start_epoch}/{cfg['epochs']} — nothing left to train.")

    for epoch in range(start_epoch, cfg["epochs"]):
        start = time.time()
        print(f"\nEpoch {epoch + 1}/{cfg['epochs']}")

        train_metrics = train_one_epoch(
            model, optimizer, train_loader, device, criterion, log_interval=cfg["log_interval"]
        )
        scheduler.step()

        val_metrics = evaluate(model, val_loader, device, criterion, class_names)
        val_tight_macro_f1 = val_metrics["macro_f1"]

        val_context_metrics = None
        val_context_macro_f1 = None
        if val_context_loader is not None:
            val_context_metrics = evaluate(model, val_context_loader, device, criterion, class_names)
            val_context_macro_f1 = val_context_metrics["macro_f1"]

        checkpoint_score = compute_checkpoint_score(cfg, val_tight_macro_f1, val_context_macro_f1)

        elapsed = time.time() - start
        record = {
            "epoch": epoch + 1,
            "train_loss": train_metrics["loss"],
            "val_loss": val_metrics["loss"],
            "val_accuracy": val_metrics["accuracy"],
            "val_macro_precision": val_metrics["macro_precision"],
            "val_macro_recall": val_metrics["macro_recall"],
            "val_tight_macro_f1": val_tight_macro_f1,
            "val_per_class_f1": val_metrics["per_class_f1"],
            "val_context_macro_f1": val_context_macro_f1,
            "val_context_per_class_f1": val_context_metrics["per_class_f1"] if val_context_metrics else None,
            "checkpoint_score": checkpoint_score,
            "lr": scheduler.get_last_lr()[0],
            "elapsed_sec": elapsed,
        }
        history.append(record)

        context_str = f"  val_context_macro_f1={val_context_macro_f1:.4f}" if val_context_macro_f1 is not None else ""
        print(
            f"  train_loss={record['train_loss']:.4f}  val_loss={record['val_loss']:.4f}  "
            f"val_acc={record['val_accuracy']:.4f}  val_tight_macro_f1={val_tight_macro_f1:.4f}{context_str}  "
            f"({elapsed:.1f}s)"
        )

        # regression guard: warn (don't hard-abort — checkpoint_score
        # already factors this in via combined_avg/context_with_floor)
        # if the tight val performance is meaningfully below where the
        # starting checkpoint was.
        if tight_f1_baseline is not None:
            tolerance = cfg.get("tight_f1_regression_tolerance", 0.10)
            if val_tight_macro_f1 < tight_f1_baseline - tolerance:
                print(
                    f"  WARNING: val_tight_macro_f1={val_tight_macro_f1:.4f} is more than {tolerance:.2f} below "
                    f"the starting checkpoint's {tight_f1_baseline:.4f} — original classifier performance may be collapsing."
                )

        with open(output_dir / "history.json", "w") as f:
            json.dump(history, f, indent=2)

        if checkpoint_score > best_checkpoint_score:
            best_checkpoint_score = checkpoint_score
            best_val_tight_macro_f1 = val_tight_macro_f1
            best_val_context_macro_f1 = val_context_macro_f1
            best_epoch = epoch + 1
            epochs_since_improvement = 0
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "epoch": epoch + 1,
                    "val_macro_f1": val_tight_macro_f1,  # kept for backward compat with anything reading this key (e.g. regression-guard baseline lookup)
                    "val_tight_macro_f1": val_tight_macro_f1,
                    "val_context_macro_f1": val_context_macro_f1,
                    "checkpoint_score": checkpoint_score,
                    "config": cfg,
                    "class_names": class_names,
                    "class_weights": class_weights_dict,
                },
                output_dir / "best.pth",
            )
            print(f"  -> new best (checkpoint_score={checkpoint_score:.4f}), saved best.pth")
        else:
            epochs_since_improvement += 1

        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "epoch": epoch + 1,
                "best_checkpoint_score": best_checkpoint_score,
                "best_val_tight_macro_f1": best_val_tight_macro_f1,
                "best_val_context_macro_f1": best_val_context_macro_f1,
                "best_epoch": best_epoch,
                "epochs_since_improvement": epochs_since_improvement,
                "tight_f1_baseline": tight_f1_baseline,
                "config": cfg,
                "class_names": class_names,
                "class_weights": class_weights_dict,
            },
            last_ckpt_path,
        )

        if epochs_since_improvement >= early_stopping_patience:
            print(
                f"\nEarly stopping: no checkpoint_score improvement for {early_stopping_patience} epochs "
                f"(best epoch {best_epoch}, checkpoint_score={best_checkpoint_score:.4f})."
            )
            break

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "epoch": epoch + 1,
            "config": cfg,
            "class_names": class_names,
            "class_weights": class_weights_dict,
        },
        output_dir / "final.pth",
    )

    summary = {
        "best_epoch": best_epoch,
        "best_checkpoint_score": best_checkpoint_score,
        "best_val_tight_macro_f1": best_val_tight_macro_f1,
        "best_val_context_macro_f1": best_val_context_macro_f1,
        "total_epochs_run": epoch + 1,
        "class_weights": class_weights_dict,
    }
    with open(output_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nDone. Best epoch {best_epoch} with checkpoint_score={best_checkpoint_score:.4f} "
          f"(tight={best_val_tight_macro_f1:.4f}, context={best_val_context_macro_f1})")
    if start_epoch < cfg["epochs"]:  # at least one epoch actually ran this call
        print("\nFinal-epoch validation report (tight val set):")
        print(classification_report_str(val_metrics))
        if val_context_metrics is not None:
            print("\nFinal-epoch validation report (context val set):")
            print(classification_report_str(val_context_metrics))
    return history, summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/classifier_baseline.yaml")
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--epochs", type=int, default=None, help="override epochs from config")
    parser.add_argument("--manifest-path", type=str, default=None, help="override manifest_path from config")
    parser.add_argument(
        "--finetune-from", type=str, default=None,
        help="override finetune_from from config — e.g. a Drive path on Colab, since artifacts/ is gitignored and won't exist on a fresh clone",
    )
    parser.add_argument("--resume", action="store_true", help="resume from last.pth in output_dir if present")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    if args.epochs is not None:
        cfg["epochs"] = args.epochs
    if args.manifest_path is not None:
        cfg["manifest_path"] = args.manifest_path
    if args.finetune_from is not None:
        cfg["finetune_from"] = args.finetune_from

    output_dir = Path(args.output_dir) if args.output_dir else Path(cfg["output_dir"])
    train(cfg, output_dir, resume=args.resume)


if __name__ == "__main__":
    main()
