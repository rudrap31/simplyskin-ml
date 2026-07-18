"""Train/eval loops for torchvision-style detection models."""
import math

import torch

from src.train.metrics import compute_map, compute_precision_recall


def train_one_epoch(model, optimizer, data_loader, device, log_interval: int = 20):
    """One epoch of training. Returns dict of average losses.

    Raises if a non-finite loss/gradient shows up — silently continuing
    training on NaN losses just wastes time and hides the real bug.
    """
    model.train()
    loss_sums = {}
    n_batches = 0

    for i, (images, targets) in enumerate(data_loader):
        images = [img.to(device) for img in images]
        targets = [{k: v.to(device) if torch.is_tensor(v) else v for k, v in t.items()} for t in targets]

        loss_dict = model(images, targets)
        losses = sum(loss_dict.values())

        loss_value = losses.item()
        if not math.isfinite(loss_value):
            raise RuntimeError(f"Non-finite loss at batch {i}: {loss_value}, components={loss_dict}")

        optimizer.zero_grad()
        losses.backward()

        for name, param in model.named_parameters():
            if param.grad is not None and not torch.isfinite(param.grad).all():
                raise RuntimeError(f"Non-finite gradient in {name} at batch {i}")

        optimizer.step()

        for k, v in loss_dict.items():
            loss_sums[k] = loss_sums.get(k, 0.0) + v.item()
        loss_sums["total_loss"] = loss_sums.get("total_loss", 0.0) + loss_value
        n_batches += 1

        if log_interval and i % log_interval == 0:
            print(f"  batch {i}/{len(data_loader)}  loss={loss_value:.4f}")

    return {k: v / n_batches for k, v in loss_sums.items()}


@torch.no_grad()
def compute_val_loss(model, data_loader, device) -> dict:
    """Detection models only return losses in train() mode. Running that
    forward pass under no_grad computes the loss without updating weights.
    Safe for this project's backbone: torchvision detection models use
    FrozenBatchNorm2d, so train() mode doesn't drift BN running stats."""
    model.train()
    loss_sums = {}
    n_batches = 0

    for images, targets in data_loader:
        images = [img.to(device) for img in images]
        targets = [{k: v.to(device) if torch.is_tensor(v) else v for k, v in t.items()} for t in targets]

        loss_dict = model(images, targets)
        for k, v in loss_dict.items():
            loss_sums[k] = loss_sums.get(k, 0.0) + v.item()
        loss_sums["total_loss"] = loss_sums.get("total_loss", 0.0) + sum(v.item() for v in loss_dict.values())
        n_batches += 1

    return {k: v / n_batches for k, v in loss_sums.items()}


@torch.no_grad()
def evaluate(model, data_loader, device, score_threshold: float = 0.5) -> dict:
    """Full validation/test pass: val loss + mAP + precision/recall."""
    val_loss = compute_val_loss(model, data_loader, device)

    model.eval()
    all_predictions = []
    all_targets = []

    for images, targets in data_loader:
        images = [img.to(device) for img in images]
        outputs = model(images)

        for out, target in zip(outputs, targets):
            all_predictions.append(
                {
                    "boxes": out["boxes"].cpu(),
                    "labels": out["labels"].cpu(),
                    "scores": out["scores"].cpu(),
                }
            )
            all_targets.append(
                {
                    "boxes": target["boxes"].cpu(),
                    "labels": target["labels"].cpu(),
                }
            )

    map_metrics = compute_map(all_predictions, all_targets)
    pr_metrics = compute_precision_recall(all_predictions, all_targets, score_threshold=score_threshold)

    return {
        "val_loss": val_loss["total_loss"],
        "val_loss_components": val_loss,
        **map_metrics,
        **pr_metrics,
    }
