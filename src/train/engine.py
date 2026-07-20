"""Train/eval loops for torchvision-style detection models."""
import math

import torch

from src.train.metrics import compute_map, compute_precision_recall


def train_one_epoch(model, optimizer, data_loader, device, scaler, log_interval: int = 20, mixed_precision: bool = False):
    """One epoch of training. Returns dict of average losses.

    Raises if a non-finite loss/gradient shows up — silently continuing
    training on NaN losses just wastes time and hides the real bug.

    mixed_precision only takes effect on CUDA (autocast on CPU/MPS isn't
    reliably beneficial/supported for this model, so it's a no-op there).

    scaler is created once by the caller (not here) and passed in across
    epochs, so its growth/backoff state persists across epochs and can be
    saved/restored for resume — a scaler recreated every epoch would lose
    that state each time.
    """
    model.train()
    loss_sums = {}
    n_batches = 0

    use_amp = mixed_precision and device.type == "cuda"

    for i, (images, targets) in enumerate(data_loader):
        images = [img.to(device) for img in images]
        targets = [{k: v.to(device) if torch.is_tensor(v) else v for k, v in t.items()} for t in targets]

        with torch.autocast(device_type="cuda", enabled=use_amp):
            loss_dict = model(images, targets)
            losses = sum(loss_dict.values())

        loss_value = losses.item()
        if not math.isfinite(loss_value):
            raise RuntimeError(f"Non-finite loss at batch {i}: {loss_value}, components={loss_dict}")

        optimizer.zero_grad()
        scaler.scale(losses).backward()

        # unscale_() is a documented no-op when the scaler is disabled
        # (use_amp=False), so this ordering is correct in both modes:
        # scale -> backward -> unscale -> inspect real gradients -> step -> update
        scaler.unscale_(optimizer)

        non_finite_param = None
        for name, param in model.named_parameters():
            if param.grad is not None and not torch.isfinite(param.grad).all():
                non_finite_param = name
                break

        if non_finite_param is not None:
            if use_amp:
                # Expected occasionally under fp16 autocast (intermediate
                # overflow), not necessarily a real bug like it would be in
                # FP32. GradScaler.step() detects this itself (it tracks
                # inf/nan during unscale_) and skips optimizer.step(); update()
                # then backs off the scale so future steps are less likely to
                # overflow. Don't raise here — that's the whole point of AMP.
                print(f"  batch {i}: non-finite gradient in {non_finite_param}, AMP step skipped, scale reduced")
            else:
                raise RuntimeError(f"Non-finite gradient in {non_finite_param} at batch {i}")

        scaler.step(optimizer)
        scaler.update()

        for k, v in loss_dict.items():
            loss_sums[k] = loss_sums.get(k, 0.0) + v.item()
        loss_sums["total_loss"] = loss_sums.get("total_loss", 0.0) + loss_value
        n_batches += 1

        if log_interval and i % log_interval == 0:
            print(f"  batch {i}/{len(data_loader)}  loss={loss_value:.4f}")

    return {k: v / n_batches for k, v in loss_sums.items()}


@torch.no_grad()
def compute_val_loss(model, data_loader, device, mixed_precision: bool = False) -> dict:
    """Detection models only return losses in train() mode. Running that
    forward pass under no_grad computes the loss without updating weights.
    Safe for this project's backbone: torchvision detection models use
    FrozenBatchNorm2d, so train() mode doesn't drift BN running stats."""
    model.train()
    loss_sums = {}
    n_batches = 0
    use_amp = mixed_precision and device.type == "cuda"

    for images, targets in data_loader:
        images = [img.to(device) for img in images]
        targets = [{k: v.to(device) if torch.is_tensor(v) else v for k, v in t.items()} for t in targets]

        with torch.autocast(device_type="cuda", enabled=use_amp):
            loss_dict = model(images, targets)
        for k, v in loss_dict.items():
            loss_sums[k] = loss_sums.get(k, 0.0) + v.item()
        loss_sums["total_loss"] = loss_sums.get("total_loss", 0.0) + sum(v.item() for v in loss_dict.values())
        n_batches += 1

    return {k: v / n_batches for k, v in loss_sums.items()}


@torch.no_grad()
def evaluate(model, data_loader, device, score_threshold: float = 0.5, mixed_precision: bool = False) -> dict:
    """Full validation/test pass: val loss + mAP + precision/recall."""
    val_loss = compute_val_loss(model, data_loader, device, mixed_precision=mixed_precision)

    model.eval()
    all_predictions = []
    all_targets = []
    use_amp = mixed_precision and device.type == "cuda"

    for images, targets in data_loader:
        images = [img.to(device) for img in images]
        with torch.autocast(device_type="cuda", enabled=use_amp):
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
