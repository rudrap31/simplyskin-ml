"""Smoke test: verify the detector pipeline on a tiny subset before
committing to any real training run.

Deliberately conservative for an 8GB machine: forced CPU (MPS memory
spikes are less predictable), batch_size=1, images pre-resized to a small
fixed size before they ever reach the model (so we control memory
ourselves rather than trusting the model's internal transform), and only
a handful of images / optimizer steps. Prints after every individual step
so a hang is easy to locate. This only proves the pipeline is wired up
correctly — it says nothing about final accuracy, and no real training
happens here or afterward on this machine (that's moving to a cloud GPU).

Checks:
  1. forward + backward pass completes
  2. losses and gradients are finite
  3. one optimizer step completes
  4. inference returns correctly positioned boxes (visual check)
  5. evaluation code runs on a few images
"""
import resource
import sys
import time
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import torch
from torch.utils.data import Subset
from torchvision.transforms.functional import to_pil_image

from src.data.acne04 import Acne04Detection
from src.data.transforms import collate_fn, get_transform
from src.models.detector import build_model
from src.train.metrics import compute_map, compute_precision_recall
from src.viz.visualize import draw_boxes, save_sample_grid

N_TRAIN = 6
N_VAL = 3
N_STEPS = 30
MIN_SIZE = 400
MAX_SIZE = 667
OUT_DIR = Path(__file__).resolve().parents[2] / "outputs" / "smoke_test"


def mem_mb() -> float:
    # ru_maxrss is bytes on macOS, KB on Linux
    kb_or_bytes = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return kb_or_bytes / (1024 * 1024) if sys.platform == "darwin" else kb_or_bytes / 1024


def log(msg: str):
    print(f"[{time.strftime('%H:%M:%S')}] (mem: {mem_mb():.0f} MB)  {msg}", flush=True)


def main():
    device = torch.device("cpu")
    log(f"device forced to: {device}")

    transform = get_transform(train=True, min_size=MIN_SIZE, max_size=MAX_SIZE)
    eval_transform = get_transform(train=False, min_size=MIN_SIZE, max_size=MAX_SIZE)

    log("loading dataset index (no images read yet)...")
    train_full = Acne04Detection(fold=0, split="train", transforms=transform)
    eval_full = Acne04Detection(fold=0, split="train", transforms=eval_transform)

    tiny_train = Subset(train_full, list(range(N_TRAIN)))
    tiny_eval = Subset(eval_full, list(range(N_VAL)))
    log(f"using {N_TRAIN} train images, {N_VAL} eval images, pre-resized to min_size={MIN_SIZE}/max_size={MAX_SIZE}")

    train_loader = torch.utils.data.DataLoader(
        tiny_train, batch_size=1, shuffle=True, num_workers=0, collate_fn=collate_fn
    )
    eval_loader = torch.utils.data.DataLoader(
        tiny_eval, batch_size=1, shuffle=False, num_workers=0, collate_fn=collate_fn
    )

    log("building model (min_size/max_size matched to pre-resize so the model's internal transform is a no-op)...")
    model = build_model(
        "fasterrcnn_resnet50_fpn", num_classes=2, pretrained=True, min_size=MIN_SIZE, max_size=MAX_SIZE
    )
    model.to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.005, momentum=0.9, weight_decay=0.0005)
    log("model built.")

    log(f"\n--- CHECKS 1-3: {N_STEPS} single-image optimizer steps ---")
    model.train()
    losses = []
    train_iter_data = [tiny_train[i] for i in range(N_TRAIN)]  # preload the tiny set once

    for step in range(N_STEPS):
        image, target = train_iter_data[step % N_TRAIN]
        images = [image.to(device)]
        targets = [{k: v.to(device) if torch.is_tensor(v) else v for k, v in target.items()}]

        log(f"step {step + 1}/{N_STEPS}: starting forward pass...")
        loss_dict = model(images, targets)
        total_loss = sum(loss_dict.values())
        loss_value = total_loss.item()
        log(f"step {step + 1}/{N_STEPS}: forward done, loss={loss_value:.4f}")

        if not torch.isfinite(total_loss):
            raise RuntimeError(f"non-finite loss at step {step + 1}: {loss_value}")

        optimizer.zero_grad()
        log(f"step {step + 1}/{N_STEPS}: starting backward pass...")
        total_loss.backward()
        log(f"step {step + 1}/{N_STEPS}: backward done")

        for name, param in model.named_parameters():
            if param.grad is not None and not torch.isfinite(param.grad).all():
                raise RuntimeError(f"non-finite gradient in {name} at step {step + 1}")
        log(f"step {step + 1}/{N_STEPS}: gradients confirmed finite")

        optimizer.step()
        log(f"step {step + 1}/{N_STEPS}: optimizer step done")

        losses.append(loss_value)

    print()
    log("[CHECK 1] loss decreases:")
    print(f"  first loss: {losses[0]:.4f}  last loss: {losses[-1]:.4f}  all losses: {[round(l, 3) for l in losses]}")
    assert losses[-1] < losses[0], "loss did not decrease over smoke test steps"
    log("  PASS")

    log("[CHECK 2] gradients stayed finite: PASS (checked every step above)")
    log("[CHECK 3] optimizer steps completed: PASS")

    log("\n--- CHECK 4: inference + box placement ---")
    model.eval()
    annotated = []
    with torch.no_grad():
        for i in range(N_VAL):
            image, target = tiny_eval[i]
            log(f"running inference on eval image {i + 1}/{N_VAL}...")
            output = model([image.to(device)])[0]

            keep = output["scores"] >= 0.3
            pred_boxes = output["boxes"][keep].cpu().tolist()
            gt_boxes = target["boxes"].tolist()
            log(f"  eval image {i + 1}: {len(gt_boxes)} gt boxes, {len(pred_boxes)} predicted boxes (score>=0.3)")

            pil_image = to_pil_image(image)
            img_with_gt = draw_boxes(pil_image, gt_boxes, labels=["gt"] * len(gt_boxes))
            img_with_both = draw_boxes(img_with_gt, pred_boxes, labels=["pred"] * len(pred_boxes))
            annotated.append(img_with_both)

    out_path = OUT_DIR / "overfit_predictions.jpg"
    save_sample_grid(annotated, out_path, cols=3, cell_size=350)
    log(f"[CHECK 4] saved visual check -> {out_path} (green=gt, orange=pred). PASS if predicted boxes cluster near gt.")

    log("\n--- CHECK 5: evaluation code runs end to end ---")
    all_predictions, all_targets = [], []
    with torch.no_grad():
        for images, targets in eval_loader:
            outputs = model([img.to(device) for img in images])
            for out, target in zip(outputs, targets):
                all_predictions.append({"boxes": out["boxes"].cpu(), "labels": out["labels"].cpu(), "scores": out["scores"].cpu()})
                all_targets.append({"boxes": target["boxes"].cpu(), "labels": target["labels"].cpu()})

    map_metrics = compute_map(all_predictions, all_targets)
    pr_metrics = compute_precision_recall(all_predictions, all_targets, score_threshold=0.3)
    log(f"  mAP@0.5={map_metrics['mAP_50']:.4f}  mAP@0.5:0.95={map_metrics['mAP']:.4f}  "
        f"precision={pr_metrics['precision']:.4f}  recall={pr_metrics['recall']:.4f}")
    log("[CHECK 5] PASS (evaluation code ran without error)")

    log("\nAll smoke test checks passed. Pipeline is verified end to end on CPU with a tiny subset.")
    log("Next step: real training moves to a cloud GPU, not this machine.")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        print("\nSMOKE TEST FAILED:", flush=True)
        traceback.print_exc()
        sys.exit(1)
