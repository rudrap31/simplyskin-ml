"""Detection metrics: mAP (via torchmetrics/pycocotools) plus a simple
IoU-matched precision/recall at a fixed operating point.

mAP is threshold-free (integrates over the precision-recall curve), which
is the right metric for comparing detectors, but it's not very intuitive
to read epoch-to-epoch. Precision/recall at a fixed score threshold gives
a second, more concrete number: "at score >= 0.5, how many of our boxes
are real lesions, and how many real lesions did we find."
"""
import torch
from torchmetrics.detection import MeanAveragePrecision
from torchvision.ops import box_iou


def compute_map(predictions: list[dict], targets: list[dict]) -> dict:
    """predictions/targets: torchvision-style list of dicts with 'boxes',
    'labels' (+ 'scores' for predictions). Returns map/map_50/map_75/mar_100."""
    metric = MeanAveragePrecision(box_format="xyxy", iou_type="bbox")
    metric.update(predictions, targets)
    result = metric.compute()
    return {
        "mAP": result["map"].item(),
        "mAP_50": result["map_50"].item(),
        "mAP_75": result["map_75"].item(),
        "mAR_100": result["mar_100"].item(),
    }


def compute_precision_recall(
    predictions: list[dict],
    targets: list[dict],
    iou_threshold: float = 0.5,
    score_threshold: float = 0.5,
) -> dict:
    """Greedy one-to-one IoU matching per image, aggregated across the
    whole dataset. Single-class only (matches this project's use case)."""
    total_tp, total_fp, total_fn = 0, 0, 0

    for pred, target in zip(predictions, targets):
        keep = pred["scores"] >= score_threshold
        pred_boxes = pred["boxes"][keep]
        pred_scores = pred["scores"][keep]
        gt_boxes = target["boxes"]

        order = torch.argsort(pred_scores, descending=True)
        pred_boxes = pred_boxes[order]

        n_gt = gt_boxes.shape[0]
        matched_gt = torch.zeros(n_gt, dtype=torch.bool)

        if pred_boxes.shape[0] > 0 and n_gt > 0:
            ious = box_iou(pred_boxes, gt_boxes)  # (num_preds, num_gt)
        else:
            ious = None

        tp = 0
        fp = 0
        for i in range(pred_boxes.shape[0]):
            if ious is None:
                fp += 1
                continue
            row = ious[i].clone()
            row[matched_gt] = 0.0
            best_iou, best_j = row.max(dim=0)
            if best_iou.item() >= iou_threshold:
                matched_gt[best_j] = True
                tp += 1
            else:
                fp += 1

        fn = n_gt - matched_gt.sum().item()

        total_tp += tp
        total_fp += fp
        total_fn += fn

    precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
    recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0

    return {
        "precision": precision,
        "recall": recall,
        "tp": total_tp,
        "fp": total_fp,
        "fn": total_fn,
    }
