"""Classification metrics for the AcneSCU lesion-crop classifier.
Implemented directly with torch (no sklearn dependency for the core
numbers) so the confusion matrix and metrics are guaranteed consistent
with each other; a sklearn-formatted classification_report string is
also provided for convenience in the eval script's saved report.
"""
import torch


def confusion_matrix(preds: torch.Tensor, labels: torch.Tensor, num_classes: int) -> torch.Tensor:
    """rows = true class, cols = predicted class."""
    cm = torch.zeros((num_classes, num_classes), dtype=torch.int64)
    for t, p in zip(labels.view(-1), preds.view(-1)):
        cm[t.long(), p.long()] += 1
    return cm


def compute_classification_metrics(preds: torch.Tensor, labels: torch.Tensor, class_names: list[str]) -> dict:
    num_classes = len(class_names)
    cm = confusion_matrix(preds, labels, num_classes)

    correct = torch.diag(cm).sum().item()
    total = cm.sum().item()
    accuracy = correct / total if total > 0 else 0.0

    per_class_precision = {}
    per_class_recall = {}
    per_class_f1 = {}
    for i, name in enumerate(class_names):
        tp = cm[i, i].item()
        fp = cm[:, i].sum().item() - tp
        fn = cm[i, :].sum().item() - tp
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        per_class_precision[name] = precision
        per_class_recall[name] = recall
        per_class_f1[name] = f1

    macro_precision = sum(per_class_precision.values()) / num_classes
    macro_recall = sum(per_class_recall.values()) / num_classes
    macro_f1 = sum(per_class_f1.values()) / num_classes

    return {
        "accuracy": accuracy,
        "macro_precision": macro_precision,
        "macro_recall": macro_recall,
        "macro_f1": macro_f1,
        "per_class_precision": per_class_precision,
        "per_class_recall": per_class_recall,
        "per_class_f1": per_class_f1,
        "confusion_matrix": cm.tolist(),
        "class_names": class_names,
    }


def classification_report_str(metrics: dict) -> str:
    """Human-readable per-class + macro summary, sklearn-report-like."""
    class_names = metrics["class_names"]
    lines = [f"{'class':26s} {'precision':>10s} {'recall':>10s} {'f1':>10s}"]
    for name in class_names:
        lines.append(
            f"{name:26s} {metrics['per_class_precision'][name]:>10.4f} "
            f"{metrics['per_class_recall'][name]:>10.4f} {metrics['per_class_f1'][name]:>10.4f}"
        )
    lines.append("")
    lines.append(f"{'accuracy':26s} {'':>10s} {'':>10s} {metrics['accuracy']:>10.4f}")
    lines.append(
        f"{'macro avg':26s} {metrics['macro_precision']:>10.4f} "
        f"{metrics['macro_recall']:>10.4f} {metrics['macro_f1']:>10.4f}"
    )
    return "\n".join(lines)
