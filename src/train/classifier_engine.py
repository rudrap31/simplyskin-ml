"""Train/eval loops for the AcneSCU lesion-crop classifier."""
import torch

from src.train.classifier_metrics import compute_classification_metrics


def train_one_epoch(model, optimizer, data_loader, device, criterion, log_interval: int = 20) -> dict:
    model.train()
    total_loss = 0.0
    n_batches = 0

    for i, (images, labels, _ann_ids, _img_ids) in enumerate(data_loader):
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        n_batches += 1

        if log_interval and i % log_interval == 0:
            print(f"  batch {i}/{len(data_loader)}  loss={loss.item():.4f}")

    return {"loss": total_loss / n_batches}


@torch.no_grad()
def evaluate(model, data_loader, device, criterion, class_names: list[str]) -> dict:
    model.eval()
    total_loss = 0.0
    n_batches = 0
    all_preds = []
    all_labels = []

    for images, labels, _ann_ids, _img_ids in data_loader:
        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)
        loss = criterion(outputs, labels)
        total_loss += loss.item()
        n_batches += 1

        preds = outputs.argmax(dim=1)
        all_preds.append(preds.cpu())
        all_labels.append(labels.cpu())

    all_preds = torch.cat(all_preds)
    all_labels = torch.cat(all_labels)

    metrics = compute_classification_metrics(all_preds, all_labels, class_names)
    metrics["loss"] = total_loss / n_batches
    return metrics
