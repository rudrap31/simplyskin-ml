"""End-to-end acne pipeline v1: full face image -> detector (frozen,
ACNE04) -> per-detection crop (same padding convention as classifier
training) -> classifier (frozen, AcneSCU) -> per-lesion broad-class label.

Both models are loaded read-only from their frozen checkpoints; nothing
here retrains or modifies either.
"""
from pathlib import Path

import torch
from PIL import Image
from torchvision.transforms.functional import to_tensor

from src.data.acnescu_crops import AcneSCUCropDataset, pad_and_clamp_box
from src.data.classifier_transforms import get_classifier_transform
from src.models.classifier import build_model as build_classifier
from src.models.detector import build_model as build_detector

CLASS_NAMES = sorted(AcneSCUCropDataset.CLASS_TO_ID, key=AcneSCUCropDataset.CLASS_TO_ID.get)


def resolve_device(requested: str = "auto") -> torch.device:
    if requested != "auto":
        return torch.device(requested)
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def load_detector(checkpoint_path: Path, device: torch.device):
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    cfg = checkpoint["config"]
    model = build_detector(
        cfg["model_name"],
        num_classes=cfg["num_classes"],
        pretrained=False,
        min_size=cfg.get("min_size"),
        max_size=cfg.get("max_size"),
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    return model, cfg


def load_classifier(checkpoint_path: Path, device: torch.device):
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    cfg = checkpoint["config"]
    class_names = checkpoint["class_names"]
    model = build_classifier(cfg["model_name"], num_classes=cfg["num_classes"], pretrained=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    return model, cfg, class_names


@torch.no_grad()
def run_pipeline_on_image(
    image: Image.Image,
    detector,
    classifier,
    classifier_class_names: list[str],
    device: torch.device,
    detector_score_threshold: float = 0.5,
    crop_padding: float = 0.15,
    classifier_input_size: int = 224,
) -> list[dict]:
    """Returns one dict per kept detection:
      box: [x1,y1,x2,y2] in original image pixel coords (detector's raw box,
           NOT the padded crop box — this is what should be matched against
           ground truth for localization)
      detector_score: float
      classified: bool (False only if the box degenerates to nothing after
           padding/clamping — extremely rare, MIN_CROP_SIZE=8px)
      predicted_class / class_confidence / class_probs: present iff classified
    """
    img_w, img_h = image.size
    img_tensor = to_tensor(image).to(device)

    output = detector([img_tensor])[0]
    keep = output["scores"] >= detector_score_threshold
    boxes = output["boxes"][keep].cpu().tolist()
    scores = output["scores"][keep].cpu().tolist()

    transform = get_classifier_transform(train=False, input_size=classifier_input_size)

    results = []
    for box, score in zip(boxes, scores):
        x1, y1, x2, y2 = box
        bbox_xywh = [x1, y1, x2 - x1, y2 - y1]
        padded = pad_and_clamp_box(bbox_xywh, img_w, img_h, crop_padding)

        record = {"box": [x1, y1, x2, y2], "detector_score": score, "classified": False}

        if padded is not None:
            crop = image.crop((padded[0], padded[1], padded[2], padded[3]))
            crop_tensor = transform(crop).unsqueeze(0).to(device)
            logits = classifier(crop_tensor)
            probs = torch.softmax(logits, dim=1)[0]
            pred_idx = int(probs.argmax().item())

            record["classified"] = True
            record["predicted_class"] = classifier_class_names[pred_idx]
            record["class_confidence"] = float(probs[pred_idx].item())
            record["class_probs"] = {
                name: float(probs[i].item()) for i, name in enumerate(classifier_class_names)
            }

        results.append(record)

    return results
