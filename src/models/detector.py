"""Detector model factory.

Kept separate from training code so a future detector (e.g. RetinaNet,
FCOS, YOLO) can be swapped in by adding another build_* function without
touching the training loop, which only relies on the shared interface:
  - train mode: model(images, targets) -> dict of losses
  - eval mode:  model(images) -> list of {boxes, labels, scores} dicts
That's the standard torchvision detection model contract.
"""
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor


def build_faster_rcnn_resnet50_fpn(
    num_classes: int,
    pretrained: bool = True,
    min_size: int | None = None,
    max_size: int | None = None,
):
    """num_classes includes background (class 0). For single-class lesion
    detection that's num_classes=2 (background + lesion).

    min_size/max_size: passed to the model's internal resize transform.
    Leave None for torchvision defaults (800/1333). Set these to match
    any pre-resizing already applied by the data transforms (see
    src/data/transforms.Resize) so the model doesn't undo it by resizing
    back up — useful on memory-constrained hardware.
    """
    weights = "DEFAULT" if pretrained else None
    kwargs = {}
    if min_size is not None:
        kwargs["min_size"] = min_size
    if max_size is not None:
        kwargs["max_size"] = max_size
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights=weights, **kwargs)

    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    return model


MODEL_REGISTRY = {
    "fasterrcnn_resnet50_fpn": build_faster_rcnn_resnet50_fpn,
}


def build_model(name: str, num_classes: int, pretrained: bool = True, **kwargs):
    if name not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model '{name}'. Available: {list(MODEL_REGISTRY)}")
    return MODEL_REGISTRY[name](num_classes=num_classes, pretrained=pretrained, **kwargs)
