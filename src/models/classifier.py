"""Lesion-crop classifier model factory. Mirrors src/models/detector.py's
pattern: a MODEL_REGISTRY + build_model() so a future architecture can be
swapped in without touching the training loop.
"""
import torchvision
from torch import nn


def build_efficientnet_b0(num_classes: int, pretrained: bool = True, **_ignored):
    """EfficientNet-B0: strong accuracy/compute tradeoff for a small
    dataset, and torchvision's classifier head is a single Linear layer
    (classifier[1]) that's trivial to replace."""
    weights = "DEFAULT" if pretrained else None
    model = torchvision.models.efficientnet_b0(weights=weights)
    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features, num_classes)
    return model


def build_resnet18(num_classes: int, pretrained: bool = True, **_ignored):
    weights = "DEFAULT" if pretrained else None
    model = torchvision.models.resnet18(weights=weights)
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)
    return model


MODEL_REGISTRY = {
    "efficientnet_b0": build_efficientnet_b0,
    "resnet18": build_resnet18,
}


def build_model(name: str, num_classes: int, pretrained: bool = True, **kwargs):
    if name not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model '{name}'. Available: {list(MODEL_REGISTRY)}")
    return MODEL_REGISTRY[name](num_classes=num_classes, pretrained=pretrained, **kwargs)
