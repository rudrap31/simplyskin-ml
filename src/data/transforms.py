"""Joint image+target transforms for detection.

We deliberately do NOT resize or normalize here — torchvision's Faster
R-CNN applies its own internal GeneralizedRCNNTransform (resize +
ImageNet normalization) inside model.forward(), and rescales target boxes
to match automatically. Doing it ourselves would double-transform boxes.
Only conservative, box-safe augmentations live here.
"""
import random

import torch
import torchvision.transforms.functional as F


class Compose:
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, target):
        for t in self.transforms:
            image, target = t(image, target)
        return image, target


class ToTensor:
    """PIL Image -> float tensor in [0, 1], shape (C, H, W)."""

    def __call__(self, image, target):
        return F.to_tensor(image), target


class RandomHorizontalFlip:
    def __init__(self, prob: float = 0.5):
        self.prob = prob

    def __call__(self, image, target):
        if random.random() >= self.prob:
            return image, target

        width = F.get_image_size(image)[0]
        image = F.hflip(image)

        boxes = target["boxes"]
        if boxes.numel() > 0:
            flipped = boxes.clone()
            flipped[:, 0] = width - boxes[:, 2]
            flipped[:, 2] = width - boxes[:, 0]
            target = dict(target)
            target["boxes"] = flipped

        return image, target


class Resize:
    """Resize so the shorter side == min_size, capped so the longer side
    never exceeds max_size — same convention torchvision's internal
    GeneralizedRCNNTransform uses. Boxes are scaled by the same factor.

    Not used by default (the model's internal transform normally handles
    this). Useful when we want to control memory usage ourselves before
    the image ever reaches the model, e.g. on memory-constrained hardware.
    """

    def __init__(self, min_size: int, max_size: int):
        self.min_size = min_size
        self.max_size = max_size

    def __call__(self, image, target):
        width, height = F.get_image_size(image)
        short, long_ = min(width, height), max(width, height)

        scale = self.min_size / short
        if long_ * scale > self.max_size:
            scale = self.max_size / long_

        new_size = (round(height * scale), round(width * scale))  # F.resize wants (h, w)
        image = F.resize(image, new_size)

        boxes = target["boxes"]
        if boxes.numel() > 0:
            target = dict(target)
            target["boxes"] = boxes * scale

        return image, target


class RandomPhotometricJitter:
    """Mild brightness/contrast jitter. Image-only, boxes untouched.

    Deliberately conservative (small ranges) and skips saturation/hue —
    inflammatory vs. comedonal lesions are partly distinguished by color,
    so we don't want to distort it aggressively.
    """

    def __init__(self, brightness: float = 0.15, contrast: float = 0.15, prob: float = 0.5):
        self.brightness = brightness
        self.contrast = contrast
        self.prob = prob

    def __call__(self, image, target):
        if random.random() >= self.prob:
            return image, target

        if self.brightness > 0:
            factor = random.uniform(1 - self.brightness, 1 + self.brightness)
            image = F.adjust_brightness(image, factor)
        if self.contrast > 0:
            factor = random.uniform(1 - self.contrast, 1 + self.contrast)
            image = F.adjust_contrast(image, factor)

        return image, target


def get_transform(train: bool, min_size: int | None = None, max_size: int | None = None) -> Compose:
    """Standard transform pipeline. train=False gives eval-safe (no aug).

    min_size/max_size: if given, resize (and rescale boxes) before the
    image ever reaches the model — see Resize. Leave None to rely on the
    model's own internal resize (the default for normal training).
    """
    transforms = []
    if min_size is not None:
        transforms.append(Resize(min_size, max_size))
    if train:
        transforms.append(RandomPhotometricJitter())
        transforms.append(RandomHorizontalFlip())
    transforms.append(ToTensor())
    return Compose(transforms)


def collate_fn(batch):
    """Detection batches have variable-sized images/box counts, so we
    can't stack them — just return parallel tuples."""
    return tuple(zip(*batch))
