"""Draw bounding boxes over images for sanity-checking annotations."""
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

# stable color per class name so grids are easy to read at a glance
CLASS_COLORS = {
    "lesion": (255, 0, 0),
    "comedonal_like": (66, 135, 245),
    "inflammatory_like": (245, 66, 66),
    "deeper_inflammatory_like": (156, 39, 176),
    "non_active_acne": (128, 128, 128),
    "gt": (0, 220, 0),
    "pred": (255, 165, 0),
}
DEFAULT_COLOR = (0, 200, 0)


def draw_boxes(
    image: Image.Image, boxes, labels=None, label_names=None, max_size: int = 900
) -> Image.Image:
    """Return a copy of image with boxes (and optional class labels) drawn.

    boxes: iterable of (x1, y1, x2, y2)
    labels: optional iterable of label names (str) parallel to boxes
    max_size: image is downscaled to this before drawing (boxes scaled to
        match) so outlines stay visible at typical thumbnail/grid sizes —
        drawing on full-resolution originals makes thin lesion boxes vanish
        once the image is later shrunk for display.
    """
    image = image.copy()
    scale = min(1.0, max_size / max(image.size))
    if scale < 1.0:
        new_size = (round(image.width * scale), round(image.height * scale))
        image = image.resize(new_size, Image.BILINEAR)

    draw = ImageDraw.Draw(image)
    for i, box in enumerate(boxes):
        x1, y1, x2, y2 = [c * scale for c in box]
        name = labels[i] if labels is not None else "lesion"
        color = CLASS_COLORS.get(name, DEFAULT_COLOR)
        draw.rectangle([x1, y1, x2, y2], outline=color, width=3)
        if label_names is not False and labels is not None:
            draw.text((x1, max(0, y1 - 12)), name, fill=color)

    return image


def save_sample_grid(samples, out_path: Path, cols: int = 4, cell_size: int = 300):
    """samples: list of PIL Images (already annotated). Tiles them into a grid."""
    rows = (len(samples) + cols - 1) // cols
    grid = Image.new("RGB", (cols * cell_size, rows * cell_size), (30, 30, 30))

    for i, img in enumerate(samples):
        img = img.copy()
        img.thumbnail((cell_size, cell_size))
        r, c = divmod(i, cols)
        x = c * cell_size + (cell_size - img.width) // 2
        y = r * cell_size + (cell_size - img.height) // 2
        grid.paste(img, (x, y))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    grid.save(out_path)
    return out_path
