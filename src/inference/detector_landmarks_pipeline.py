"""Production inference pipeline v1: face image -> MediaPipe Face
Landmarker -> facial-region polygons -> frozen Faster R-CNN acne detector
-> region assignment -> structured JSON. No classifier in this path (see
artifacts/README.md — the lesion-subtype classifier is experimental only).

This module only extracts objective computer-vision evidence (box
coordinates, confidence, assigned facial region). It deliberately does not
infer lesion subtype, medical cause, or recommendations — that happens
later in the backend via a vision LLM.
"""
from pathlib import Path

import torch
from PIL import Image, ImageDraw
from torchvision.transforms.functional import to_tensor

from src.inference.pipeline import load_detector, resolve_device
from src.landmarks.face_landmarker import DEFAULT_MODEL_PATH, FaceLandmarkDetector
from src.landmarks.regions import assign_detection_to_region, build_face_regions

REGION_COLORS = {
    "forehead": (255, 60, 60),
    "image_left_cheek": (60, 220, 60),
    "image_right_cheek": (60, 100, 255),
    "nose": (255, 220, 40),
    "chin": (230, 60, 230),
    "other_face": (60, 220, 220),
}
EXCLUSION_COLORS = {
    "left_eye": (130, 0, 0), "right_eye": (130, 0, 0),
    "left_eyebrow": (0, 110, 0), "right_eyebrow": (0, 110, 0),
    "lips": (0, 0, 130),
}
DETECTION_BOX_COLOR = (255, 140, 0)
UNASSIGNED_BOX_COLOR = (160, 160, 160)

PIPELINE_VERSION = "detector_landmarks_v1"
REGION_NAMES = ["forehead", "image_left_cheek", "image_right_cheek", "nose", "chin", "other_face"]
T_ZONE_REGIONS = {"forehead", "nose", "chin"}
CHEEK_REGIONS = {"image_left_cheek", "image_right_cheek"}


def _empty_summary():
    return {
        "total_lesions": 0,
        "assigned_lesions": 0,
        "unassigned_lesions": 0,
        "average_detection_confidence": 0.0,
        "counts_by_region": {name: 0 for name in REGION_NAMES},
        "dominant_region": None,
        "cheek_count": 0,
        "t_zone_count": 0,
        "image_left_right_cheek_difference": 0,
    }


@torch.no_grad()
def run_detector(detector, image: Image.Image, device, score_threshold: float) -> list:
    """Runs the frozen detector with the same preprocessing/threshold used
    in production (src.inference.pipeline). torchvision's Faster R-CNN
    applies NMS internally (RoIHeads, box_nms_thresh) as part of standard
    inference — no separate duplicate-suppression step is added here."""
    img_tensor = to_tensor(image).to(device)
    output = detector([img_tensor])[0]
    keep = output["scores"] >= score_threshold
    boxes = output["boxes"][keep].cpu().tolist()
    scores = output["scores"][keep].cpu().tolist()
    return list(zip(boxes, scores))


def run_pipeline_on_image(
    image_path: Path,
    detector,
    landmark_detector: FaceLandmarkDetector,
    device,
    score_threshold: float = 0.5,
) -> tuple:
    """Returns (result, face_regions). result is a dict matching the stable
    JSON contract (see stop-point report for the schema); face_regions is
    a landmarks.regions.FaceRegions (or None if no face was detected) —
    not part of the JSON contract, returned separately so callers can draw
    the region-polygon overlay without recomputing it."""
    image = Image.open(image_path).convert("RGB")
    img_w, img_h = image.size

    landmark_result = landmark_detector.detect(image_path=image_path)

    result = {
        "pipeline_version": PIPELINE_VERSION,
        "face_detected": landmark_result.face_detected,
        "multiple_faces_detected": landmark_result.multiple_faces_detected,
        "image": {"width": img_w, "height": img_h},
        "detections": [],
        "summary": _empty_summary(),
        "quality": {"landmark_confidence": None, "warnings": list(landmark_result.warnings)},
        "models": {
            "detector": "fasterrcnn_resnet50_fpn (artifacts/detector_v1/best.pth)",
            "face_landmarker": f"mediapipe {landmark_result.model_metadata.get('mediapipe_version')} / {Path(landmark_result.model_metadata.get('model_asset_path', DEFAULT_MODEL_PATH)).name}",
        },
    }

    if not landmark_result.face_detected:
        result["quality"]["warnings"].append("Region assignment skipped: no face detected.")
        return result, None

    face_regions = build_face_regions(landmark_result.landmarks, img_w, img_h)
    face_area = face_regions.face_oval.area

    raw_detections = run_detector(detector, image, device, score_threshold)

    detections = []
    confidences = []
    counts_by_region = {name: 0 for name in REGION_NAMES}
    unassigned_count = 0

    for box, score in raw_detections:
        x1, y1, x2, y2 = box
        assignment = assign_detection_to_region(box, face_regions)
        region = assignment["region"]

        detection_record = {
            "box": [x1, y1, x2, y2],
            "normalized_box": [x1 / img_w, y1 / img_h, x2 / img_w, y2 / img_h],
            "confidence": score,
            "region": region,
            "region_overlap_ratio": assignment["region_overlap_ratio"],
            "relative_face_area": ((x2 - x1) * (y2 - y1)) / face_area if face_area > 0 else 0.0,
        }
        detections.append(detection_record)
        confidences.append(score)

        if region == "unassigned":
            unassigned_count += 1
        else:
            counts_by_region[region] += 1

    total = len(detections)
    assigned = total - unassigned_count
    dominant_region = max(counts_by_region, key=counts_by_region.get) if any(counts_by_region.values()) else None
    left_count = counts_by_region["image_left_cheek"]
    right_count = counts_by_region["image_right_cheek"]

    result["detections"] = detections
    result["summary"] = {
        "total_lesions": total,
        "assigned_lesions": assigned,
        "unassigned_lesions": unassigned_count,
        "average_detection_confidence": sum(confidences) / len(confidences) if confidences else 0.0,
        "counts_by_region": counts_by_region,
        "dominant_region": dominant_region,
        "cheek_count": left_count + right_count,
        "t_zone_count": sum(counts_by_region[r] for r in T_ZONE_REGIONS),
        "image_left_right_cheek_difference": left_count - right_count,
    }

    return result, face_regions  # face_regions returned separately for overlay drawing (not part of the JSON contract)


def _draw_polygon(draw, poly, fill_rgb, alpha=90, outline_alpha=255):
    geoms = poly.geoms if poly.geom_type == "MultiPolygon" else [poly]
    for g in geoms:
        if g.is_empty:
            continue
        pts = list(g.exterior.coords)
        draw.polygon(pts, fill=fill_rgb + (alpha,), outline=fill_rgb + (outline_alpha,))


def draw_region_overlay(image: Image.Image, face_regions, max_size: int = 1400) -> Image.Image:
    """Semi-transparent region polygons (+ exclusion zones) over the face,
    for visual inspection of the landmark -> region mapping."""
    img = image.copy().convert("RGB")
    scale = min(1.0, max_size / max(img.size))
    if scale < 1.0:
        img = img.resize((round(img.width * scale), round(img.height * scale)), Image.BILINEAR)

    from shapely.affinity import scale as shapely_scale

    draw = ImageDraw.Draw(img, "RGBA")
    for name, poly in face_regions.regions.items():
        p = shapely_scale(poly, xfact=scale, yfact=scale, origin=(0, 0)) if scale < 1.0 else poly
        _draw_polygon(draw, p, REGION_COLORS[name])
    for name, poly in face_regions.exclusions.items():
        p = shapely_scale(poly, xfact=scale, yfact=scale, origin=(0, 0)) if scale < 1.0 else poly
        _draw_polygon(draw, p, EXCLUSION_COLORS[name], alpha=140)

    return img


def draw_detections(image: Image.Image, detections: list, max_size: int = 1400) -> Image.Image:
    """Detector boxes labeled with assigned region + confidence."""
    img = image.copy().convert("RGB")
    scale = min(1.0, max_size / max(img.size))
    if scale < 1.0:
        img = img.resize((round(img.width * scale), round(img.height * scale)), Image.BILINEAR)

    draw = ImageDraw.Draw(img)
    for det in detections:
        x1, y1, x2, y2 = [c * scale for c in det["box"]]
        color = UNASSIGNED_BOX_COLOR if det["region"] == "unassigned" else DETECTION_BOX_COLOR
        draw.rectangle([x1, y1, x2, y2], outline=color, width=3)
        label = f"{det['region']} {det['confidence']:.2f}"
        draw.text((x1 + 2, max(0, y1 - 14)), label, fill=color)

    return img
