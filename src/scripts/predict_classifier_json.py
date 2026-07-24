"""Backend-facing entrypoint: runs the EXPERIMENTAL detector + lesion-
classifier pipeline on one image and prints ONLY the JSON result to
stdout. Separate, differently-shaped contract from detector_landmarks_v1
(no face/landmark step here — see docs/SCHEMA.md for why the two aren't
unified: this pipeline has no notion of facial region, only per-lesion
subtype). Not the default production pipeline; kept for experimentation
per artifacts/README.md.

Usage:
    python3 src/scripts/predict_classifier_json.py --image /path/to/face.jpg
"""
import argparse
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from PIL import Image

from src.inference.pipeline import load_classifier, load_detector, resolve_device
from src.inference.pipeline import run_pipeline_on_image as run_detector_classifier


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", type=str, required=True)
    parser.add_argument("--detector-checkpoint", type=str, default="artifacts/detector_v1/best.pth")
    parser.add_argument("--classifier-checkpoint", type=str, default="artifacts/classifier_v2/best.pth")
    parser.add_argument("--score-threshold", type=float, default=0.5)
    parser.add_argument("--device", type=str, default="auto")
    args = parser.parse_args()

    device = resolve_device(args.device)
    detector, _ = load_detector(Path(args.detector_checkpoint), device)
    classifier, _, class_names = load_classifier(Path(args.classifier_checkpoint), device)

    image = Image.open(args.image).convert("RGB")
    img_w, img_h = image.size

    raw_detections = run_detector_classifier(
        image, detector, classifier, class_names, device, detector_score_threshold=args.score_threshold
    )

    detections = []
    confidences = []
    class_counts = Counter()
    for det in raw_detections:
        x1, y1, x2, y2 = det["box"]
        record = {
            "box": [x1, y1, x2, y2],
            "normalized_box": [x1 / img_w, y1 / img_h, x2 / img_w, y2 / img_h],
            "confidence": det["detector_score"],
            "predicted_class": det.get("predicted_class"),
            "class_confidence": det.get("class_confidence"),
        }
        detections.append(record)
        confidences.append(det["detector_score"])
        if det.get("predicted_class"):
            class_counts[det["predicted_class"]] += 1

    result = {
        "pipeline_version": "detector_classifier_v1",
        "experimental": True,
        "image": {"width": img_w, "height": img_h},
        "detections": detections,
        "summary": {
            "total_lesions": len(detections),
            "counts_by_class": dict(class_counts),
            "average_detection_confidence": sum(confidences) / len(confidences) if confidences else 0.0,
        },
        "models": {
            "detector": "fasterrcnn_resnet50_fpn (artifacts/detector_v1/best.pth)",
            "classifier": f"efficientnet_b0 ({args.classifier_checkpoint})",
        },
    }

    print(json.dumps(result))


if __name__ == "__main__":
    main()
