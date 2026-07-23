"""Run the detector + facial-landmarks pipeline on a single image. Saves
the region-polygon overlay, the labeled detection boxes, and the JSON
result.

Usage:
    python3 src/scripts/run_detector_landmarks_pipeline.py --image path/to/face.jpg
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from PIL import Image

from src.inference.detector_landmarks_pipeline import draw_detections, draw_region_overlay, run_pipeline_on_image
from src.inference.pipeline import load_detector, resolve_device
from src.landmarks.face_landmarker import FaceLandmarkDetector


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", type=str, required=True)
    parser.add_argument("--detector-checkpoint", type=str, default="artifacts/detector_v1/best.pth")
    parser.add_argument("--landmark-model", type=str, default=None, help="override models/face_landmarker.task path")
    parser.add_argument("--score-threshold", type=float, default=0.5)
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--device", type=str, default="auto")
    args = parser.parse_args()

    image_path = Path(args.image)
    device = resolve_device(args.device)
    print(f"device: {device}")

    detector, _ = load_detector(Path(args.detector_checkpoint), device)
    landmark_kwargs = {"model_path": Path(args.landmark_model)} if args.landmark_model else {}
    landmark_detector = FaceLandmarkDetector(**landmark_kwargs)

    result, face_regions = run_pipeline_on_image(
        image_path, detector, landmark_detector, device, score_threshold=args.score_threshold
    )

    output_dir = Path(args.output_dir) if args.output_dir else Path("runs/detector_landmarks_single") / image_path.stem
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(output_dir / "result.json", "w") as f:
        json.dump(result, f, indent=2)

    image = Image.open(image_path).convert("RGB")
    if face_regions is not None:
        overlay = draw_region_overlay(image, face_regions)
        overlay.save(output_dir / "region_overlay.jpg")
        detections_img = draw_detections(image, result["detections"])
        detections_img.save(output_dir / "detections.jpg")
    else:
        print("No face detected — skipping overlay images.")

    print(f"\nface_detected: {result['face_detected']}")
    if result["quality"]["warnings"]:
        print(f"warnings: {result['quality']['warnings']}")
    print(f"total_lesions: {result['summary']['total_lesions']}")
    print(f"counts_by_region: {result['summary']['counts_by_region']}")
    print(f"\nSaved: {output_dir / 'result.json'}")
    if face_regions is not None:
        print(f"Saved: {output_dir / 'region_overlay.jpg'}")
        print(f"Saved: {output_dir / 'detections.jpg'}")


if __name__ == "__main__":
    main()
