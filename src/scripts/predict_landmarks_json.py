"""Backend-facing entrypoint: runs the frozen detector_landmarks_v1
pipeline on one image and prints ONLY the JSON result to stdout (nothing
else — safe for a caller to `JSON.parse(stdout)`). All model-loading
noise (MediaPipe/absl/TFLite logs) goes to stderr, never stdout.

Usage:
    python3 src/scripts/predict_landmarks_json.py --image /path/to/face.jpg
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.inference.detector_landmarks_pipeline import run_pipeline_on_image
from src.inference.pipeline import load_detector, resolve_device
from src.landmarks.face_landmarker import FaceLandmarkDetector


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", type=str, required=True)
    parser.add_argument("--detector-checkpoint", type=str, default="artifacts/detector_v1/best.pth")
    parser.add_argument("--landmark-model", type=str, default=None)
    parser.add_argument("--score-threshold", type=float, default=0.5)
    parser.add_argument("--device", type=str, default="auto")
    args = parser.parse_args()

    device = resolve_device(args.device)
    detector, _ = load_detector(Path(args.detector_checkpoint), device)
    landmark_kwargs = {"model_path": Path(args.landmark_model)} if args.landmark_model else {}
    landmark_detector = FaceLandmarkDetector(**landmark_kwargs)

    result, _face_regions = run_pipeline_on_image(
        Path(args.image), detector, landmark_detector, device, score_threshold=args.score_threshold
    )

    print(json.dumps(result))


if __name__ == "__main__":
    main()
