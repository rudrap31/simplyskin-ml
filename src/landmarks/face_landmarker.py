"""Face landmark detection via MediaPipe Tasks Python Face Landmarker
(IMAGE mode). Pretrained, not trained/fine-tuned here.

Model asset: download once with:
    curl -L -o models/face_landmarker.task \\
        "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task"
(models/ is gitignored — this asset is not committed to the repo.)
"""
from dataclasses import dataclass, field
from pathlib import Path

import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MODEL_PATH = REPO_ROOT / "models" / "face_landmarker.task"
MODEL_ASSET_URL = "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task"

NUM_LANDMARKS = 478  # 468 base mesh points + 10 iris refinement points


@dataclass
class FaceLandmarkResult:
    face_detected: bool
    landmarks: list  # list of (x, y, z) normalized coords for the SELECTED face, or [] if none
    multiple_faces_detected: bool
    num_faces_found: int
    warnings: list = field(default_factory=list)
    model_metadata: dict = field(default_factory=dict)


def _face_bbox_area_and_center(face_landmarks) -> tuple:
    """Normalized-space bbox area + center, for comparing candidate faces
    without needing pixel dimensions (landmarks are already 0..1 normalized
    per-image, so relative size/centrality comparisons are valid directly)."""
    xs = [lm.x for lm in face_landmarks]
    ys = [lm.y for lm in face_landmarks]
    x_min, x_max = min(xs), max(xs)
    y_min, y_max = min(ys), max(ys)
    area = (x_max - x_min) * (y_max - y_min)
    center = ((x_min + x_max) / 2, (y_min + y_max) / 2)
    return area, center


def _select_primary_face(faces_landmarks: list) -> int:
    """Largest face by normalized bbox area; ties broken by distance to
    image center (0.5, 0.5) — smaller distance wins."""
    best_idx = 0
    best_key = None
    for i, face in enumerate(faces_landmarks):
        area, center = _face_bbox_area_and_center(face)
        dist_to_center = ((center[0] - 0.5) ** 2 + (center[1] - 0.5) ** 2) ** 0.5
        key = (-area, dist_to_center)  # maximize area, then minimize centrality distance
        if best_key is None or key < best_key:
            best_key = key
            best_idx = i
    return best_idx


class FaceLandmarkDetector:
    """Reusable wrapper — construct once, call detect() per image."""

    def __init__(self, model_path: Path = DEFAULT_MODEL_PATH, max_num_faces: int = 3):
        model_path = Path(model_path)
        if not model_path.exists():
            raise FileNotFoundError(
                f"Face Landmarker model asset not found at {model_path}. Download it with:\n"
                f'  curl -L -o {model_path} "{MODEL_ASSET_URL}"'
            )
        self.model_path = model_path
        self.max_num_faces = max_num_faces

        base_options = mp_python.BaseOptions(model_asset_path=str(model_path))
        options = mp_vision.FaceLandmarkerOptions(
            base_options=base_options,
            running_mode=mp_vision.RunningMode.IMAGE,
            num_faces=max_num_faces,
            output_face_blendshapes=False,
            output_facial_transformation_matrixes=False,
        )
        self._landmarker = mp_vision.FaceLandmarker.create_from_options(options)
        self._model_metadata = {
            "mediapipe_version": mp.__version__,
            "model_asset_path": str(model_path),
            "model_asset_url": MODEL_ASSET_URL,
            "max_num_faces": max_num_faces,
        }

    def detect(self, image_path: Path = None, mp_image=None) -> FaceLandmarkResult:
        """Pass either image_path (str/Path) or a pre-built mp.Image via
        mp_image (e.g. when the caller already has the image loaded)."""
        if mp_image is None:
            if image_path is None:
                raise ValueError("Must pass image_path or mp_image")
            mp_image = mp.Image.create_from_file(str(image_path))

        result = self._landmarker.detect(mp_image)
        num_faces = len(result.face_landmarks)

        if num_faces == 0:
            return FaceLandmarkResult(
                face_detected=False,
                landmarks=[],
                multiple_faces_detected=False,
                num_faces_found=0,
                warnings=["No face detected in image."],
                model_metadata=dict(self._model_metadata),
            )

        warnings = []
        multiple = num_faces > 1
        if multiple:
            warnings.append(
                f"{num_faces} faces detected; selected the largest/most central face and ignored the rest."
            )

        primary_idx = _select_primary_face(result.face_landmarks) if multiple else 0
        selected = result.face_landmarks[primary_idx]
        landmarks = [(lm.x, lm.y, lm.z) for lm in selected]

        return FaceLandmarkResult(
            face_detected=True,
            landmarks=landmarks,
            multiple_faces_detected=multiple,
            num_faces_found=num_faces,
            warnings=warnings,
            model_metadata=dict(self._model_metadata),
        )
