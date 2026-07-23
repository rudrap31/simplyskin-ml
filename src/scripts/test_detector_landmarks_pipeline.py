"""Tests for the detector + facial-landmarks pipeline. Plain assert/print
style (mirrors smoke_test_detector.py / test_classifier_pipeline.py — no
pytest dependency in this repo).

Covers:
  1. Polygon overlap assignment
  2. Center-point fallback
  3. Detections outside the face
  4. Excluded eye/lip areas
  5. Empty detections
  6. No face detected
  7. Multiple faces (selection + warning)
  8. JSON schema stability
  9. End-to-end smoke test on a real fixture image

Usage:
    python3 src/scripts/test_detector_landmarks_pipeline.py
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from PIL import Image
from shapely.geometry import Polygon

from src.landmarks.face_landmarker import FaceLandmarkDetector, _select_primary_face
from src.landmarks.regions import FaceRegions, REGION_NAMES, assign_detection_to_region

REPO_ROOT = Path(__file__).resolve().parents[2]
FIXTURE_IMAGE = REPO_ROOT / "datasets" / "acnescu" / "train" / "210_jpg.rf.8228215ed0c1b99e7918c38956472441.jpg"


def check(name, condition):
    status = "PASS" if condition else "FAIL"
    print(f"[{status}] {name}")
    if not condition:
        raise AssertionError(name)


def make_synthetic_regions() -> FaceRegions:
    """Simple non-overlapping 100x100 squares laid out in a 3x2 grid, for
    deterministic assignment-logic testing independent of real landmarks."""
    def square(x0, y0):
        return Polygon([(x0, y0), (x0 + 100, y0), (x0 + 100, y0 + 100), (x0, y0 + 100)])

    regions = {
        "forehead": square(100, 0),
        "image_left_cheek": square(0, 100),
        "image_right_cheek": square(200, 100),
        "nose": square(100, 100),
        "chin": square(100, 200),
        "other_face": square(200, 200),
    }
    exclusions = {
        "left_eye": square(0, 0),  # top-left corner, isolated
        "right_eye": square(1000, 1000),  # far away, isolated
        "left_eyebrow": square(1000, 1100),
        "right_eyebrow": square(1000, 1200),
        "lips": square(100, 300),  # just below chin square
    }
    face_oval = Polygon([(0, 0), (300, 0), (300, 300), (0, 300)])
    return FaceRegions(regions=regions, exclusions=exclusions, face_oval=face_oval)


def test_polygon_overlap_assignment():
    print("\n1. Polygon overlap assignment")
    fr = make_synthetic_regions()
    # box fully inside "forehead" square (100,0)-(200,100)
    box = [110, 10, 190, 90]
    result = assign_detection_to_region(box, fr)
    check("box fully inside forehead assigns to forehead", result["region"] == "forehead")
    check("full-overlap assignment method is 'overlap'", result["assignment_method"] == "overlap")
    check("full overlap ratio is 1.0", abs(result["region_overlap_ratio"] - 1.0) < 1e-9)

    # box mostly in nose, slightly poking into forehead — nose should still win
    box2 = [110, 90, 190, 190]  # spans forehead(90-100) and nose(100-190): mostly nose
    result2 = assign_detection_to_region(box2, fr)
    check("box mostly overlapping nose assigns to nose", result2["region"] == "nose")


def test_center_point_fallback():
    print("\n2. Center-point fallback")
    fr = make_synthetic_regions()
    # box straddling forehead/image_left_cheek/nose/chin boundary corner (100,100)
    # — a region (nose) still clears the 10% threshold here, so this
    # resolves via overlap, not fallback (confirms overlap wins when clear).
    box = [50, 50, 150, 150]
    result = assign_detection_to_region(box, fr)
    check("straddling box with a clear-enough overlap resolves via 'overlap', not fallback", result["assignment_method"] == "overlap")

    # true fallback case: a wide, short box centered exactly on chin's
    # center (150, 250) but wide enough that its overlap with chin AND
    # other_face (the only two regions its y-range touches) both fall
    # below the 10% threshold -> forces centroid fallback. Center (150,250)
    # sits strictly inside chin's interior, not on any boundary.
    fr2 = make_synthetic_regions()
    box2 = [-850, 240, 1150, 260]  # area 2000*20=40000; chin/other_face overlap each 100*20=2000 -> ratio 0.05 < 0.10
    result2 = assign_detection_to_region(box2, fr2)
    check("ambiguous low-overlap box triggers centroid_fallback", result2["assignment_method"] == "centroid_fallback")
    check("centroid fallback correctly lands in chin (its center (150,250) is chin's interior)", result2["region"] == "chin")


def test_outside_face():
    print("\n3. Detections outside the face")
    fr = make_synthetic_regions()
    box = [500, 500, 600, 600]  # far outside face_oval (0,0)-(300,300)
    result = assign_detection_to_region(box, fr)
    check("box entirely outside face_oval is unassigned", result["region"] == "unassigned")
    check("assignment_method reports outside_face_oval", result["assignment_method"] == "outside_face_oval")


def test_excluded_regions():
    print("\n4. Excluded eye/lip areas")
    fr = make_synthetic_regions()
    # box centered inside the 'lips' exclusion square (100,300)-(200,400),
    # but that's outside face_oval (0-300) — use left_eye instead, which is
    # inside face_oval bounds (0,0)-(100,100) and doesn't overlap any assignable region
    box = [10, 10, 90, 90]  # inside left_eye exclusion square (0,0)-(100,100), no overlap with named regions
    result = assign_detection_to_region(box, fr)
    check("box centered in an excluded eye zone is unassigned", result["region"] == "unassigned")
    check("assignment_method reports excluded:left_eye", result["assignment_method"] == "excluded:left_eye")


def test_empty_detections():
    print("\n5. Empty detections (pipeline-level, not just assignment)")
    from src.inference.detector_landmarks_pipeline import _empty_summary

    summary = _empty_summary()
    check("empty summary has zero total_lesions", summary["total_lesions"] == 0)
    check("empty summary has None dominant_region", summary["dominant_region"] is None)
    check("empty summary counts_by_region covers all 6 regions at 0", all(v == 0 for v in summary["counts_by_region"].values()))


def test_no_face_detected():
    print("\n6. No face detected")
    import tempfile

    detector = FaceLandmarkDetector()
    with tempfile.TemporaryDirectory() as tmpdir:
        blank_path = Path(tmpdir) / "blank.jpg"
        Image.new("RGB", (400, 400), color=(10, 10, 10)).save(blank_path)
        result = detector.detect(image_path=blank_path)
    check("face_detected is False for a blank image", result.face_detected is False)
    check("landmarks list is empty", result.landmarks == [])
    check("a warning is recorded", len(result.warnings) > 0)


def test_multiple_faces_selection():
    print("\n7. Multiple faces (largest/most-central selection)")
    class FakeLM:
        def __init__(self, x, y):
            self.x, self.y, self.z = x, y, 0.0

    # face A: small, off-center (top-left corner)
    face_a = [FakeLM(0.05, 0.05), FakeLM(0.15, 0.05), FakeLM(0.15, 0.15), FakeLM(0.05, 0.15)]
    # face B: large, centered
    face_b = [FakeLM(0.2, 0.2), FakeLM(0.8, 0.2), FakeLM(0.8, 0.8), FakeLM(0.2, 0.8)]
    idx = _select_primary_face([face_a, face_b])
    check("largest face (face B) is selected over a small off-center face", idx == 1)

    # two same-size faces, one more central than the other
    face_c = [FakeLM(0.0, 0.0), FakeLM(0.3, 0.0), FakeLM(0.3, 0.3), FakeLM(0.0, 0.3)]  # corner
    face_d = [FakeLM(0.35, 0.35), FakeLM(0.65, 0.35), FakeLM(0.65, 0.65), FakeLM(0.35, 0.65)]  # centered
    idx2 = _select_primary_face([face_c, face_d])
    check("equal-size faces: more central one is selected", idx2 == 1)


EXPECTED_TOP_LEVEL_KEYS = {
    "pipeline_version", "face_detected", "multiple_faces_detected", "image",
    "detections", "summary", "quality", "models",
}


def test_json_schema_stability():
    print("\n8. JSON schema stability")
    from src.inference.detector_landmarks_pipeline import _empty_summary

    summary_keys = {
        "total_lesions", "assigned_lesions", "unassigned_lesions", "average_detection_confidence",
        "counts_by_region", "dominant_region", "cheek_count", "t_zone_count", "image_left_right_cheek_difference",
    }
    check("_empty_summary() has exactly the expected summary keys", set(_empty_summary().keys()) == summary_keys)
    check("_empty_summary() counts_by_region has exactly the 6 region names", set(_empty_summary()["counts_by_region"].keys()) == set(REGION_NAMES))
    # EXPECTED_TOP_LEVEL_KEYS checked against a real run in the e2e smoke test below (test 9)


def test_end_to_end_smoke():
    print("\n9. End-to-end smoke test (real fixture image)")
    if not FIXTURE_IMAGE.exists():
        print(f"  SKIPPED: fixture image not found at {FIXTURE_IMAGE}")
        return

    from src.inference.detector_landmarks_pipeline import run_pipeline_on_image
    from src.inference.pipeline import load_detector, resolve_device

    device = resolve_device("auto")
    detector_checkpoint = REPO_ROOT / "artifacts" / "detector_v1" / "best.pth"
    if not detector_checkpoint.exists():
        print(f"  SKIPPED: detector checkpoint not found at {detector_checkpoint}")
        return

    detector, _ = load_detector(detector_checkpoint, device)
    landmark_detector = FaceLandmarkDetector()

    result, face_regions = run_pipeline_on_image(FIXTURE_IMAGE, detector, landmark_detector, device, score_threshold=0.5)

    check("result has all expected top-level keys", EXPECTED_TOP_LEVEL_KEYS.issubset(result.keys()))
    check("result is JSON-serializable", isinstance(json.dumps(result), str))
    check("face_detected is True on a real face image", result["face_detected"] is True)
    check("face_regions is not None when face_detected", face_regions is not None)
    check("pipeline_version is set", result["pipeline_version"] == "detector_landmarks_v1")
    check("image width/height populated", result["image"]["width"] > 0 and result["image"]["height"] > 0)
    if result["detections"]:
        det = result["detections"][0]
        check("detection has all expected fields", {"box", "normalized_box", "confidence", "region", "region_overlap_ratio", "relative_face_area"}.issubset(det.keys()))
        check("detection region is a valid region name or 'unassigned'", det["region"] in REGION_NAMES + ["unassigned"])
        check("normalized_box values are in [0, 1]", all(0.0 <= c <= 1.0 for c in det["normalized_box"]))
    check(
        "summary total_lesions equals len(detections)",
        result["summary"]["total_lesions"] == len(result["detections"]),
    )
    check(
        "assigned + unassigned == total",
        result["summary"]["assigned_lesions"] + result["summary"]["unassigned_lesions"] == result["summary"]["total_lesions"],
    )


def main():
    test_polygon_overlap_assignment()
    test_center_point_fallback()
    test_outside_face()
    test_excluded_regions()
    test_empty_detections()
    test_no_face_detected()
    test_multiple_faces_selection()
    test_json_schema_stability()
    test_end_to_end_smoke()
    print("\nAll checks passed.")


if __name__ == "__main__":
    main()
