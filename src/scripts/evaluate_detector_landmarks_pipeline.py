"""Batch technical validation of the detector + facial-landmarks pipeline
on ~20-30 varied face images. NOT a medical-accuracy evaluation — this
checks landmark detection, region construction, detector inference, and
spatial assignment behave sensibly.

Image set: a spread of real AcneSCU photos (varied acne severity/framing,
all this dataset naturally offers) plus a handful of synthetically
generated edge cases (rotation, lighting, no-face, non-face crop) since
AcneSCU's photos are all standardized clinical frontal shots with
consistent lighting — clearly labeled as synthetic in the report.

Usage:
    python3 src/scripts/evaluate_detector_landmarks_pipeline.py
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from PIL import Image, ImageEnhance

from src.inference.detector_landmarks_pipeline import draw_detections, draw_region_overlay, run_pipeline_on_image
from src.inference.pipeline import load_detector, resolve_device
from src.landmarks.face_landmarker import FaceLandmarkDetector

REPO_ROOT = Path(__file__).resolve().parents[2]
ACNESCU_DIR = REPO_ROOT / "datasets" / "acnescu" / "train"
OUTPUT_DIR = REPO_ROOT / "runs" / "detector_landmarks_eval"
SCRATCH_DIR = OUTPUT_DIR / "_synthetic_inputs"

BASE_IMAGE_NAMES = [
    "0_jpg.rf.af0ac71c9dc4de657316d7062ca12768.jpg",
    "109_jpg.rf.4344ef00511a488ffcc9f98c548655e4.jpg",
    "12_jpg.rf.4fbbaab01cdb0f03f7c1b0e14eb7fbb2.jpg",
    "130_jpg.rf.ab30f3e257654c4c18111477b66c6838.jpg",
    "141_jpg.rf.aab8eceaeb91b4089a676099707effe9.jpg",
    "152_jpg.rf.034452579247bf15be6b7ffc87bba10e.jpg",
    "164_jpg.rf.cd329f99784079452ecb4e8c6bdc5271.jpg",
    "175_jpg.rf.3dfcd0b4944cc7367f17fe9e4fce54d4.jpg",
    "186_jpg.rf.a9b3154e825c4a3112dd8bee6e8cad0f.jpg",
    "197_jpg.rf.346eb773738e486f9914ad6293b19227.jpg",
    "207_jpg.rf.1003700a1ed67793480a0edd12794a2b.jpg",
    "218_jpg.rf.6504747496f11fa836adca43ff30e4ae.jpg",
    "229_jpg.rf.bace6ad3566944279e9a6b5d663e463c.jpg",
    "24_jpg.rf.7a07090004f700901900d04e108d179e.jpg",
    "250_jpg.rf.5f3dd6e8ed3febb450a4b0ee57377bed.jpg",
    "261_jpg.rf.a68767fa16e6b8d7794dea935eba9286.jpg",
    "272_jpg.rf.aff5129cc76f2596237168da16f81d07.jpg",
    "35_jpg.rf.7254eb47c188a6e08e2b935be4400986.jpg",
    "46_jpg.rf.f9dab7d07dd8cd5705319b37da6dd893.jpg",
    "57_jpg.rf.83b8db2a8e78e1a9bb9fd37c8dc17281.jpg",
    "68_jpg.rf.0b1f6073dbb9d86c0c0f300ef9fda535.jpg",
    "79_jpg.rf.e84eecb189dc5ea10d3ffbe5ea7e7fb8.jpg",
    "9_jpg.rf.830767e18c27dae0992207691c2bb39a.jpg",
]


def make_synthetic_cases() -> list:
    """Returns list of (case_id, image_path, category, is_synthetic).
    Generates the synthetic variants once and writes them to SCRATCH_DIR."""
    SCRATCH_DIR.mkdir(parents=True, exist_ok=True)
    cases = []

    src = Image.open(ACNESCU_DIR / "68_jpg.rf.0b1f6073dbb9d86c0c0f300ef9fda535.jpg").convert("RGB")

    rotated_15 = src.rotate(-15, expand=True, fillcolor=(0, 0, 0))
    p = SCRATCH_DIR / "synthetic_rotated_15deg.jpg"
    rotated_15.save(p)
    cases.append(("synthetic_rotated_15deg", p, "slightly rotated face", True))

    rotated_90 = src.rotate(-90, expand=True, fillcolor=(0, 0, 0))
    p = SCRATCH_DIR / "synthetic_rotated_90deg.jpg"
    rotated_90.save(p)
    cases.append(("synthetic_rotated_90deg", p, "landmark edge case: extreme rotation", True))

    darker = ImageEnhance.Brightness(src).enhance(0.35)
    p = SCRATCH_DIR / "synthetic_dark_lighting.jpg"
    darker.save(p)
    cases.append(("synthetic_dark_lighting", p, "different lighting (dark)", True))

    brighter = ImageEnhance.Brightness(src).enhance(1.8)
    p = SCRATCH_DIR / "synthetic_bright_lighting.jpg"
    brighter.save(p)
    cases.append(("synthetic_bright_lighting", p, "different lighting (bright/overexposed)", True))

    small_face = Image.new("RGB", (src.width * 3, src.height * 3), color=(30, 30, 30))
    small_face.paste(src.resize((src.width // 3, src.height // 3)), (src.width, src.height))
    p = SCRATCH_DIR / "synthetic_small_face.jpg"
    small_face.save(p)
    cases.append(("synthetic_small_face", p, "different face size (small, padded)", True))

    blank = Image.new("RGB", (800, 800), color=(20, 20, 20))
    p = SCRATCH_DIR / "synthetic_no_face.jpg"
    blank.save(p)
    cases.append(("synthetic_no_face", p, "landmark failure case: no face present", True))

    non_face_crop = src.crop((0, 0, 300, 200))  # top-left corner, background only
    p = SCRATCH_DIR / "synthetic_non_face_crop.jpg"
    non_face_crop.save(p)
    cases.append(("synthetic_non_face_crop", p, "landmark failure case: non-face crop", True))

    return cases


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    device = resolve_device("auto")
    print(f"device: {device}")

    detector, _ = load_detector(REPO_ROOT / "artifacts" / "detector_v1" / "best.pth", device)
    landmark_detector = FaceLandmarkDetector()

    cases = [(name.replace(".jpg", ""), ACNESCU_DIR / name, "real AcneSCU photo (varied severity/framing)", False) for name in BASE_IMAGE_NAMES]
    cases += make_synthetic_cases()

    print(f"Evaluating {len(cases)} images ({len(BASE_IMAGE_NAMES)} real, {len(cases) - len(BASE_IMAGE_NAMES)} synthetic)...")

    report_rows = []
    for case_id, image_path, category, is_synthetic in cases:
        case_dir = OUTPUT_DIR / case_id
        case_dir.mkdir(parents=True, exist_ok=True)

        image = Image.open(image_path).convert("RGB")
        image.save(case_dir / "original.jpg")

        try:
            result, face_regions = run_pipeline_on_image(image_path, detector, landmark_detector, device, score_threshold=0.5)
        except Exception as e:
            print(f"  [{case_id}] ERROR: {e}")
            report_rows.append(
                {"case_id": case_id, "category": category, "is_synthetic": is_synthetic, "error": str(e)}
            )
            continue

        with open(case_dir / "result.json", "w") as f:
            json.dump(result, f, indent=2)

        if face_regions is not None:
            draw_region_overlay(image, face_regions).save(case_dir / "region_overlay.jpg")
            draw_detections(image, result["detections"]).save(case_dir / "detections.jpg")

        summary = result["summary"]
        report_rows.append(
            {
                "case_id": case_id,
                "category": category,
                "is_synthetic": is_synthetic,
                "face_detected": result["face_detected"],
                "multiple_faces_detected": result["multiple_faces_detected"],
                "warnings": result["quality"]["warnings"],
                "total_lesions": summary["total_lesions"],
                "assigned_lesions": summary["assigned_lesions"],
                "unassigned_lesions": summary["unassigned_lesions"],
                "counts_by_region": summary["counts_by_region"],
                "dominant_region": summary["dominant_region"],
            }
        )
        print(
            f"  [{case_id}] face_detected={result['face_detected']}  "
            f"lesions={summary['total_lesions']} (unassigned={summary['unassigned_lesions']})  "
            f"warnings={result['quality']['warnings']}"
        )

    with open(OUTPUT_DIR / "eval_summary.json", "w") as f:
        json.dump(report_rows, f, indent=2)

    generate_html_report(report_rows, OUTPUT_DIR)
    print(f"\nSaved: {OUTPUT_DIR / 'eval_summary.json'}")
    print(f"Saved: {OUTPUT_DIR / 'report.html'}")


def generate_html_report(rows: list, output_dir: Path):
    valid_rows = [r for r in rows if "error" not in r]
    n = len(rows)
    n_face_detected = sum(1 for r in valid_rows if r.get("face_detected"))
    n_no_face = sum(1 for r in valid_rows if not r.get("face_detected"))
    n_multi_face = sum(1 for r in valid_rows if r.get("multiple_faces_detected"))
    n_errors = sum(1 for r in rows if "error" in r)

    total_lesions = sum(r.get("total_lesions", 0) for r in valid_rows)
    total_assigned = sum(r.get("assigned_lesions", 0) for r in valid_rows)
    total_unassigned = sum(r.get("unassigned_lesions", 0) for r in valid_rows)
    assigned_rate = total_assigned / total_lesions if total_lesions else 0.0

    region_totals = {}
    for r in valid_rows:
        for region, count in r.get("counts_by_region", {}).items():
            region_totals[region] = region_totals.get(region, 0) + count

    rows_html = []
    for r in rows:
        case_id = r["case_id"]
        if "error" in r:
            rows_html.append(
                f"<tr class='error-row'><td>{case_id}</td><td>{r['category']}</td>"
                f"<td colspan='6'>ERROR: {r['error']}</td></tr>"
            )
            continue
        synth_tag = " (synthetic)" if r["is_synthetic"] else ""
        warn_str = "; ".join(r["warnings"]) if r["warnings"] else ""
        region_str = ", ".join(f"{k}={v}" for k, v in r["counts_by_region"].items() if v > 0) or "(none)"
        row_class = "no-face-row" if not r["face_detected"] else ""
        thumb = f"{case_id}/detections.jpg" if r["face_detected"] else f"{case_id}/original.jpg"
        rows_html.append(
            f"<tr class='{row_class}'>"
            f"<td><img src='{thumb}' class='thumb'></td>"
            f"<td>{case_id}{synth_tag}<br><span class='cat'>{r['category']}</span></td>"
            f"<td>{r['face_detected']}</td>"
            f"<td>{r['multiple_faces_detected']}</td>"
            f"<td>{r['total_lesions']}</td>"
            f"<td>{r['assigned_lesions']}/{r['unassigned_lesions']}</td>"
            f"<td>{region_str}</td>"
            f"<td>{warn_str}</td>"
            f"</tr>"
        )

    html = f"""<!doctype html>
<html><head><meta charset="utf-8"><title>Detector+Landmarks Pipeline — Technical Validation</title>
<style>
body {{ font-family: -apple-system, sans-serif; margin: 24px; color: #222; }}
h1, h2 {{ margin-top: 1.5em; }}
table {{ border-collapse: collapse; width: 100%; margin-top: 8px; }}
th, td {{ border: 1px solid #ccc; padding: 6px 8px; text-align: left; font-size: 13px; vertical-align: top; }}
th {{ background: #f0f0f0; }}
.thumb {{ width: 140px; height: auto; }}
.cat {{ color: #666; font-size: 11px; }}
.no-face-row {{ background: #fff3cd; }}
.error-row {{ background: #f8d7da; }}
.stat-grid {{ display: flex; gap: 24px; flex-wrap: wrap; }}
.stat-box {{ border: 1px solid #ddd; border-radius: 8px; padding: 12px 18px; background: #fafafa; }}
.stat-box .num {{ font-size: 24px; font-weight: bold; }}
</style></head>
<body>
<h1>Detector + Facial-Landmarks Pipeline — Technical Validation</h1>
<p>This is a technical validation of landmark detection, region construction, detector inference, and
spatial assignment. It is NOT a medical-accuracy evaluation.</p>

<h2>Overview</h2>
<div class="stat-grid">
  <div class="stat-box"><div class="num">{n}</div>images evaluated</div>
  <div class="stat-box"><div class="num">{n_face_detected}</div>face detected</div>
  <div class="stat-box"><div class="num">{n_no_face}</div>no face detected</div>
  <div class="stat-box"><div class="num">{n_multi_face}</div>multiple faces detected</div>
  <div class="stat-box"><div class="num">{n_errors}</div>pipeline errors</div>
  <div class="stat-box"><div class="num">{total_lesions}</div>total detections (score&ge;0.5)</div>
  <div class="stat-box"><div class="num">{assigned_rate:.1%}</div>assigned vs. unassigned rate</div>
</div>

<h2>Lesion counts by region (all images combined)</h2>
<table><tr><th>region</th><th>count</th></tr>
{"".join(f"<tr><td>{k}</td><td>{v}</td></tr>" for k, v in sorted(region_totals.items(), key=lambda kv: -kv[1]))}
</table>

<h2>Per-image results</h2>
<table>
<tr><th>preview</th><th>case</th><th>face_detected</th><th>multi_face</th><th>total_lesions</th><th>assigned/unassigned</th><th>counts_by_region</th><th>warnings</th></tr>
{"".join(rows_html)}
</table>

</body></html>
"""
    with open(output_dir / "report.html", "w") as f:
        f.write(html)


if __name__ == "__main__":
    main()
