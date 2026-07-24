# SimplySkin ML

## 1. Overview

SimplySkin ML is the computer-vision pipeline behind SimplySkin's facial
acne analysis feature. It takes a user-submitted face photo and produces
objective, structured evidence about detected acne lesions and their
location on the face — replacing what was originally a single GPT-4.1
vision call asked to eyeball redness/acne/hydration scores directly from
the image.

Concretely, the pipeline detects facial acne lesions in a photo, maps
each detection to a facial region (forehead, cheeks, nose, chin, etc.)
using face landmarks, and emits a stable JSON contract summarizing lesion
counts and locations. This repo is deliberately scoped to producing that
objective CV evidence only — it does not infer lesion subtype, medical
cause, or generate consumer-facing advice; that interpretation happens
downstream in the SimplySkin backend via a vision LLM. The production
pipeline documented here powers the SimplySkin mobile app's face-scan
feature.

## 2. Architecture

```
Face Image
    │
    ▼
Faster R-CNN Lesion Detector
    │
    ▼
MediaPipe Face Landmarker
    │
    ▼
Facial Region Assignment
    │
    ▼
Structured JSON
    │
    ▼
Vision LLM  (SimplySkin backend, not in this repo)
    │
    ▼
Skin Analysis
```

- **Faster R-CNN Lesion Detector** — a single-class object detector
  (lesion vs. background) that finds candidate acne lesion bounding boxes
  in the photo.
- **MediaPipe Face Landmarker** — a pretrained model that locates 478
  facial landmark points, used to build a face-relative coordinate frame
  (this repo does no landmark training).
- **Facial Region Assignment** — the landmarks are used to build polygon
  regions (forehead, cheeks, nose, chin, etc.), and each detected lesion
  is assigned to whichever region it overlaps most.
- **Structured JSON** — a versioned, LLM-agnostic contract summarizing
  every detection plus aggregate stats (counts by region, dominant
  region, assigned/unassigned rate). See [`docs/SCHEMA.md`](docs/SCHEMA.md).
- **Vision LLM** — outside this repo. The SimplySkin backend feeds this
  JSON (as grounding evidence) plus the image into GPT-4.1 to produce the
  consumer-facing analysis.

## 3. Models

### Lesion Detector
- **Model**: Faster R-CNN, ResNet-50-FPN backbone (`torchvision`),
  pretrained on COCO then fine-tuned.
- **Dataset**: [ACNE04](https://github.com/xpwu95/ldl) — single class
  (lesion), fold 0, train/val split carved from the official trainval
  set, official test fold held out untouched.
- **Output**: bounding boxes + confidence scores for candidate acne
  lesions. No lesion subtype — that's what the experimental classifier
  (§5) explored separately.
- Frozen checkpoint: `artifacts/detector_v1/best.pth` (not committed —
  see §8).

### Face Landmarks
- **Model**: MediaPipe Face Landmarker (Tasks API, `IMAGE` running mode),
  pretrained `.task` asset — no training performed in this repo.
- **Used for**: building the face-relative region polygons that lesion
  detections get assigned to. Handles no-face and multiple-face cases
  explicitly (see `src/landmarks/face_landmarker.py`).

## 4. Production Pipeline

The pipeline actually running in the app (`src/inference/detector_landmarks_pipeline.py`,
schema version `detector_landmarks_v1`):

1. **Detector** — frozen Faster R-CNN finds lesion boxes above a
   confidence threshold (0.5).
2. **Facial landmarks** — MediaPipe locates the face; if none is found,
   the pipeline returns `face_detected: false` rather than guessing.
3. **Region assignment** — each detection is assigned to the
   facial-region polygon it overlaps most (≥10% of its area), falling
   back to which polygon contains its center point if no region clears
   that bar; detections outside the face or inside excluded zones
   (eyes/eyebrows/lips) are marked `unassigned` rather than forced into a
   region. Full rule set: `src/landmarks/regions.py::assign_detection_to_region`.
4. **JSON contract** — a stable, versioned result (`docs/SCHEMA.md`)
   independent of any particular LLM or UI.
5. **LLM** — the SimplySkin backend's `/api/face/ml` route feeds this
   JSON plus the image into GPT-4.1, instructed to treat the CV
   detections as ground truth for the acne score rather than re-counting
   lesions itself.

Region names (`forehead`, `image_left_cheek`, `image_right_cheek`,
`nose`, `chin`, `other_face`) are from the **viewer/image** perspective,
not the subject's mirrored anatomical left/right — see `docs/SCHEMA.md`
for why, and what the backend needs to normalize before calling this
pipeline.

## 5. Experimental Classifier

**Goal**: explore a two-stage detector → classifier pipeline that, in
addition to localizing lesions, would classify each one into a broad
subtype (`comedonal_like`, `inflammatory_like`, `deeper_inflammatory_like`,
`non_active_acne`).

**Dataset**: [AcneSCU](https://universe.roboflow.com/) — remapped from
10 fine-grained categories down to the 4 broad classes above.

**Findings**: the classifier performed well in isolation — 90.8%
accuracy / 0.844 macro F1 on a held-out AcneSCU test-crop evaluation.
End-to-end performance degraded when the classifier was fed real
detector-generated crops instead of AcneSCU's own annotation crops: on
unmodified detector boxes, the classifier initially predicted
`inflammatory_like` only 4.4% of the time despite that being the visually
correct class for most of those crops.

The primary limitation was a distribution shift between detector outputs
(trained on ACNE04) and classifier training data (AcneSCU). Detector-
generated lesion crops contained substantially different context and
bounding-box characteristics than the crops seen during classifier
training, leading to unreliable subtype predictions in the full
pipeline. A targeted fix — fine-tuning the classifier with crop-scale
augmentation so it saw AcneSCU-style crops re-cropped at detector-like
scales — measurably corrected this for 3 of the 4 classes (the same
unmodified detector boxes then predicted `inflammatory_like` 64.2% of
the time). One class, `comedonal_like`, remained unreliable across every
classifier version and crop scale tested, pointing to a distinct,
unresolved gap rather than a pure crop-scale issue.

Because the detector-only (landmarks) pipeline proved more robust and
reliable in production, the classifier remains in the repository as an
experimental component (`artifacts/classifier_v1`, `artifacts/classifier_v2`,
`src/train/train_classifier.py`, and the `src/scripts/*classifier*` /
`*crop_scale*` scripts) and may be revisited with improved datasets or
unified detector/classifier training data.

## 6. Evaluation

- **Detector**: validated on the ACNE04 val split — mAP@0.5 ≈ 0.36; a
  full threshold sweep found the best precision/recall balance
  (precision 0.47, recall 0.48, F1 0.47) at confidence threshold 0.5,
  which is what production uses. Official ACNE04 test-fold evaluation is
  still outstanding (see §9).
- **Face landmarks + region assignment**: technically validated (not a
  medical-accuracy eval) on 30 varied images — 23 real photos plus
  synthetic rotation/lighting/size/no-face edge cases. 100% face
  detection on real photos, 95.1% of detections assigned to a specific
  region. Full report: `runs/detector_landmarks_eval/report.html`
  (generated locally, not committed — see §8 to reproduce).
- **JSON contract**: schema-stability and end-to-end tests in
  `src/scripts/test_detector_landmarks_pipeline.py` (27 checks).
- **Classifier**: see §5 findings above; full metrics and diagnostics
  live under `runs/` when reproduced locally (`src/scripts/evaluate_classifier.py`,
  `src/scripts/crop_scale_diagnostic.py`, `src/scripts/comedonal_end_to_end_check.py`).

## 7. Repository Structure

```
src/
  data/         Dataset loaders, annotation remapping, crop generation (ACNE04, AcneSCU)
  models/       Model factories (detector, classifier)
  train/        Training loops, metrics, engines
  inference/    Production + experimental inference pipelines
  landmarks/    MediaPipe wrapper + facial region polygon construction
  scripts/      CLI entrypoints: data prep, training, evaluation, diagnostics, tests
  viz/          Visualization helpers

configs/        YAML training configs (detector + classifier)
docs/           JSON schema documentation + example output
artifacts/      Trained checkpoints + run notes (gitignored — see §8)
models/         Downloaded MediaPipe model asset (gitignored — see §8)
datasets/       Dataset files (gitignored — see §8)
runs/           Generated evaluation outputs (gitignored, reproducible via scripts)
```

## 8. Running Locally

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Download the MediaPipe Face Landmarker model asset
curl -L -o models/face_landmarker.task \
    "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task"

# 3. Datasets + trained checkpoints are not committed (large binaries).
#    Place ACNE04 / AcneSCU under datasets/, and detector/classifier
#    checkpoints under artifacts/<name>/best.pth — see configs/*.yaml
#    for the training commands that produce them.

# 4. Run inference on one image (production pipeline)
python3 src/scripts/run_detector_landmarks_pipeline.py --image path/to/face.jpg

# 5. Batch technical validation + HTML report
python3 src/scripts/evaluate_detector_landmarks_pipeline.py

# 6. Tests
python3 src/scripts/test_detector_landmarks_pipeline.py   # production pipeline
python3 src/scripts/test_classifier_pipeline.py           # experimental classifier
```

## 9. Future Work

- Improve lesion subtype classification using a unified detection/
  classification dataset (or joint training) instead of two separately-
  sourced datasets with different crop conventions.
- Specifically revisit `comedonal_like` recognition, which stayed
  unreliable even after the crop-scale fix that helped the other three
  classes.
- Run the still-outstanding official ACNE04 test-fold evaluation.
- Evaluate on more diverse skin tones and imaging conditions.
- Expand longitudinal skin progression tracking.
- Investigate end-to-end multi-task models (joint detection + subtype
  classification in one network) as an alternative to the two-stage
  approach.
