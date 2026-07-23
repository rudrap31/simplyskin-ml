"""Face-relative region polygons built from MediaPipe Face Mesh landmarks
(478-point topology: 468 base mesh + 10 iris points), and lesion-to-region
assignment.

Landmark indices below are the standard, widely-published MediaPipe Face
Mesh canonical topology (stable across the whole MediaPipe ecosystem).
Rather than trusting memorized "left"/"right" semantic labels for each
symmetric landmark pair (a real error-prone step), every symmetric pair is
resolved to actual image-left / image-right AT RUNTIME by comparing pixel
x-coordinates (see `_split_left_right`) — so a mislabeled pair only swaps
which named index is "the temple" vs "the other temple", never swaps which
SIDE of the actual photo a region ends up on.

IMPORTANT convention: "image_left_cheek"/"image_right_cheek" mean the
region appearing on the LEFT/RIGHT side of the IMAGE as displayed (i.e.
what a person looks at when viewing the photo), NOT the subject's own
anatomical left/right (which would be mirrored in a selfie). The
"image_" prefix is deliberate and load-bearing — it's the whole point of
the naming, not decoration:
  - These names are from the VIEWER/IMAGE coordinate perspective only.
    This repo has no notion of the subject's anatomical left/right.
  - If the backend needs subject-anatomical wording (e.g. "your left
    cheek" in a UI), IT must do that conversion — using its own knowledge
    of whether the photo was taken with a front camera (mirrored) or
    otherwise, which this repo cannot know from pixels alone.
  - Consequently, INPUT IMAGE ORIENTATION MUST BE NORMALIZED before
    calling this pipeline (e.g. EXIF-rotated to upright, not passed
    through pre-mirrored). This module assumes pixel (0,0) is the
    top-left of the image as a human would view it. If the caller feeds
    in a mirrored or rotated frame, "image_left_cheek" will faithfully
    describe the wrong physical side of the subject's face.

Regions are built as convex hulls of a handful of anchor landmarks per
region (temples, eye corners, nose alae, mouth corners, jaw points, brow
line) rather than exact stitched boundary rings — this is deliberately
approximate (per spec) and avoids polygon self-intersection bugs from
getting an exact point-ordering wrong. Visually verify via the overlay
script; documented as a known limitation in the stop-point report.
"""
from dataclasses import dataclass

from shapely.geometry import LineString, Point, Polygon
from shapely.ops import unary_union

# --- canonical MediaPipe Face Mesh landmark indices -------------------

# Face oval, in ring order (traces the boundary sequentially).
FACE_OVAL_IDX = [
    10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 397, 365,
    379, 378, 400, 377, 152, 148, 176, 149, 150, 136, 172, 58, 132, 93,
    234, 127, 162, 21, 54, 103, 67, 109,
]
# Position of the two temple/cheekbone points (widest points) within the ring above.
_TEMPLE_POS_A = FACE_OVAL_IDX.index(454)
_TEMPLE_POS_B = FACE_OVAL_IDX.index(234)
CHIN_TIP_IDX = 152

# Eye rings (16 pts each, ring order) — one arbitrary symmetric pair, resolved
# to image-left/right at runtime, not trusted as literal "left"/"right" here.
EYE_RING_A_IDX = [263, 249, 390, 373, 374, 380, 381, 382, 362, 466, 388, 387, 386, 385, 384, 398]
EYE_RING_B_IDX = [33, 7, 163, 144, 145, 153, 154, 155, 133, 246, 161, 160, 159, 158, 157, 173]

# Eyebrow lines (~10 pts each, not a closed ring — buffered into a thin band).
EYEBROW_LINE_A_IDX = [276, 283, 282, 295, 285, 300, 293, 334, 296, 336]
EYEBROW_LINE_B_IDX = [46, 53, 52, 65, 55, 70, 63, 105, 66, 107]

# Outer lips ring.
LIPS_OUTER_IDX = [61, 185, 40, 39, 37, 0, 267, 269, 270, 409, 291, 375, 321, 405, 314, 17, 84, 181, 91, 146]

# Single anchor points (symmetric pairs resolved to image-left/right at runtime).
EYE_OUTER_CORNER_PAIR = (263, 33)
EYE_INNER_CORNER_PAIR = (362, 133)
NOSE_ALA_PAIR = (327, 98)  # nostril wing / nose-side reference
MOUTH_CORNER_PAIR = (291, 61)
JAW_MID_PAIR = (379, 150)  # cheek lower-boundary reference
JAW_NEAR_CHIN_PAIR = (377, 148)  # chin-region side boundary reference
NOSE_BRIDGE_TOP_IDX = 168
UPPER_LIP_CENTER_IDX = 0
LOWER_LIP_CENTER_IDX = 17

MIN_MEANINGFUL_OVERLAP_RATIO = 0.10  # below this, treat overlap as "ambiguous" and fall back to centroid containment
EYEBROW_BAND_HALF_WIDTH_FRACTION = 0.035  # of inter-temple distance


@dataclass
class FaceRegions:
    regions: dict  # name -> shapely Polygon, for the 6 assignable regions
    exclusions: dict  # name -> shapely Polygon, for eyes/eyebrows/lips (not assignable)
    face_oval: Polygon


def _to_pixel(landmarks_norm, idx, img_w, img_h):
    x, y, _ = landmarks_norm[idx]
    return (x * img_w, y * img_h)


def _split_left_right(group_a, group_b):
    """Given two point groups (each a list of (x,y) pixel points) that form
    a symmetric pair, return (image_left_group, image_right_group) by
    comparing mean x — resolves left/right from actual pixel geometry, not
    from any assumed semantic labeling of which index set is which."""
    mean_a = sum(p[0] for p in group_a) / len(group_a)
    mean_b = sum(p[0] for p in group_b) / len(group_b)
    return (group_a, group_b) if mean_a <= mean_b else (group_b, group_a)


def _split_left_right_point(pt_a, pt_b):
    left, right = _split_left_right([pt_a], [pt_b])
    return left[0], right[0]


def build_face_regions(landmarks_norm, img_w: int, img_h: int) -> FaceRegions:
    """landmarks_norm: list of 478 (x, y, z) normalized MediaPipe landmarks.
    Returns pixel-space shapely Polygons for the 6 assignable regions plus
    the exclusion zones and the face oval."""
    P = lambda idx: _to_pixel(landmarks_norm, idx, img_w, img_h)

    # .buffer(0) repairs minor self-intersections from the exact landmark
    # ring order (a standard shapely idiom) — resulting polygon is the
    # union of the ring's lobes, which for eye/lip-sized contours is a
    # negligible visual difference from the intended shape.
    face_oval_pts = [P(i) for i in FACE_OVAL_IDX]
    face_oval = Polygon(face_oval_pts).buffer(0)

    eye_a_pts = [P(i) for i in EYE_RING_A_IDX]
    eye_b_pts = [P(i) for i in EYE_RING_B_IDX]
    left_eye_pts, right_eye_pts = _split_left_right(eye_a_pts, eye_b_pts)
    left_eye = Polygon(left_eye_pts).buffer(0)
    right_eye = Polygon(right_eye_pts).buffer(0)

    brow_a_pts = [P(i) for i in EYEBROW_LINE_A_IDX]
    brow_b_pts = [P(i) for i in EYEBROW_LINE_B_IDX]
    left_brow_pts, right_brow_pts = _split_left_right(brow_a_pts, brow_b_pts)

    temple_a, temple_b = P(FACE_OVAL_IDX[_TEMPLE_POS_A]), P(FACE_OVAL_IDX[_TEMPLE_POS_B])
    face_scale = Point(temple_a).distance(Point(temple_b))
    brow_buffer = face_scale * EYEBROW_BAND_HALF_WIDTH_FRACTION
    left_eyebrow = LineString(left_brow_pts).buffer(brow_buffer)
    right_eyebrow = LineString(right_brow_pts).buffer(brow_buffer)

    lips = Polygon([P(i) for i in LIPS_OUTER_IDX]).buffer(0)

    left_temple, right_temple = _split_left_right_point(temple_a, temple_b)

    eye_outer_a, eye_outer_b = P(EYE_OUTER_CORNER_PAIR[0]), P(EYE_OUTER_CORNER_PAIR[1])
    left_eye_outer, right_eye_outer = _split_left_right_point(eye_outer_a, eye_outer_b)

    eye_inner_a, eye_inner_b = P(EYE_INNER_CORNER_PAIR[0]), P(EYE_INNER_CORNER_PAIR[1])
    left_eye_inner, right_eye_inner = _split_left_right_point(eye_inner_a, eye_inner_b)

    ala_a, ala_b = P(NOSE_ALA_PAIR[0]), P(NOSE_ALA_PAIR[1])
    left_ala, right_ala = _split_left_right_point(ala_a, ala_b)

    mouth_a, mouth_b = P(MOUTH_CORNER_PAIR[0]), P(MOUTH_CORNER_PAIR[1])
    left_mouth, right_mouth = _split_left_right_point(mouth_a, mouth_b)

    jaw_mid_a, jaw_mid_b = P(JAW_MID_PAIR[0]), P(JAW_MID_PAIR[1])
    left_jaw_mid, right_jaw_mid = _split_left_right_point(jaw_mid_a, jaw_mid_b)

    jaw_chin_a, jaw_chin_b = P(JAW_NEAR_CHIN_PAIR[0]), P(JAW_NEAR_CHIN_PAIR[1])
    left_jaw_chin, right_jaw_chin = _split_left_right_point(jaw_chin_a, jaw_chin_b)

    chin_tip = P(CHIN_TIP_IDX)
    nose_bridge_top = P(NOSE_BRIDGE_TOP_IDX)
    upper_lip_center = P(UPPER_LIP_CENTER_IDX)
    lower_lip_center = P(LOWER_LIP_CENTER_IDX)

    # --- forehead: top-of-oval arc (from one temple, through top-center,
    # to the other temple) + the eyebrow line, per spec's instruction to
    # estimate the (landmark-less) forehead-top boundary from face
    # oval/temple points, bounded below by the eyebrow line. Isolated here
    # so this heuristic can be adjusted independently of everything else.
    # FACE_OVAL_IDX position 0 is the top-center point (landmark 10);
    # walking from _TEMPLE_POS_B (234, one temple) down through position 0
    # then on to _TEMPLE_POS_A (454, the other temple) traces the SHORT
    # (top) arc between the two temples — not the long way around through
    # the chin, which is what a naive single forward slice would give.
    forehead_top_arc_idx = FACE_OVAL_IDX[_TEMPLE_POS_B:] + FACE_OVAL_IDX[: _TEMPLE_POS_A + 1]
    forehead_top_arc_pts = [P(i) for i in forehead_top_arc_idx]
    forehead_hull_pts = forehead_top_arc_pts + left_brow_pts + right_brow_pts
    forehead = Polygon(forehead_hull_pts).convex_hull

    image_left_cheek = Polygon(
        [left_temple, left_eye_outer, left_ala, left_mouth, left_jaw_mid]
    ).convex_hull
    image_right_cheek = Polygon(
        [right_temple, right_eye_outer, right_ala, right_mouth, right_jaw_mid]
    ).convex_hull

    nose = Polygon(
        [nose_bridge_top, left_eye_inner, left_ala, upper_lip_center, right_ala, right_eye_inner]
    ).convex_hull

    # chin: originally just [lower_lip_center, jaw_near_chin pair, chin_tip]
    # — a narrow vertical strip. Visual review of runs/detector_landmarks_eval
    # (3 independent images) showed this systematically pushed real
    # jawline/chin lesions into "other_face" instead. Widened using the
    # mouth-corner and jaw-mid anchors (same points cheek's lower boundary
    # uses) so chin covers the full band below the mouth down to the jaw —
    # this does mean chin and cheek can now overlap near jaw_mid, resolved
    # the same way as any other overlapping region: greatest overlap ratio.
    chin = Polygon(
        [lower_lip_center, left_mouth, left_jaw_mid, left_jaw_chin, chin_tip, right_jaw_chin, right_jaw_mid, right_mouth]
    ).convex_hull

    exclusions = {
        "left_eye": left_eye,
        "right_eye": right_eye,
        "left_eyebrow": left_eyebrow,
        "right_eyebrow": right_eyebrow,
        "lips": lips,
    }
    exclusion_union = unary_union(list(exclusions.values()))

    def clip(poly):
        """Keep within the face oval, remove any overlap with excluded areas."""
        poly = poly.intersection(face_oval)
        poly = poly.difference(exclusion_union)
        return poly

    forehead = clip(forehead)
    nose = clip(nose)
    chin = clip(chin)
    # cheeks are built from wide temple-to-mouth anchors and, uncorrected,
    # overlap noticeably into the nose/nasolabial area — subtract the
    # already-computed nose region so the overlay reads cleanly.
    image_left_cheek = clip(image_left_cheek).difference(nose)
    image_right_cheek = clip(image_right_cheek).difference(nose)

    named_union = unary_union([forehead, image_left_cheek, image_right_cheek, nose, chin, exclusion_union])
    other_face = face_oval.difference(named_union)

    regions = {
        "forehead": forehead,
        "image_left_cheek": image_left_cheek,
        "image_right_cheek": image_right_cheek,
        "nose": nose,
        "chin": chin,
        "other_face": other_face,
    }

    return FaceRegions(regions=regions, exclusions=exclusions, face_oval=face_oval)


REGION_NAMES = ["forehead", "image_left_cheek", "image_right_cheek", "nose", "chin", "other_face"]


def assign_detection_to_region(box_xyxy, face_regions: FaceRegions) -> dict:
    """Assign one detector box (pixel xyxy) to a region.

    Rules (in order):
      1. Compute overlap area (box ∩ region polygon) for each of the 6
         assignable regions; overlap_ratio = overlap_area / box_area.
      2. If the best region's overlap_ratio >= MIN_MEANINGFUL_OVERLAP_RATIO
         (0.10), assign it. This is "overlap-based assignment".
      3. Otherwise (all overlaps are small/ambiguous — e.g. a box mostly
         outside the face, or straddling a region boundary with no single
         clearly-dominant region), fall back to the box's CENTER POINT:
         assign whichever region polygon contains it ("centroid fallback").
      4. If the center point falls inside an excluded zone (eyes/eyebrows/
         lips) or outside the face oval entirely, the detection is marked
         `unassigned` rather than forced into a region — this is checked
         BEFORE the centroid fallback, so an excluded/off-face center never
         gets silently assigned just because some other region's polygon
         happens to overlap the box slightly.

    Returns dict with region, region_overlap_ratio, assignment_method.
    """
    x1, y1, x2, y2 = box_xyxy
    box_poly = Polygon([(x1, y1), (x2, y1), (x2, y2), (x1, y2)])
    box_area = box_poly.area
    center = Point((x1 + x2) / 2, (y1 + y2) / 2)

    overlaps = {}
    for name in REGION_NAMES:
        region_poly = face_regions.regions[name]
        if region_poly.is_empty or box_area == 0:
            overlaps[name] = 0.0
            continue
        inter_area = box_poly.intersection(region_poly).area
        overlaps[name] = inter_area / box_area

    best_region = max(overlaps, key=overlaps.get)
    best_ratio = overlaps[best_region]

    if best_ratio >= MIN_MEANINGFUL_OVERLAP_RATIO:
        return {"region": best_region, "region_overlap_ratio": best_ratio, "assignment_method": "overlap"}

    # excluded-zone / off-face center check, before centroid fallback
    for excl_name, excl_poly in face_regions.exclusions.items():
        if excl_poly.contains(center):
            return {"region": "unassigned", "region_overlap_ratio": 0.0, "assignment_method": f"excluded:{excl_name}"}
    if not face_regions.face_oval.contains(center):
        return {"region": "unassigned", "region_overlap_ratio": 0.0, "assignment_method": "outside_face_oval"}

    for name in REGION_NAMES:
        if face_regions.regions[name].contains(center):
            return {"region": name, "region_overlap_ratio": overlaps[name], "assignment_method": "centroid_fallback"}

    return {"region": "unassigned", "region_overlap_ratio": 0.0, "assignment_method": "no_containment"}
