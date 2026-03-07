# =============================================================================
# team_classifier.py — Jersey-colour-based team separation for detected players
#
# Public API:
#   extract_player_crops(image, detections)  -> list[np.ndarray]
#   get_dominant_color(player_crop)          -> np.ndarray  (RGB triplet)
#   cluster_teams(player_crops)              -> list[dict]
#   classify_teams(image, detections)        -> DetectionResult  (full pipeline)
#   draw_teams(image, detections)            -> np.ndarray
#
# Pipeline:
#   1. extract_player_crops  — crop jersey regions for every player/goalkeeper
#   2. get_dominant_color    — KMeans(k=2) per crop to isolate jersey vs background
#   3. cluster_teams         — KMeans(k=2) across all dominant colors → Team A / B
#   4. classify_teams        — ties the pipeline to the detect.py DetectionResult format
# =============================================================================

from __future__ import annotations

import warnings

import cv2
import numpy as np
from sklearn.cluster import KMeans
from typing import TypedDict


# ---------------------------------------------------------------------------
# Types (compatible with detect.py DetectionResult)
# ---------------------------------------------------------------------------

class BBox(TypedDict):
    label: str
    confidence: float
    bbox: list[float]
    team: str | None   # "team_a" | "team_b" | None for non-player roles


class DetectionResult(TypedDict):
    detections: list[BBox]


# ---------------------------------------------------------------------------
# Step 1 — Extract player crops from the frame
# ---------------------------------------------------------------------------

def extract_player_crops(
    image: np.ndarray,
    detections: DetectionResult,
) -> tuple[list[np.ndarray], list[int]]:
    """
    Crop the bounding box region of every player and goalkeeper from the frame.

    Only "player" and "goalkeeper" labels are extracted — referees and the
    ball are excluded since they don't belong to either team.

    The crop uses only the upper 60% of each bounding box (jersey torso region)
    to avoid the shorts and legs, which are often a different color and would
    corrupt the dominant-color extraction.

    Args:
        image:      Full frame in BGR format (H × W × 3 numpy array).
        detections: DetectionResult from detect.detect_objects().

    Returns:
        Tuple of:
          - crops:   List of BGR crop arrays, one per matched detection.
          - indices: Corresponding indices into detections["detections"],
                     so callers can map cluster results back to the original list.
    """
    crops: list[np.ndarray] = []
    indices: list[int] = []

    h_img, w_img = image.shape[:2]

    for i, det in enumerate(detections["detections"]):
        # Only classify player and goalkeeper — they belong to teams
        if det["label"] not in ("player", "goalkeeper"):
            continue

        x1, y1, x2, y2 = [int(v) for v in det["bbox"]]

        # Clamp to image bounds
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w_img, x2), min(h_img, y2)

        if x2 - x1 < 6 or y2 - y1 < 6:
            continue

        # Upper 60% of the box → jersey torso; lower 40% → shorts/legs (skip)
        y_jersey_end = y1 + int((y2 - y1) * 0.6)
        crop = image[y1:y_jersey_end, x1:x2]

        if crop.size == 0:
            continue

        crops.append(crop)
        indices.append(i)

    return crops, indices


# ---------------------------------------------------------------------------
# Step 2 — Dominant jersey color per crop
# ---------------------------------------------------------------------------

def get_dominant_color(player_crop: np.ndarray, n_colors: int = 2) -> np.ndarray:
    """
    Extract the dominant jersey color from a single player crop using KMeans.

    Steps:
      1. Resize the crop to 32×64 px for faster processing.
      2. Convert BGR → RGB.
      3. Reshape to a flat list of pixels (N, 3).
      4. Mask out green pitch pixels (HSV-based) so grass bleed-in is ignored.
      5. Run KMeans(k=2) — the two clusters typically separate jersey vs shadow.
      6. Return the centroid of the larger cluster as the dominant color.

    Args:
        player_crop: BGR crop array (H × W × 3, uint8).
        n_colors:    Number of KMeans clusters (default 2: jersey + background).

    Returns:
        Dominant color as a float32 RGB array of shape (3,).
        Returns grey [128, 128, 128] if the crop is too small or all-grass.
    """
    if player_crop.size == 0:
        return np.array([128.0, 128.0, 128.0], dtype=np.float32)

    # Resize for speed — small crops process in microseconds
    resized = cv2.resize(player_crop, (32, 64), interpolation=cv2.INTER_AREA)

    # Convert to RGB (KMeans will return RGB centroids)
    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)

    # Mask out green pitch pixels in HSV space
    hsv = cv2.cvtColor(resized, cv2.COLOR_BGR2HSV)
    grass_mask = cv2.inRange(hsv, (35, 40, 40), (85, 255, 255))
    jersey_mask = grass_mask == 0  # True where pixel is NOT grass

    pixels = rgb.reshape(-1, 3)[jersey_mask.flatten()]

    # Fall back to all pixels if almost nothing survived the grass mask
    if len(pixels) < 20:
        pixels = rgb.reshape(-1, 3)

    pixels = pixels.astype(np.float32)

    n_unique = len(np.unique(pixels, axis=0))
    k = min(n_colors, n_unique)
    if k < 1:
        return np.array([128.0, 128.0, 128.0], dtype=np.float32)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=FutureWarning)
        warnings.filterwarnings("ignore", message=".*ConvergenceWarning.*")
        warnings.filterwarnings("ignore", message=".*Number of distinct clusters.*")
        km = KMeans(n_clusters=k, n_init=3, random_state=42)
        km.fit(pixels)

    counts = np.bincount(km.labels_)
    dominant_idx = int(np.argmax(counts))
    return km.cluster_centers_[dominant_idx].astype(np.float32)


# ---------------------------------------------------------------------------
# Step 3 — Cluster all dominant colors into two teams
# ---------------------------------------------------------------------------

def cluster_teams(
    player_crops: list[np.ndarray],
) -> list[dict[str, int | str]]:
    """
    Assign each player crop to Team A or Team B by clustering dominant colors.

    Steps:
      1. Compute the dominant jersey color for every crop via get_dominant_color().
      2. Stack all colors into a feature matrix (N, 3).
      3. Run KMeans(k=2) to partition players into two jersey-color groups.
      4. Map cluster 0 → "A", cluster 1 → "B".

    Args:
        player_crops: List of BGR crop arrays (one per player/goalkeeper).

    Returns:
        List of assignment dicts, one per crop, in the same order as the input:
        [
            {"player_id": 0, "team": "A"},
            {"player_id": 1, "team": "B"},
            ...
        ]
        If fewer than 2 crops are provided, all players are assigned to "A"
        (cannot cluster with only one sample).
    """
    if not player_crops:
        return []

    # Compute dominant jersey color for every crop
    dominant_colors = np.array(
        [get_dominant_color(crop) for crop in player_crops],
        dtype=np.float32,
    )  # shape: (N, 3)

    # Need at least 2 distinct players to form 2 clusters
    if len(dominant_colors) < 2:
        return [{"player_id": 0, "team": "A"}]

    n_unique = len(np.unique(dominant_colors, axis=0))
    k = min(2, len(dominant_colors), n_unique)
    if k < 2:
        return [{"player_id": i, "team": "A"} for i in range(len(player_crops))]

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        km = KMeans(n_clusters=k, n_init=10, random_state=42)
        km.fit(dominant_colors)

    assignments: list[dict[str, int | str]] = []
    for player_id, cluster_label in enumerate(km.labels_):
        assignments.append({
            "player_id": player_id,
            "team": "A" if int(cluster_label) == 0 else "B",
        })

    return assignments


# ---------------------------------------------------------------------------
# Stateful classifier — consistent team labels across video frames
# ---------------------------------------------------------------------------

class TeamClassifier:
    """
    Maintains reference jersey-color centroids so Team A / Team B labels
    stay consistent across frames.  Create one instance per video session.

    First frame with enough players: KMeans establishes initial centroids.
    Subsequent frames: each player is assigned to the nearest reference
    centroid (no KMeans re-run), with a slow EMA update to handle
    gradual lighting changes.
    """

    def __init__(self, ema_alpha: float = 0.05) -> None:
        self._ref_a: np.ndarray | None = None
        self._ref_b: np.ndarray | None = None
        self._ema_alpha = ema_alpha
        self._initialized = False

    def classify_frame(
        self,
        frame: np.ndarray,
        detections: dict,
    ) -> tuple[list[dict], list[int]]:
        """
        Classify players into teams with temporal consistency.

        Returns:
            (assignments, player_indices) where assignments is
            [{"player_id": int, "team": "A"|"B"}, ...] and player_indices
            maps player_id → index in detections["detections"].
        """
        crops, indices = extract_player_crops(frame, detections)

        if len(crops) < 2:
            return [], indices

        colors = np.array(
            [get_dominant_color(crop) for crop in crops],
            dtype=np.float32,
        )

        if not self._initialized:
            assignments = cluster_teams(crops)

            a_colors = [colors[a["player_id"]] for a in assignments if a["team"] == "A"]
            b_colors = [colors[a["player_id"]] for a in assignments if a["team"] == "B"]

            if a_colors and b_colors:
                self._ref_a = np.mean(a_colors, axis=0).astype(np.float32)
                self._ref_b = np.mean(b_colors, axis=0).astype(np.float32)
                self._initialized = True

            return assignments, indices

        assignments: list[dict] = []
        new_a: list[np.ndarray] = []
        new_b: list[np.ndarray] = []

        for pid, color in enumerate(colors):
            dist_a = float(np.linalg.norm(color - self._ref_a))
            dist_b = float(np.linalg.norm(color - self._ref_b))
            team = "A" if dist_a <= dist_b else "B"
            assignments.append({"player_id": pid, "team": team})

            if team == "A":
                new_a.append(color)
            else:
                new_b.append(color)

        alpha = self._ema_alpha
        if new_a:
            self._ref_a = (1 - alpha) * self._ref_a + alpha * np.mean(new_a, axis=0)
        if new_b:
            self._ref_b = (1 - alpha) * self._ref_b + alpha * np.mean(new_b, axis=0)

        return assignments, indices

    def assign_by_color(self, frame: np.ndarray, bbox: list[float]) -> str | None:
        """Classify a single player by jersey color against stored centroids."""
        if not self._initialized:
            return None
        x1, y1, x2, y2 = [int(v) for v in bbox]
        h_img, w_img = frame.shape[:2]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w_img, x2), min(h_img, y2)
        if x2 - x1 < 2 or y2 - y1 < 2:
            return None
        y_jersey = y1 + int((y2 - y1) * 0.6)
        crop = frame[y1:y_jersey, x1:x2]
        if crop.size == 0:
            return None
        color = get_dominant_color(crop)
        dist_a = float(np.linalg.norm(color - self._ref_a))
        dist_b = float(np.linalg.norm(color - self._ref_b))
        return "A" if dist_a <= dist_b else "B"


# ---------------------------------------------------------------------------
# Step 4 — Full pipeline: detection → team classification
# ---------------------------------------------------------------------------

def classify_teams(
    image_bgr: np.ndarray,
    detections: DetectionResult,
    min_players: int = 2,
) -> DetectionResult:
    """
    Full pipeline: run team classification on a DetectionResult and return an
    enriched DetectionResult where each entry has a ``team`` field.

    Non-player roles (referee, ball) pass through with team=None.

    Args:
        image_bgr:   Full frame in BGR format.
        detections:  Output of detect.detect_objects().
        min_players: Minimum player crops required to attempt clustering.
                     Below this threshold all players get team=None.

    Returns:
        DetectionResult with an additional ``team`` key per detection:
          - "team_a" | "team_b" for classified players/goalkeepers
          - None for referees, ball, or when clustering cannot run
    """
    dets = detections["detections"]

    # Step 1: extract crops and remember which detection index each belongs to
    crops, det_indices = extract_player_crops(image_bgr, detections)

    # Step 3: cluster only if we have enough players
    team_map: dict[int, str] = {}  # detection index → "team_a" | "team_b"

    if len(crops) >= min_players:
        assignments = cluster_teams(crops)
        for assignment in assignments:
            pid = assignment["player_id"]
            det_idx = det_indices[pid]
            team_letter = assignment["team"]  # "A" or "B"
            team_map[det_idx] = f"team_{team_letter.lower()}"

    # Build enriched output
    result: list[BBox] = []
    for i, det in enumerate(dets):
        result.append(
            BBox(
                label=det["label"],
                confidence=det["confidence"],
                bbox=det["bbox"],
                team=team_map.get(i),  # None for non-players
            )
        )

    return DetectionResult(detections=result)


# ---------------------------------------------------------------------------
# Visualisation helper
# ---------------------------------------------------------------------------

def draw_teams(
    image_bgr: np.ndarray,
    detections: DetectionResult,
    colors: dict[str, tuple[int, int, int]] | None = None,
    text_color: tuple[int, int, int] = (255, 255, 255),
    thickness: int = 2,
    font_scale: float = 0.55,
) -> np.ndarray:
    """
    Draw bounding boxes colour-coded by team assignment.

    Default box colours:
      team_a     → cyan   (255, 200,   0)
      team_b     → orange (  0, 100, 255)
      goalkeeper → yellow (  0, 255, 255)
      referee    → purple (180,   0, 180)
      ball       → white  (255, 255, 255)

    Args:
        image_bgr:  Source frame (a copy is made — original not modified).
        detections: Output of classify_teams().
        colors:     Optional dict to override any default color by key.
        text_color: BGR color for all label text.
        thickness:  Box line thickness in pixels.
        font_scale: OpenCV FONT_HERSHEY_SIMPLEX scale factor.

    Returns:
        Annotated BGR frame as a new numpy array.
    """
    _defaults: dict[str, tuple[int, int, int]] = {
        "team_a":     (255, 200,   0),
        "team_b":     (  0, 100, 255),
        "goalkeeper": (  0, 255, 255),
        "referee":    (180,   0, 180),
        "ball":       (255, 255, 255),
        "player":     (160, 160, 160),  # unclassified fallback
    }
    if colors:
        _defaults.update(colors)

    out = image_bgr.copy()

    for det in detections["detections"]:
        x1, y1, x2, y2 = [int(v) for v in det["bbox"]]
        label = det["label"]
        conf  = det["confidence"]
        team  = det.get("team")

        color_key = team if team else label
        box_color = _defaults.get(color_key, (160, 160, 160))

        cv2.rectangle(out, (x1, y1), (x2, y2), box_color, thickness)

        caption = f"{team or label} {conf:.2f}"
        (text_w, text_h), baseline = cv2.getTextSize(
            caption, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness
        )
        label_y = max(y1 - 6, text_h + baseline)
        cv2.rectangle(
            out,
            (x1, label_y - text_h - baseline),
            (x1 + text_w, label_y + baseline),
            box_color,
            cv2.FILLED,
        )
        cv2.putText(
            out, caption, (x1, label_y),
            cv2.FONT_HERSHEY_SIMPLEX, font_scale,
            text_color, thickness, cv2.LINE_AA,
        )

    return out


# ---------------------------------------------------------------------------
# Test block
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import json
    import sys
    from pathlib import Path
    from detect import detect_objects

    if len(sys.argv) < 2:
        print("Usage: python team_classifier.py <path_to_image>")
        sys.exit(0)

    img_path = sys.argv[1]
    frame = cv2.imread(img_path)
    if frame is None:
        print(f"Could not read image: {img_path}")
        sys.exit(1)

    # --- Step 1: YOLO detection ---
    raw = detect_objects(img_path)
    print(f"YOLO detections : {len(raw['detections'])}")

    # --- Step 2: Extract crops ---
    crops, indices = extract_player_crops(frame, raw)
    print(f"Player crops    : {len(crops)}")

    # --- Step 3: Dominant colors ---
    print("Dominant colors (RGB):")
    for i, crop in enumerate(crops):
        color = get_dominant_color(crop)
        print(f"  player {i:2d}: R={color[0]:.0f}  G={color[1]:.0f}  B={color[2]:.0f}")

    # --- Step 4: Cluster into teams ---
    assignments = cluster_teams(crops)
    team_a = [a for a in assignments if a["team"] == "A"]
    team_b = [a for a in assignments if a["team"] == "B"]
    print(f"\nTeam A: {len(team_a)} players")
    print(f"Team B: {len(team_b)} players")
    print(json.dumps(assignments, indent=2))

    # --- Full pipeline + annotated image ---
    classified = classify_teams(frame, raw)
    annotated  = draw_teams(frame, classified)
    out_path   = Path(img_path).stem + "_teams.jpg"
    cv2.imwrite(out_path, annotated)
    print(f"\nSaved annotated image: {out_path}")
