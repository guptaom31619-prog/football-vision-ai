# =============================================================================
# detect.py — YOLOv8 inference pipeline for football player detection
#
# Public API:
#   detect_objects(image_path, ...)   -> DetectionResult   (single image)
#   detect_and_track(frame, tracker)  -> DetectionResult   (video frame, with IDs)
#   draw_detections(image_path, ...)  -> np.ndarray
#
# Pipeline per frame:
#   1. YOLO inference         → raw bboxes + class labels
#   2. Team classification    → "A" or "B" added per player/goalkeeper
#   3. Player tracking        → stable integer "id" added per player/goalkeeper
#      (tracking is opt-in via detect_and_track; detect_objects skips it)
#
# Notes:
#   - detect_objects is stateless — safe to call in parallel.
#   - detect_and_track is stateful — caller must supply a PlayerTracker instance
#     that persists across frames. Create one PlayerTracker per video/session.
#   - Ball and referee are never assigned team or id fields.
# =============================================================================

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import TypedDict

import cv2
import numpy as np
from ultralytics import YOLO

# Team classifier — KMeans jersey-color clustering → "A" / "B" per player
from team_classifier import extract_player_crops, cluster_teams, TeamClassifier

# Player tracker — DeepSort, assigns stable integer IDs across video frames
from tracker import PlayerTracker

# ---------------------------------------------------------------------------
# Model — loaded once at module import, reused across all calls
# ---------------------------------------------------------------------------

# Custom-trained football detection model.
# Trained on 4 classes: player (0), ball (1), goalkeeper (2), referee (3).
# Falls back to the lightweight pretrained checkpoint if best.pt is missing
# (e.g. development environment before training has been run).
_TRAINED_WEIGHTS = Path("runs/detect/runs/detect/football_model/weights/best.pt")
_FALLBACK_WEIGHTS = "yolov8n.pt"
WEIGHTS = str(_TRAINED_WEIGHTS) if _TRAINED_WEIGHTS.exists() else _FALLBACK_WEIGHTS

if not _TRAINED_WEIGHTS.exists():
    import warnings
    warnings.warn(
        f"Trained weights not found at {_TRAINED_WEIGHTS}. "
        "Falling back to yolov8n.pt (COCO classes). Run train.py to generate best.pt.",
        stacklevel=1,
    )

model: YOLO = YOLO(WEIGHTS)


# ---------------------------------------------------------------------------
# Types
# ---------------------------------------------------------------------------

class BBox(TypedDict):
    label: str
    id: int | None     # Stable track ID (players/goalkeepers only; None otherwise)
    team: str | None   # "A" | "B" for players/goalkeepers; None for referee/ball
    confidence: float
    bbox: list[float]  # [x1, y1, x2, y2] in absolute pixel coordinates


class DetectionResult(TypedDict):
    detections: list[BBox]


# ---------------------------------------------------------------------------
# Football constraints — enforce real match limits per frame
# ---------------------------------------------------------------------------

_MAX_PLAYERS_PER_TEAM = 11
_MAX_REFEREES = 4
_MAX_BALLS = 1


def _enforce_football_limits(detections: list[BBox]) -> list[BBox]:
    """
    Keep only the top-N detections per role sorted by confidence:
      - 11 players per team (A and B separately)
      - 4 referees
      - 1 ball
    Drops excess low-confidence duplicates caused by overlapping boxes.
    """
    team_a = sorted(
        [d for d in detections if d["label"] in ("player", "goalkeeper") and d.get("team") == "A"],
        key=lambda d: -d["confidence"],
    )[:_MAX_PLAYERS_PER_TEAM]

    team_b = sorted(
        [d for d in detections if d["label"] in ("player", "goalkeeper") and d.get("team") == "B"],
        key=lambda d: -d["confidence"],
    )[:_MAX_PLAYERS_PER_TEAM]

    unclassified = sorted(
        [d for d in detections if d["label"] in ("player", "goalkeeper") and d.get("team") is None],
        key=lambda d: -d["confidence"],
    )
    # Fill remaining slots if teams are under 11
    remaining_a = _MAX_PLAYERS_PER_TEAM - len(team_a)
    remaining_b = _MAX_PLAYERS_PER_TEAM - len(team_b)
    extra = unclassified[:remaining_a + remaining_b]

    referees = sorted(
        [d for d in detections if d["label"] == "referee"],
        key=lambda d: -d["confidence"],
    )[:_MAX_REFEREES]

    balls = sorted(
        [d for d in detections if d["label"] == "ball"],
        key=lambda d: -d["confidence"],
    )[:_MAX_BALLS]

    return team_a + team_b + extra + referees + balls


# ---------------------------------------------------------------------------
# Core functions
# ---------------------------------------------------------------------------

def detect_objects(
    image_path: str | Path,
    yolo: YOLO = model,
    conf_threshold: float = 0.5,
) -> DetectionResult:
    """
    Run YOLOv8 inference on a single image, then classify players into teams
    via jersey-color clustering.

    Args:
        image_path:      Path to the input image (JPEG / PNG).
        yolo:            YOLO model instance to use for inference.
        conf_threshold:  Minimum confidence score to include a detection.

    Returns:
        DetectionResult dict with a "detections" list, each entry containing:
          - label:      Class name ("player", "ball", "goalkeeper", "referee")
          - id:         None (tracking not run — use detect_and_track for video)
          - team:       "A" or "B" for players/goalkeepers; None for referee/ball
          - confidence: Float in [0, 1], rounded to 4 decimal places
          - bbox:       [x1, y1, x2, y2] absolute pixel coordinates as floats

    Raises:
        FileNotFoundError: If the image path does not exist.
        ValueError:        If OpenCV cannot decode the file as an image.
    """
    path = Path(image_path)
    if not path.exists():
        raise FileNotFoundError(f"Image not found: {path}")

    # Load image with OpenCV — needed for both YOLO and team crop extraction
    frame = cv2.imread(str(path))
    if frame is None:
        raise ValueError(f"Could not decode image: {path}")

    # --- Step 1: YOLO inference ---
    # Returns one Results object per image in the batch (batch size = 1 here)
    results = yolo.predict(str(path), conf=conf_threshold, iou=0.4, verbose=False)

    detections: list[BBox] = []
    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            detections.append(
                BBox(
                    label=yolo.names[int(box.cls)],
                    id=None,    # filled in by tracker when using detect_and_track
                    team=None,  # filled in by team classifier below
                    confidence=round(float(box.conf), 4),
                    bbox=[x1, y1, x2, y2],
                )
            )

    raw_result = DetectionResult(detections=detections)

    # --- Step 2: Team classification ---
    # Extract upper-torso crops for every player and goalkeeper detection.
    # Referee and ball are skipped inside extract_player_crops automatically.
    player_crops, player_indices = extract_player_crops(frame, raw_result)

    if len(player_crops) >= 2:
        assignments = cluster_teams(player_crops)
        for assignment in assignments:
            det_idx = player_indices[assignment["player_id"]]
            detections[det_idx]["team"] = assignment["team"]

    # --- Step 3: Enforce football limits (11+11+4+1) ---
    raw_result["detections"] = _enforce_football_limits(detections)

    return raw_result


def detect_and_track(
    frame: np.ndarray,
    player_tracker: PlayerTracker,
    team_clf: TeamClassifier | None = None,
    yolo: YOLO = model,
    conf_threshold: float = 0.5,
) -> DetectionResult:
    """
    Run detection + team classification + player tracking on a single BGR frame.

    Use this function for video processing and live streams where stable
    player IDs are needed across frames.  For single images, use detect_objects.

    Pipeline:
      1. YOLO inference on the in-memory frame (no disk I/O).
      2. Team classification via jersey-color KMeans.
      3. DeepSort tracking — assigns each player/goalkeeper a stable integer id.
      4. Track IDs are matched back to detections by IoU overlap between the
         tracker's refined bbox and the original YOLO bbox.

    Args:
        frame:          BGR numpy array (H × W × 3, uint8) for the current frame.
        player_tracker: A PlayerTracker instance that persists across frames.
                        Create one per video/session and reuse it every frame.
        yolo:           YOLO model instance (defaults to module-level best.pt).
        conf_threshold: Minimum YOLO confidence to include a detection.

    Returns:
        DetectionResult where player and goalkeeper entries include:
          - id:   Stable integer track ID (consistent across frames in the session)
          - team: "A" or "B"
        Ball and referee entries have id=None and team=None.
    """
    # --- Step 1: YOLO on in-memory frame (no temp file) ---
    results = yolo.predict(frame, conf=conf_threshold, iou=0.4, verbose=False)

    detections: list[BBox] = []
    for r in results:
        for box in r.boxes:
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            detections.append(
                BBox(
                    label=yolo.names[int(box.cls)],
                    id=None,
                    team=None,
                    confidence=round(float(box.conf), 4),
                    bbox=[x1, y1, x2, y2],
                )
            )

    raw_result = DetectionResult(detections=detections)

    # --- Step 2: Team classification ---
    if team_clf is not None:
        assignments, player_indices = team_clf.classify_frame(frame, raw_result)
    else:
        player_crops, player_indices = extract_player_crops(frame, raw_result)
        assignments = cluster_teams(player_crops) if len(player_crops) >= 2 else []

    for assignment in assignments:
        det_idx = player_indices[assignment["player_id"]]
        detections[det_idx]["team"] = assignment["team"]

    # Fallback: assign any remaining unclassified players via reference centroids
    if team_clf is not None:
        for det in detections:
            if det["label"] in ("player", "goalkeeper") and det.get("team") is None:
                det["team"] = team_clf.assign_by_color(frame, det["bbox"])

    # --- Step 3: Enforce football limits (11+11+4+1) ---
    detections = _enforce_football_limits(detections)
    raw_result["detections"] = detections

    # --- Step 4: Tracking ---
    tracked = player_tracker.update(raw_result, frame=frame)

    # --- Step 5: Match track IDs back to YOLO detections via IoU ---
    # DeepSort may refine bbox positions slightly (Kalman smoothing), so we
    # find the best-matching original detection for each confirmed track.
    for track in tracked:
        tx1, ty1, tx2, ty2 = track["bbox"]

        best_iou   = 0.0
        best_idx   = -1

        for i, det in enumerate(detections):
            if det["label"] not in ("player", "goalkeeper"):
                continue
            dx1, dy1, dx2, dy2 = det["bbox"]

            # IoU between track bbox and detection bbox
            ix1 = max(tx1, dx1)
            iy1 = max(ty1, dy1)
            ix2 = min(tx2, dx2)
            iy2 = min(ty2, dy2)
            inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
            if inter == 0:
                continue
            union = (
                (tx2 - tx1) * (ty2 - ty1)
                + (dx2 - dx1) * (dy2 - dy1)
                - inter
            )
            iou = inter / union if union > 0 else 0.0

            if iou > best_iou:
                best_iou = iou
                best_idx = i

        # Assign the track ID to the best-matching detection (IoU > 0.3 guard)
        if best_idx >= 0 and best_iou > 0.3:
            detections[best_idx]["id"] = track["id"]

    return raw_result


def draw_detections(
    image_path: str | Path,
    detections: DetectionResult,
    text_color: tuple[int, int, int] = (255, 255, 255),
    thickness: int = 2,
    font_scale: float = 0.6,
) -> np.ndarray:
    """
    Draw bounding boxes colour-coded by role and team assignment.

    Label format:
      - Tracked player with team:   "Player #3 (Team A)"
      - Tracked player, no team:    "Player #3 0.91"
      - Untracked player with team: "Player (Team A) 0.91"
      - Referee / ball:             "Referee 0.88"

    Box colours by role/team:
      Team A players/goalkeepers → cyan   (255, 200,   0)
      Team B players/goalkeepers → orange (  0, 120, 255)
      Referee                    → purple (180,   0, 180)
      Ball                       → white  (255, 255, 255)
      Unclassified player        → grey   (160, 160, 160)

    Args:
        image_path:  Path to the original image.
        detections:  Output of detect_objects() — includes team field.
        text_color:  BGR colour for all label text.
        thickness:   Line thickness for boxes (pixels).
        font_scale:  OpenCV font scale for label text.

    Returns:
        Annotated image as a BGR numpy array (H × W × 3, uint8).
        The original file on disk is NOT modified.

    Raises:
        FileNotFoundError: If the image path does not exist.
        ValueError:        If OpenCV cannot decode the file.
    """
    # Colour palette — keyed by team label or role name
    _colors: dict[str, tuple[int, int, int]] = {
        "A":        (255, 200,   0),   # Team A → cyan
        "B":        (  0, 120, 255),   # Team B → orange
        "referee":  (180,   0, 180),   # referee → purple
        "ball":     (255, 255, 255),   # ball → white
        "player":   (160, 160, 160),   # unclassified fallback → grey
        "goalkeeper": (160, 160, 160),
    }

    path = Path(image_path)
    if not path.exists():
        raise FileNotFoundError(f"Image not found: {path}")

    image = cv2.imread(str(path))
    if image is None:
        raise ValueError(f"Could not decode image: {path}")

    for det in detections["detections"]:
        x1, y1, x2, y2 = [int(v) for v in det["bbox"]]
        label = det["label"]
        team  = det.get("team")   # "A", "B", or None
        pid   = det.get("id")     # integer track ID or None
        conf  = det["confidence"]

        # Pick box colour: team assignment takes priority over role colour
        box_color = _colors.get(team or label, (160, 160, 160))

        # Build caption:
        #   Tracked player  → "Player #3 (Team A)"
        #   Untracked player with team → "player (Team A) 0.92"
        #   Referee / ball  → "referee 0.88"
        role = label.capitalize()
        if pid is not None and team:
            caption = f"{role} #{pid} (Team {team})"
        elif pid is not None:
            caption = f"{role} #{pid} {conf:.2f}"
        elif team:
            caption = f"{role} (Team {team}) {conf:.2f}"
        else:
            caption = f"{role} {conf:.2f}"

        # Bounding box rectangle
        cv2.rectangle(image, (x1, y1), (x2, y2), box_color, thickness)

        # Filled label background sized to caption width
        (text_w, text_h), baseline = cv2.getTextSize(
            caption, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness
        )
        label_y = max(y1 - 6, text_h + baseline)
        cv2.rectangle(
            image,
            (x1, label_y - text_h - baseline),
            (x1 + text_w, label_y + baseline),
            box_color,
            cv2.FILLED,
        )

        # Label text
        cv2.putText(
            image,
            caption,
            (x1, label_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            text_color,
            thickness,
            cv2.LINE_AA,
        )

    return image


def draw_detections_on_frame(
    frame: np.ndarray,
    detections: DetectionResult,
) -> np.ndarray:
    """
    Draw bounding boxes on a BGR frame with highly distinct colors:
      Team A  → bright cyan    (BGR 255, 255, 0)
      Team B  → bright orange  (BGR 0, 140, 255)
      Referee → bright yellow  (BGR 0, 255, 255)
      Ball    → magenta/pink   (BGR 255, 0, 255)
    """
    # BGR colors — maximally distinct on a green pitch
    _TEAM_A   = (255, 255,   0)   # cyan
    _TEAM_B   = (  0, 140, 255)   # orange
    _REFEREE  = (  0, 255, 255)   # yellow
    _BALL     = (255,   0, 255)   # magenta
    _DEFAULT  = (200, 200, 200)   # grey fallback

    h, w = frame.shape[:2]
    scale = max(0.3, min(0.45, w / 2500))
    box_thick = max(1, round(w / 640))
    text_thick = max(1, round(w / 800))

    image = frame.copy()

    for det in detections["detections"]:
        x1, y1, x2, y2 = [int(v) for v in det["bbox"]]
        label = det["label"]
        team  = det.get("team")

        # Pick color based on type
        if label == "ball":
            box_color = _BALL
            caption = "Ball"
        elif label == "referee":
            box_color = _REFEREE
            caption = "Referee"
        elif team == "A":
            box_color = _TEAM_A
            caption = "Team A"
        elif team == "B":
            box_color = _TEAM_B
            caption = "Team B"
        else:
            box_color = _DEFAULT
            caption = label.capitalize()

        # Draw box — slightly thicker for ball so it's visible
        t = box_thick + 1 if label == "ball" else box_thick
        cv2.rectangle(image, (x1, y1), (x2, y2), box_color, t)

        # Label background + text
        (tw, th), bl = cv2.getTextSize(
            caption, cv2.FONT_HERSHEY_SIMPLEX, scale, text_thick
        )
        ly = max(y1 - 4, th + bl + 2)
        cv2.rectangle(
            image, (x1, ly - th - bl - 2), (x1 + tw + 6, ly + 2),
            box_color, cv2.FILLED,
        )
        cv2.putText(
            image, caption, (x1 + 3, ly - 1),
            cv2.FONT_HERSHEY_SIMPLEX, scale,
            (0, 0, 0), text_thick, cv2.LINE_AA,
        )

    return image


# ---------------------------------------------------------------------------
# Test block
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # Usage: python detect.py [path/to/image.jpg]
    # Provide a local football image to test detection with the trained model.
    if len(sys.argv) < 2:
        print("Usage: python detect.py <path_to_image>")
        print(f"Model loaded: {WEIGHTS}")
        print(f"Classes     : {model.names}")
        sys.exit(0)

    test_image = sys.argv[1]
    print(f"Model  : {WEIGHTS}")
    print(f"Classes: {model.names}")
    print(f"Image  : {test_image}\n")

    # Run detection through the shared pipeline
    result = detect_objects(test_image)

    # Pretty-print structured output
    print("Detection results:")
    print(json.dumps(result, indent=2))
    print(f"\nTotal detections: {len(result['detections'])}")

    # Summarise by class
    from collections import Counter
    counts = Counter(d["label"] for d in result["detections"])
    print("\nBy class:")
    for cls, n in sorted(counts.items()):
        print(f"  {cls}: {n}")

    # Save annotated image next to the source
    annotated = draw_detections(test_image, result)
    output_path = Path(test_image).stem + "_annotated.jpg"
    cv2.imwrite(output_path, annotated)
    print(f"\nAnnotated image saved to: {output_path}")
