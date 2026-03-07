# =============================================================================
# tracker.py — Multi-object tracking for football players across video frames
#
# Public API:
#   PlayerTracker              — stateful tracker class (one instance per video)
#   track_players(detections)  — convenience function using the module singleton
#
# How it works:
#   DeepSort combines two signals to maintain consistent IDs frame-to-frame:
#     1. Appearance features  — a small CNN re-identifies players by how they look.
#     2. Kalman filter motion — predicts where each player will be next frame.
#   Together these handle occlusions and momentary mis-detections far better
#   than pure IoU matching (e.g. SORT).
#
#   Only "player" and "goalkeeper" labels are tracked.
#   "referee" and "ball" are filtered out before the tracker sees them.
#
# Usage (per-video, stateful):
#   from tracker import PlayerTracker
#   t = PlayerTracker()
#   for frame_detections in video_frames:
#       tracked = t.update(frame_detections)
#
# Usage (stateless single-frame helper — shares a module-level instance):
#   from tracker import track_players
#   tracked = track_players(detections)
# =============================================================================

from __future__ import annotations

from typing import TypedDict

from deep_sort_realtime.deepsort_tracker import DeepSort


# ---------------------------------------------------------------------------
# Types
# ---------------------------------------------------------------------------

class TrackedPlayer(TypedDict):
    id: int              # Unique track ID, stable across frames
    bbox: list[float]    # [x1, y1, x2, y2] absolute pixel coordinates
    label: str           # "player" or "goalkeeper"


# Labels that should be tracked (team players only — not ball or referee)
_TRACKED_LABELS = {"player", "goalkeeper"}


# ---------------------------------------------------------------------------
# PlayerTracker — stateful, one instance per video/stream
# ---------------------------------------------------------------------------

class PlayerTracker:
    """
    Wraps DeepSort and converts between the YOLO DetectionResult format and
    the format expected by the tracker.

    Each instance holds its own DeepSort state — create a new PlayerTracker
    for every new video clip or live session so track IDs restart from 1.

    Args:
        max_age:        Number of consecutive missed frames before a track is
                        deleted. Higher values keep IDs alive through longer
                        occlusions but may cause ID drift.
        n_init:         Frames a detection must be seen before it becomes a
                        confirmed track. Reduces spurious one-frame detections.
        max_cosine_dist: Appearance similarity threshold (0–1). Lower = stricter
                        re-identification matching.
    """

    def __init__(
        self,
        max_age: int = 30,
        n_init: int = 3,
        max_cosine_dist: float = 0.4,
    ) -> None:
        # DeepSort tracker — maintains Kalman state and appearance embeddings
        # for each active track across frames.
        # embedder_gpu=False: run appearance CNN on CPU (no CUDA on this machine).
        self._tracker = DeepSort(
            max_age=max_age,
            n_init=n_init,
            max_cosine_distance=max_cosine_dist,
            embedder_gpu=False,
        )

    def update(self, detections: dict, frame=None) -> list[TrackedPlayer]:
        """
        Feed one frame's detections into the tracker and return confirmed tracks.

        Args:
            detections: DetectionResult from detect.detect_objects().
                        Shape: {"detections": [{"label", "confidence", "bbox"}, ...]}
            frame:      Optional BGR numpy array of the current frame.
                        When provided DeepSort extracts appearance features
                        for more robust re-identification; if None the tracker
                        falls back to motion-only matching.

        Returns:
            List of TrackedPlayer dicts for every currently confirmed track:
              - id:    Integer track ID, stable as long as the player is visible.
              - bbox:  [x1, y1, x2, y2] in the same pixel space as the input.
              - label: "player" or "goalkeeper".
        """
        # --- Step 1: Filter to trackable roles ---
        # Referee and ball are ignored — they don't need persistent identity.
        player_dets = [
            d for d in detections["detections"]
            if d["label"] in _TRACKED_LABELS
        ]

        # --- Step 2: Convert to DeepSort input format ---
        # DeepSort expects a list of:  ([x, y, w, h], confidence, class_name)
        # YOLO gives us:               [x1, y1, x2, y2]
        # So we convert xyxy → xywh.
        detection_list = []
        for det in player_dets:
            x1, y1, x2, y2 = det["bbox"]
            w = x2 - x1
            h = y2 - y1
            detection_list.append(
                ([x1, y1, w, h], det["confidence"], det["label"])
            )

        # --- Step 3: Run DeepSort update ---
        # The tracker returns one Track object per active track (confirmed +
        # tentative). We only expose confirmed tracks (is_confirmed() == True).
        raw_tracks = self._tracker.update_tracks(detection_list, frame=frame)

        # --- Step 4: Build output ---
        tracked: list[TrackedPlayer] = []
        for track in raw_tracks:
            # Skip tentative tracks — they haven't appeared in n_init frames yet
            if not track.is_confirmed():
                continue

            # Convert the tracker's internal ltrb (left-top-right-bottom) bbox
            # back to our standard [x1, y1, x2, y2] float format.
            x1, y1, x2, y2 = track.to_ltrb()

            tracked.append(
                TrackedPlayer(
                    id=int(track.track_id),
                    bbox=[float(x1), float(y1), float(x2), float(y2)],
                    label=track.det_class or "player",
                )
            )

        return tracked


# ---------------------------------------------------------------------------
# Module-level singleton — for stateless / single-frame usage
# ---------------------------------------------------------------------------

# One shared tracker instance for callers that don't manage their own state.
# Suitable for the live webcam endpoint where frames arrive sequentially.
# For video file processing, instantiate PlayerTracker() directly per video.
_default_tracker = PlayerTracker()


def track_players(detections: dict, frame=None) -> list[TrackedPlayer]:
    """
    Convenience wrapper around the module-level PlayerTracker singleton.

    Suitable for sequential frame streams (live webcam, video processing).
    For independent videos, instantiate PlayerTracker() directly to avoid
    state bleed between clips.

    Args:
        detections: DetectionResult from detect.detect_objects().
        frame:      Optional BGR numpy array — enables appearance features.

    Returns:
        List of TrackedPlayer dicts with stable IDs.
    """
    return _default_tracker.update(detections, frame=frame)


# ---------------------------------------------------------------------------
# Test block
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import json
    import sys
    import cv2
    from pathlib import Path
    from detect import detect_objects

    if len(sys.argv) < 2:
        print("Usage: python tracker.py <image_or_video_path>")
        sys.exit(0)

    path = Path(sys.argv[1])

    if path.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}:
        # ── Single-image test ────────────────────────────────────────────────
        # Run the same image through the tracker N times to let tracks confirm
        # (DeepSort requires n_init=3 frames before a track is confirmed).
        print(f"Image mode — simulating 5 consecutive frames of: {path}")
        t = PlayerTracker(n_init=1)  # lower n_init so tracks confirm instantly
        frame = cv2.imread(str(path))
        dets  = detect_objects(str(path))

        for i in range(5):
            tracked = t.update(dets, frame=frame)
            print(f"\n  Frame {i+1}: {len(tracked)} confirmed tracks")
            for tp in tracked:
                bbox_str = ", ".join(f"{v:.0f}" for v in tp["bbox"])
                print(f"    ID {tp['id']:3d}  {tp['label']:<12}  [{bbox_str}]")

    else:
        # ── Video test ───────────────────────────────────────────────────────
        print(f"Video mode: {path}")
        cap = cv2.VideoCapture(str(path))
        if not cap.isOpened():
            print("Could not open video.")
            sys.exit(1)

        t = PlayerTracker()
        frame_idx = 0
        STEP = 5  # process every 5th frame for speed

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if frame_idx % STEP == 0:
                # Write frame to temp file for detect_objects
                import tempfile
                tmp = tempfile.NamedTemporaryFile(suffix=".jpg", delete=False)
                cv2.imwrite(tmp.name, frame)
                tmp.close()

                dets    = detect_objects(tmp.name)
                tracked = t.update(dets, frame=frame)
                Path(tmp.name).unlink(missing_ok=True)

                ids = [tp["id"] for tp in tracked]
                print(f"  Frame {frame_idx:4d}: {len(tracked)} tracks  IDs={ids}")

            frame_idx += 1

        cap.release()
        print("\nDone.")
