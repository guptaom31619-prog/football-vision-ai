# =============================================================================
# api.py — FastAPI backend for Football Vision AI
#
# Endpoints:
#   GET  /                        — Health check
#   POST /detect/image            — Detect + team classify + annotated image
#   POST /detect/video            — Detect + track + stats + annotated video + heatmaps
#   GET  /heatmaps/{filename}     — Serve a generated heatmap PNG
#   GET  /outputs/{filename}      — Serve annotated output (video/image/snapshot)
#
# Run with:
#   uvicorn api:app --port 8000 --timeout-keep-alive 600
# =============================================================================

from __future__ import annotations

import json
import logging
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Generator

import cv2
import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse

from detect import (
    detect_objects, detect_and_track, draw_detections_on_frame,
    model as yolo_model,
)
from tracker import PlayerTracker
from team_classifier import TeamClassifier
from stats_engine import MatchStats
from heatmap import HeatmapGenerator

logger = logging.getLogger("uvicorn.error")

_HEATMAP_DIR = Path(__file__).parent / "heatmaps"
_HEATMAP_DIR.mkdir(parents=True, exist_ok=True)

_OUTPUT_DIR = Path(__file__).parent / "outputs"
_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

app = FastAPI(
    title="Football Vision AI",
    version="1.0.0",
    description="YOLOv8-powered football player detection API",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

VIDEO_FRAME_STEP = 10

ALLOWED_IMAGE_TYPES = {"image/jpeg", "image/png", "image/webp"}
ALLOWED_VIDEO_TYPES = {"video/mp4", "video/avi", "video/quicktime", "video/x-matroska"}

_CLEANUP_MAX_AGE_HOURS = 24


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _save_upload_to_temp(upload: UploadFile, suffix: str) -> Path:
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    tmp.write(upload.file.read())
    tmp.flush()
    tmp.close()
    return Path(tmp.name)


def _cleanup_old_outputs() -> None:
    """Delete output/heatmap files older than _CLEANUP_MAX_AGE_HOURS."""
    cutoff = time.time() - _CLEANUP_MAX_AGE_HOURS * 3600
    for directory in (_OUTPUT_DIR, _HEATMAP_DIR):
        for f in directory.iterdir():
            if f.is_file() and f.stat().st_mtime < cutoff:
                f.unlink(missing_ok=True)


def _reencode_to_h264(src: Path) -> Path:
    """Re-encode video to H.264 via imageio-ffmpeg for browser compatibility."""
    try:
        import imageio.v3 as iio
        from imageio.plugins.ffmpeg import get_exe
        ffmpeg_exe = get_exe()
    except Exception:
        try:
            from imageio_ffmpeg import get_ffmpeg_exe
            ffmpeg_exe = get_ffmpeg_exe()
        except Exception:
            return src

    dst = src.with_name(src.stem + "_h264.mp4")
    try:
        subprocess.run(
            [
                ffmpeg_exe, "-y", "-i", str(src),
                "-c:v", "libx264", "-preset", "fast",
                "-crf", "23", "-pix_fmt", "yuv420p",
                "-movflags", "+faststart", "-an",
                str(dst),
            ],
            capture_output=True, timeout=600,
        )
        if dst.exists() and dst.stat().st_size > 0:
            src.unlink(missing_ok=True)
            return dst
    except Exception as exc:
        logger.warning("H.264 re-encode failed: %s", exc)

    return src


# ---------------------------------------------------------------------------
# Startup
# ---------------------------------------------------------------------------

@app.on_event("startup")
async def _startup_cleanup() -> None:
    _cleanup_old_outputs()
    logger.info("Cleaned up outputs older than %dh", _CLEANUP_MAX_AGE_HOURS)


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get("/")
async def health_check() -> JSONResponse:
    return JSONResponse({"message": "Football Vision AI API Running"})


@app.post("/detect/image")
async def detect_image(file: UploadFile = File(...)) -> JSONResponse:
    if file.content_type not in ALLOWED_IMAGE_TYPES:
        raise HTTPException(
            status_code=415,
            detail=f"Unsupported file type '{file.content_type}'. "
                   f"Accepted: {', '.join(ALLOWED_IMAGE_TYPES)}",
        )

    suffix = Path(file.filename or "upload.jpg").suffix or ".jpg"
    tmp_path = _save_upload_to_temp(file, suffix)

    try:
        result = detect_objects(tmp_path, yolo=yolo_model)

        frame = cv2.imread(str(tmp_path))
        if frame is not None:
            annotated = draw_detections_on_frame(frame, result)
            out_name = f"annotated_{int(time.time())}.jpg"
            out_path = _OUTPUT_DIR / out_name
            cv2.imwrite(str(out_path), annotated)
        else:
            out_name = None
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    finally:
        tmp_path.unlink(missing_ok=True)

    frame_stats = MatchStats()
    frame_stats.update(result)

    return JSONResponse({
        "detections": result["detections"],
        "stats": frame_stats.get_stats(),
        "annotated_image": out_name,
    })


@app.get("/heatmaps/{filename}")
async def serve_heatmap(filename: str) -> FileResponse:
    path = _HEATMAP_DIR / filename
    if not path.resolve().is_relative_to(_HEATMAP_DIR.resolve()):
        raise HTTPException(status_code=400, detail="Invalid filename.")
    if not path.exists():
        raise HTTPException(status_code=404, detail=f"Heatmap '{filename}' not found.")
    return FileResponse(path, media_type="image/png")


@app.get("/outputs/{filename}")
async def serve_output(filename: str) -> FileResponse:
    """Serve an annotated output file (video or image)."""
    path = _OUTPUT_DIR / filename
    if not path.resolve().is_relative_to(_OUTPUT_DIR.resolve()):
        raise HTTPException(status_code=400, detail="Invalid filename.")
    if not path.exists():
        raise HTTPException(status_code=404, detail=f"Output '{filename}' not found.")
    media = "video/mp4" if path.suffix == ".mp4" else f"image/{path.suffix.lstrip('.')}"
    return FileResponse(path, media_type=media)


@app.post("/detect/video")
async def detect_video(file: UploadFile = File(...)) -> StreamingResponse:
    """
    Process video: detect, track, classify teams, compute stats, generate
    heatmaps, and render a fully annotated output video with bounding boxes.

    Streams NDJSON progress lines so the frontend shows a live progress bar.
    Final line contains the result payload.
    """
    if file.content_type not in ALLOWED_VIDEO_TYPES:
        raise HTTPException(
            status_code=415,
            detail=f"Unsupported file type '{file.content_type}'. "
                   f"Accepted: {', '.join(ALLOWED_VIDEO_TYPES)}",
        )

    suffix = Path(file.filename or "upload.mp4").suffix or ".mp4"
    tmp_path = _save_upload_to_temp(file, suffix)

    def _process() -> Generator[str, None, None]:
        frame_results: list[dict] = []

        try:
            cap = cv2.VideoCapture(str(tmp_path))
            if not cap.isOpened():
                yield json.dumps({"type": "error", "detail": "Could not open video."}) + "\n"
                return

            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps          = cap.get(cv2.CAP_PROP_FPS) or 30.0
            width        = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height       = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

            # --- Set up annotated output video writer ---
            ts = int(time.time())
            raw_name = f"annotated_{ts}_raw.mp4"
            raw_path = _OUTPUT_DIR / raw_name
            for codec in ["avc1", "H264", "mp4v"]:
                fourcc = cv2.VideoWriter_fourcc(*codec)
                writer = cv2.VideoWriter(str(raw_path), fourcc, fps, (width, height))
                if writer.isOpened():
                    break

            frame_idx     = 0
            last_frame    = None
            sampled_count = 0
            snapshot_frame = None

            video_tracker  = PlayerTracker()
            video_team_clf = TeamClassifier()
            video_stats    = MatchStats()
            video_stats.set_frame_size(width, height)
            video_heatmap  = HeatmapGenerator()

            current_detection = None

            logger.info(
                "Processing video: %d frames (%.1f fps, %dx%d)",
                total_frames, fps, width, height,
            )

            yield json.dumps({
                "type": "progress", "percent": 0,
                "frame": 0, "total": total_frames, "sampled": 0,
            }) + "\n"

            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                last_frame = frame

                if frame_idx % VIDEO_FRAME_STEP == 0:
                    current_detection = detect_and_track(frame, video_tracker, team_clf=video_team_clf, yolo=yolo_model)
                    video_stats.update(current_detection)
                    video_heatmap.update(current_detection)
                    sampled_count += 1

                    frame_results.append({
                        "frame_index": frame_idx,
                        "timestamp_sec": round(frame_idx / fps, 3),
                        "detections": current_detection["detections"],
                    })

                    # Pick a good snapshot — frame near 30% with most detections
                    if snapshot_frame is None and frame_idx > total_frames * 0.3:
                        snapshot_frame = frame.copy()

                    pct = min(99, int((frame_idx / max(1, total_frames)) * 100))
                    logger.info(
                        "  [%3d%%]  frame %d / %d  (%d sampled)",
                        pct, frame_idx, total_frames, sampled_count,
                    )
                    yield json.dumps({
                        "type": "progress", "percent": pct,
                        "frame": frame_idx, "total": total_frames,
                        "sampled": sampled_count,
                    }) + "\n"

                if current_detection is not None:
                    annotated = draw_detections_on_frame(frame, current_detection)
                    writer.write(annotated)
                else:
                    writer.write(frame)

                frame_idx += 1

            cap.release()
            writer.release()

            # --- Re-encode to H.264 for browser compatibility ---
            final_video_path = _reencode_to_h264(raw_path)
            final_video_name = final_video_path.name

            # --- Save annotated snapshot ---
            snapshot_name = None
            if snapshot_frame is not None and current_detection is not None:
                snap_annotated = draw_detections_on_frame(snapshot_frame, current_detection)
                snapshot_name = f"snapshot_{ts}.jpg"
                cv2.imwrite(str(_OUTPUT_DIR / snapshot_name), snap_annotated)

            # --- Generate team-level heatmaps ---
            heatmap_paths: list[str] = []
            if last_frame is not None and video_heatmap.player_positions:
                for old in _HEATMAP_DIR.glob("*.png"):
                    old.unlink(missing_ok=True)
                frame_shape = (last_frame.shape[0], last_frame.shape[1])
                saved = video_heatmap.generate_all(frame_shape)
                heatmap_paths = [p.name for p in saved]

            logger.info(
                "Done: %d sampled, %d heatmaps, output → %s",
                sampled_count, len(heatmap_paths), final_video_name,
            )

            yield json.dumps({
                "type":            "result",
                "total_frames":    total_frames,
                "frames_sampled":  len(frame_results),
                "frame_step":      VIDEO_FRAME_STEP,
                "results":         frame_results,
                "stats":           video_stats.get_stats(),
                "heatmaps":        heatmap_paths,
                "annotated_video": final_video_name,
                "snapshot":        snapshot_name,
            }) + "\n"

        finally:
            tmp_path.unlink(missing_ok=True)

    return StreamingResponse(_process(), media_type="application/x-ndjson")
