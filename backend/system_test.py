# =============================================================================
# system_test.py — End-to-end test for the Football Vision AI pipeline
#
# Validates:
#   1. Trained model file exists
#   2. Local detection (detect_objects) returns correct schema
#   3. FastAPI server starts and responds on :8000
#   4. POST /detect/image  → 200 + JSON with "detections"
#   5. POST /detect/video  → 200 + JSON with "results"
#
# Usage:
#   python system_test.py
# =============================================================================

from __future__ import annotations

import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import requests

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

BASE_URL = "http://127.0.0.1:8000"
MODEL_PATH = Path("runs/detect/runs/detect/football_model/weights/best.pt")
TEST_IMAGE = Path("test_data/test_image.jpg")
TEST_VIDEO = Path("test_data/test_video.mp4")

SERVER_STARTUP_WAIT = 8          # seconds to wait after launching uvicorn
SERVER_REQUEST_TIMEOUT = 120     # generous timeout for video processing

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

results: dict[str, str] = {}


def log(msg: str) -> None:
    print(f"[TEST] {msg}")


def record(name: str, passed: bool, detail: str = "") -> None:
    status = "PASS" if passed else "FAIL"
    results[name] = status
    suffix = f"  ({detail})" if detail else ""
    marker = "\033[92mPASS\033[0m" if passed else "\033[91mFAIL\033[0m"
    print(f"       → {marker}{suffix}")


def server_is_running() -> bool:
    try:
        r = requests.get(f"{BASE_URL}/", timeout=3)
        return r.status_code == 200
    except requests.ConnectionError:
        return False


# ---------------------------------------------------------------------------
# Test 1 — Model file exists
# ---------------------------------------------------------------------------

def test_model_file() -> None:
    log("Checking trained model file...")
    exists = MODEL_PATH.exists()
    record("Model File", exists, str(MODEL_PATH) if exists else "NOT FOUND")


# ---------------------------------------------------------------------------
# Test 2 — Local detection via detect_objects()
# ---------------------------------------------------------------------------

def test_local_detection() -> None:
    log("Running local image detection via detect_objects()...")

    if not TEST_IMAGE.exists():
        record("Local Detection", False, f"{TEST_IMAGE} not found")
        return

    try:
        from detect import detect_objects, model as yolo_model

        result = detect_objects(str(TEST_IMAGE), yolo=yolo_model)

        has_key = "detections" in result
        det_count = len(result.get("detections", []))

        log(f"  Detections returned: {det_count}")
        for d in result["detections"][:5]:
            label = d.get("label", "?")
            conf = d.get("confidence", 0)
            team = d.get("team")
            team_str = f" (Team {team})" if team else ""
            log(f"    - {label}{team_str}  {conf:.2f}")

        record("Local Detection", has_key, f"{det_count} objects detected")

    except Exception as exc:
        record("Local Detection", False, str(exc))


# ---------------------------------------------------------------------------
# Test 3 — FastAPI server startup
# ---------------------------------------------------------------------------

def start_server() -> subprocess.Popen | None:
    """Launch uvicorn in a subprocess and return the process handle."""
    if server_is_running():
        log("Server already running on :8000 — reusing it.")
        return None

    log(f"Starting FastAPI server (waiting {SERVER_STARTUP_WAIT}s)...")
    proc = subprocess.Popen(
        [sys.executable, "-m", "uvicorn", "api:app", "--port", "8000"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    time.sleep(SERVER_STARTUP_WAIT)
    return proc


def test_server_health() -> bool:
    log("Checking server health endpoint (GET /)...")
    try:
        r = requests.get(f"{BASE_URL}/", timeout=5)
        ok = r.status_code == 200 and "message" in r.json()
        record("API Server Health", ok, r.json().get("message", ""))
        return ok
    except Exception as exc:
        record("API Server Health", False, str(exc))
        return False


# ---------------------------------------------------------------------------
# Test 4 — Image detection API
# ---------------------------------------------------------------------------

def test_api_image() -> None:
    log("Testing POST /detect/image...")

    if not TEST_IMAGE.exists():
        record("API Image Endpoint", False, f"{TEST_IMAGE} not found")
        return

    try:
        with open(TEST_IMAGE, "rb") as f:
            r = requests.post(
                f"{BASE_URL}/detect/image",
                files={"file": ("test.jpg", f, "image/jpeg")},
                timeout=SERVER_REQUEST_TIMEOUT,
            )

        ok = r.status_code == 200
        body = r.json() if ok else {}
        has_detections = "detections" in body
        has_stats = "stats" in body
        det_count = len(body.get("detections", []))

        log(f"  Status: {r.status_code}")
        log(f"  Detections: {det_count}")
        log(f"  Stats included: {has_stats}")

        record("API Image Endpoint", ok and has_detections, f"{det_count} detections, status={r.status_code}")

    except Exception as exc:
        record("API Image Endpoint", False, str(exc))


# ---------------------------------------------------------------------------
# Test 5 — Video detection API
# ---------------------------------------------------------------------------

def test_api_video() -> None:
    log("Testing POST /detect/video...")

    if not TEST_VIDEO.exists():
        record("API Video Endpoint", False, f"{TEST_VIDEO} not found")
        return

    try:
        with open(TEST_VIDEO, "rb") as f:
            r = requests.post(
                f"{BASE_URL}/detect/video",
                files={"file": ("test.mp4", f, "video/mp4")},
                timeout=SERVER_REQUEST_TIMEOUT,
            )

        ok = r.status_code == 200
        body = r.json() if ok else {}
        has_results = "results" in body
        has_stats = "stats" in body
        has_heatmaps = "heatmaps" in body
        frames = body.get("frames_sampled", 0)

        log(f"  Status: {r.status_code}")
        log(f"  Frames sampled: {frames}")
        log(f"  Stats included: {has_stats}")
        log(f"  Heatmaps included: {has_heatmaps}")

        record("API Video Endpoint", ok and has_results, f"{frames} frames, status={r.status_code}")

    except Exception as exc:
        record("API Video Endpoint", False, str(exc))


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

def print_report() -> None:
    print("\n")
    print("=" * 44)
    print("         SYSTEM TEST REPORT")
    print("=" * 44)

    all_pass = True
    for name, status in results.items():
        icon = "\033[92m✓\033[0m" if status == "PASS" else "\033[91m✗\033[0m"
        print(f"  {icon}  {name}: {status}")
        if status != "PASS":
            all_pass = False

    print("-" * 44)
    if all_pass:
        print("  \033[92mSYSTEM STATUS: ALL TESTS PASSED\033[0m")
    else:
        print("  \033[91mSYSTEM STATUS: SOME TESTS FAILED\033[0m")
    print("=" * 44)
    print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    print()
    print("=" * 44)
    print("  Football Vision AI — System Test")
    print("=" * 44)
    print()

    # 1. Model file
    test_model_file()

    # 2. Local detection
    test_local_detection()

    # 3-5. API tests (start server if needed)
    server_proc = start_server()

    try:
        server_up = test_server_health()

        if server_up:
            test_api_image()
            test_api_video()
        else:
            record("API Image Endpoint", False, "Server not reachable")
            record("API Video Endpoint", False, "Server not reachable")
    finally:
        if server_proc is not None:
            log("Shutting down test server...")
            os.kill(server_proc.pid, signal.SIGTERM)
            server_proc.wait(timeout=5)

    print_report()

    all_pass = all(s == "PASS" for s in results.values())
    sys.exit(0 if all_pass else 1)


if __name__ == "__main__":
    main()
