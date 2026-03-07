# =============================================================================
# download_dataset.py — Football detection dataset downloader
#
# Supports two download sources (choose with --source flag):
#
#   1. kaggle  (default)
#      Downloads "adilshamim8/football-players-detection" from Kaggle.
#      Requires a Kaggle API token at ~/.kaggle/kaggle.json
#      Get yours at: https://www.kaggle.com/settings → API → Create New Token
#
#   2. roboflow
#      Downloads the Roboflow "football-players-detection" universe dataset.
#      Requires a free Roboflow API key (pass via --rf-key or RF_API_KEY env var).
#      Get yours at: https://app.roboflow.com → Settings → Roboflow API
#
# Usage:
#   python download_dataset.py                        # Kaggle, default paths
#   python download_dataset.py --source roboflow --rf-key <YOUR_KEY>
#   python download_dataset.py --source kaggle --dataset adilshamim8/football-players-detection
#
# Output structure:
#   backend/dataset/
#       images/
#           train/   (.jpg / .png)
#           val/
#       labels/
#           train/   (YOLO .txt annotations)
#           val/
# =============================================================================

from __future__ import annotations

import argparse
import os
import shutil
import sys
import zipfile
from pathlib import Path

# Load .env automatically so KAGGLE_USERNAME, KAGGLE_KEY, RF_API_KEY
# are available as environment variables without any manual export.
try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).parent / ".env")
except ImportError:
    pass  # python-dotenv is optional; env vars can still be set manually


# ── Constants ─────────────────────────────────────────────────────────────────

# Kaggle dataset slug — replace with any compatible YOLO-format dataset
DEFAULT_KAGGLE_DATASET = "adilshamim8/football-players-detection"

# Roboflow workspace / project / version
RF_WORKSPACE = "roboflow-jvuqo"
RF_PROJECT   = "football-players-detection-3zvbc"
RF_VERSION   = 1

# Local paths
DATASET_DIR  = Path("dataset")
RAW_ZIP      = Path("_raw_download.zip")
RAW_EXTRACT  = Path("_raw_extracted")

# YOLO split names we want to produce
SPLITS = ("train", "val")


# ── Utilities ─────────────────────────────────────────────────────────────────

def _make_dirs() -> None:
    """Create the canonical dataset directory tree if it doesn't exist."""
    for split in SPLITS:
        (DATASET_DIR / "images" / split).mkdir(parents=True, exist_ok=True)
        (DATASET_DIR / "labels" / split).mkdir(parents=True, exist_ok=True)
    print(f"[dataset] Output directory ready: {DATASET_DIR.resolve()}")


def _count(directory: Path, extensions: tuple[str, ...]) -> int:
    """Count files in `directory` matching any of the given extensions."""
    if not directory.exists():
        return 0
    return sum(1 for f in directory.iterdir() if f.suffix.lower() in extensions)


def _print_stats() -> None:
    """Print a summary table of the assembled dataset."""
    img_ext = (".jpg", ".jpeg", ".png", ".bmp", ".webp")
    lbl_ext = (".txt",)

    print("\n" + "=" * 50)
    print("  DATASET STATISTICS")
    print("=" * 50)
    for split in SPLITS:
        n_img = _count(DATASET_DIR / "images" / split, img_ext)
        n_lbl = _count(DATASET_DIR / "labels" / split, lbl_ext)
        print(f"  {split:6s}  images: {n_img:>5}   labels: {n_lbl:>5}")
    print("=" * 50)


def _cleanup() -> None:
    """Remove temporary download artefacts."""
    if RAW_ZIP.exists():
        RAW_ZIP.unlink()
    if RAW_EXTRACT.exists():
        shutil.rmtree(RAW_EXTRACT)


# ── File helpers ──────────────────────────────────────────────────────────────

IMAGE_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
LABEL_EXT = {".txt"}


def _copy_files(src_dir: Path, split: str) -> None:
    """
    Walk `src_dir` and copy image / label files into the canonical layout.

    Handles two annotation formats:
      A) YOLO .txt files  — copied as-is
      B) _annotations.csv — converted to per-image YOLO .txt files
         CSV columns: filename, width, height, class, xmin, ymin, xmax, ymax
    """
    img_dest = DATASET_DIR / "images" / split
    lbl_dest = DATASET_DIR / "labels" / split

    img_count = lbl_count = 0

    # ── Copy images ────────────────────────────────────────────────────────────
    for file in src_dir.rglob("*"):
        if file.is_file() and file.suffix.lower() in IMAGE_EXT:
            shutil.copy2(file, img_dest / file.name)
            img_count += 1

    # ── Handle annotations ─────────────────────────────────────────────────────
    csv_file  = src_dir / "_annotations.csv"
    yolo_txts = list(src_dir.rglob("*.txt"))
    # Exclude class-list metadata files
    yolo_txts = [f for f in yolo_txts if f.name not in ("classes.txt", "obj.names")]

    if csv_file.exists():
        # Format B: convert CSV → per-image YOLO .txt
        lbl_count = _convert_csv_to_yolo(csv_file, lbl_dest)
    else:
        # Format A: copy existing YOLO .txt files
        for file in yolo_txts:
            content = file.read_text(errors="ignore").strip()
            if content and _is_yolo_label(content):
                shutil.copy2(file, lbl_dest / file.name)
                lbl_count += 1

    print(f"[dataset]   {split:6s} → {img_count} images, {lbl_count} labels")


# Class name → YOLO class_id mapping (must match football.yaml)
CLASS_MAP: dict[str, int] = {
    "player":     0,
    "ball":       1,
    "goalkeeper": 2,
    "referee":    3,
}


def _convert_csv_to_yolo(csv_path: Path, label_dest: Path) -> int:
    """
    Convert a Roboflow-style _annotations.csv to per-image YOLO .txt files.

    CSV columns expected:
      filename, width, height, class, xmin, ymin, xmax, ymax

    YOLO output per row:
      class_id  x_center  y_center  box_width  box_height  (all normalised 0-1)

    Returns the number of label files written.
    """
    import csv
    from collections import defaultdict

    annotations: dict[str, list[str]] = defaultdict(list)
    skipped = 0

    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            class_name = row["class"].strip().lower()
            class_id   = CLASS_MAP.get(class_name)

            if class_id is None:
                skipped += 1
                continue  # unknown class — skip rather than crash

            img_w  = float(row["width"])
            img_h  = float(row["height"])
            xmin   = float(row["xmin"])
            ymin   = float(row["ymin"])
            xmax   = float(row["xmax"])
            ymax   = float(row["ymax"])

            # Convert absolute pixel coords → normalised YOLO format
            x_center = ((xmin + xmax) / 2) / img_w
            y_center = ((ymin + ymax) / 2) / img_h
            box_w    = (xmax - xmin) / img_w
            box_h    = (ymax - ymin) / img_h

            # Clamp to [0, 1] to guard against annotation artefacts
            x_center = max(0.0, min(1.0, x_center))
            y_center = max(0.0, min(1.0, y_center))
            box_w    = max(0.0, min(1.0, box_w))
            box_h    = max(0.0, min(1.0, box_h))

            yolo_line = f"{class_id} {x_center:.6f} {y_center:.6f} {box_w:.6f} {box_h:.6f}"
            # Key by filename stem so the .txt matches the image filename
            stem = Path(row["filename"]).stem
            annotations[stem].append(yolo_line)

    # Write one .txt file per image
    for stem, lines in annotations.items():
        out_path = label_dest / f"{stem}.txt"
        out_path.write_text("\n".join(lines) + "\n")

    if skipped:
        print(f"[dataset]   (skipped {skipped} rows with unknown class names)")

    return len(annotations)


def _is_yolo_label(content: str) -> bool:
    """
    Return True if the file content looks like a YOLO annotation.
    Each non-empty line should have 5 space-separated numeric tokens:
      class_id  x_center  y_center  width  height
    """
    for line in content.splitlines():
        line = line.strip()
        if not line:
            continue
        parts = line.split()
        if len(parts) != 5:
            return False
        try:
            int(parts[0])           # class_id must be an integer
            [float(p) for p in parts[1:]]  # coordinates must be floats
        except ValueError:
            return False
    return True


# ── Source: Kaggle ─────────────────────────────────────────────────────────────

def _check_kaggle_credentials() -> None:
    """
    Ensure Kaggle credentials are available — either via env vars
    (KAGGLE_USERNAME + KAGGLE_KEY from .env) or via ~/.kaggle/kaggle.json.
    Env vars take priority; if neither is present, exit with clear instructions.
    """
    username = os.environ.get("KAGGLE_USERNAME", "")
    key      = os.environ.get("KAGGLE_KEY", "")

    if username and key:
        # Credentials already in environment (loaded from .env) — nothing to do
        print(f"[kaggle] Credentials loaded from .env (user: {username})")
        return

    # Fall back to the JSON file
    kaggle_json = Path.home() / ".kaggle" / "kaggle.json"
    if kaggle_json.exists():
        kaggle_json.chmod(0o600)
        print(f"[kaggle] Credentials loaded from {kaggle_json}")
        return

    sys.exit(
        "[kaggle] No credentials found.\n"
        "         Option A (recommended): add to backend/.env\n"
        "           KAGGLE_USERNAME=your_username\n"
        "           KAGGLE_KEY=your_api_key\n\n"
        "         Option B: place kaggle.json at ~/.kaggle/kaggle.json\n"
        "           Get it at: https://www.kaggle.com/settings → API → Create New Token\n"
    )


def download_from_kaggle(dataset_slug: str) -> None:
    """
    Download a Kaggle dataset, unzip it, and move files into the
    canonical YOLO dataset layout.

    Args:
        dataset_slug: "<owner>/<dataset-name>" as shown in the Kaggle URL.
    """
    _check_kaggle_credentials()

    # kaggle v2 exposes KaggleApi directly at the top level
    from kaggle import KaggleApi

    print("[kaggle] Authenticating…")
    api = KaggleApi()
    api.authenticate()

    print(f"[kaggle] Downloading dataset: {dataset_slug}")

    # Download all dataset files as a zip into cwd
    api.dataset_download_files(
        dataset_slug,
        path=".",
        unzip=False,
        quiet=False,
    )

    # Kaggle names the zip after the dataset slug's last segment
    slug_name = dataset_slug.split("/")[-1]
    downloaded_zip = Path(f"{slug_name}.zip")

    if not downloaded_zip.exists():
        candidates = list(Path(".").glob("*.zip"))
        if not candidates:
            sys.exit("[kaggle] Download failed — no zip file found after download.")
        downloaded_zip = candidates[0]

    print(f"[kaggle] Extracting: {downloaded_zip}")
    RAW_EXTRACT.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(downloaded_zip, "r") as zf:
        zf.extractall(RAW_EXTRACT)
    downloaded_zip.unlink()

    _organise_extracted(RAW_EXTRACT)


# ── Source: Roboflow ───────────────────────────────────────────────────────────

def download_from_roboflow(api_key: str) -> None:
    """
    Download the football players detection dataset from Roboflow Universe
    in YOLOv8 format, then move files into the canonical layout.

    Args:
        api_key: Roboflow private API key (from app.roboflow.com → Settings).
    """
    from roboflow import Roboflow

    print(f"[roboflow] Connecting to workspace: {RF_WORKSPACE}")
    rf      = Roboflow(api_key=api_key)
    project = rf.workspace(RF_WORKSPACE).project(RF_PROJECT)
    version = project.version(RF_VERSION)

    print(f"[roboflow] Downloading dataset version {RF_VERSION} in YOLOv8 format…")
    # Downloads and extracts to a folder named after the project in cwd
    dataset = version.download("yolov8", location=str(RAW_EXTRACT))

    print(f"[roboflow] Downloaded to: {dataset.location}")
    _organise_extracted(Path(dataset.location))


# ── Layout normalisation ───────────────────────────────────────────────────────

def _organise_extracted(root: Path) -> None:
    """
    Walk the extracted directory tree and copy image / label files into
    the canonical DATASET_DIR structure.

    Handles two common layouts that datasets ship in:

    Layout A (explicit split folders):
        root/train/images/*.jpg   root/train/labels/*.txt
        root/valid/images/*.jpg   root/valid/labels/*.txt

    Layout B (flat — no split folders):
        root/images/*.jpg
        root/labels/*.txt
        → All assigned to 'train'; val remains empty (user should split manually)

    Layout C (Roboflow YOLOv8 export):
        root/train/images/   root/train/labels/
        root/valid/images/   root/valid/labels/
    """
    print(f"[dataset] Organising files from: {root}")

    # Map variant spellings to canonical split names
    split_aliases = {
        "train":  "train",
        "training": "train",
        "val":    "val",
        "valid":  "val",
        "validation": "val",
        "test":   "val",   # treat test as val when no dedicated val exists
    }

    assigned: dict[str, int] = {s: 0 for s in SPLITS}

    # ── Try Layout A / C first ─────────────────────────────────────────────
    found_splits = False
    for child in root.iterdir():
        canonical = split_aliases.get(child.name.lower())
        if child.is_dir() and canonical:
            found_splits = True
            _copy_files(child, canonical)
            assigned[canonical] += 1

    if found_splits:
        # Check for a nested extra layer (e.g. root/train/images/train/...)
        _cleanup_raw()
        return

    # ── Layout B: flat structure ───────────────────────────────────────────
    print("[dataset] No split folders detected — placing all files in 'train'.")
    print("[dataset] You should manually split some images into 'val/' before training.")
    _copy_files(root, "train")
    _cleanup_raw()


def _cleanup_raw() -> None:
    if RAW_EXTRACT.exists():
        shutil.rmtree(RAW_EXTRACT)


# ── Entry point ───────────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download and prepare a football detection dataset for YOLOv8.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--source",
        choices=["kaggle", "roboflow"],
        default="kaggle",
        help="Dataset source",
    )
    parser.add_argument(
        "--dataset",
        default=DEFAULT_KAGGLE_DATASET,
        help="Kaggle dataset slug: <owner>/<name>",
    )
    parser.add_argument(
        "--rf-key",
        default=os.environ.get("RF_API_KEY", ""),
        help="Roboflow API key (or set RF_API_KEY in .env)",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()

    # ── Step 1: create output directories ────────────────────────────────────
    _make_dirs()

    # ── Step 2: download ──────────────────────────────────────────────────────
    try:
        if args.source == "kaggle":
            download_from_kaggle(args.dataset)
        else:
            if not args.rf_key:
                sys.exit(
                    "[roboflow] No API key provided.\n"
                    "           Pass --rf-key <KEY> or set the RF_API_KEY environment variable.\n"
                    "           Get your key at: https://app.roboflow.com → Settings → API"
                )
            download_from_roboflow(args.rf_key)
    except KeyboardInterrupt:
        print("\n[dataset] Download cancelled.")
        _cleanup()
        sys.exit(1)
    except Exception as exc:
        _cleanup()
        sys.exit(f"[dataset] Download failed: {exc}")

    # ── Step 3: print statistics ──────────────────────────────────────────────
    _print_stats()

    print(
        "\n[dataset] Ready. Start training with:\n"
        "          python train.py\n"
    )


if __name__ == "__main__":
    main()
