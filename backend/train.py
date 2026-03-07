# =============================================================================
# train.py — YOLOv8 training pipeline for Football Vision AI
#
# Usage:
#   python train.py                            # default: 50 epochs, yolov8m.pt
#   python train.py --epochs 100 --batch 16   # custom hyperparameters
#   python train.py --device cpu              # force CPU training
#
# After training, weights are saved to:
#   runs/detect/football_model/weights/best.pt   ← swap into detect.py
#   runs/detect/football_model/weights/last.pt
# =============================================================================

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from ultralytics import YOLO


# ── Training hyperparameters ──────────────────────────────────────────────────

DATA    = "football.yaml"      # dataset config — defines classes and split paths
MODEL   = "yolov8m.pt"         # pretrained base checkpoint (~52 MB, auto-downloaded)
EPOCHS  = 50                   # number of full passes over the training set
IMGSZ   = 640                  # input resolution (pixels); YOLOv8 default
BATCH   = 8                    # images per gradient update step
PROJECT = "runs"               # parent output directory — YOLO appends /detect/<name> automatically
NAME    = "football_model"     # sub-directory for this run's weights / logs


# ── Dataset validation ────────────────────────────────────────────────────────

def _validate_dataset(data_yaml: str) -> None:
    """
    Verify the dataset directory structure exists before starting a long
    training run.  Exits with a clear error message if anything is missing.
    """
    import yaml

    yaml_path = Path(data_yaml)
    if not yaml_path.exists():
        sys.exit(
            f"[train] Dataset config not found: {yaml_path}\n"
            "        Run download_dataset.py first, or check the path."
        )

    with open(yaml_path) as f:
        cfg = yaml.safe_load(f)

    # Resolve dataset root — may be relative to the yaml file or absolute
    root = Path(cfg.get("path", "."))
    if not root.is_absolute():
        root = yaml_path.parent / root

    # Check both split directories exist
    missing = []
    for split in ("train", "val"):
        split_dir = root / cfg.get(split, f"images/{split}")
        if not split_dir.exists():
            missing.append(str(split_dir))

    if missing:
        sys.exit(
            "[train] Missing dataset directories:\n"
            + "\n".join(f"        {p}" for p in missing)
            + "\n\n        Run: python download_dataset.py"
        )

    # Quick image count for confidence
    train_dir = root / cfg.get("train", "images/train")
    n = len(list(train_dir.glob("*.jpg")) + list(train_dir.glob("*.png")))
    print(f"[train] Dataset OK — {n} training images in {train_dir}")


# ── Core training function ────────────────────────────────────────────────────

def train_model(
    data:    str = DATA,
    model:   str = MODEL,
    epochs:  int = EPOCHS,
    imgsz:   int = IMGSZ,
    batch:   int = BATCH,
    project: str = PROJECT,
    name:    str = NAME,
    device:  str = "cpu",
) -> None:
    """
    Fine-tune a pretrained YOLOv8 model on the football analytics dataset.

    Pipeline:
      1. Validate the dataset directory structure.
      2. Load the YOLOv8 base checkpoint (downloaded automatically on first run).
      3. Run training — weights, logs, and curves saved to runs/detect/football_model/.
      4. Print final validation metrics: mAP, Precision, Recall.
      5. Print the path to the best saved weights file.

    Args:
        data:    Path to the YOLO dataset YAML (football.yaml).
        model:   YOLOv8 checkpoint to fine-tune from (yolov8n/s/m/l/x.pt).
        epochs:  Number of training epochs.
        imgsz:   Input image resolution in pixels (square).
        batch:   Images per batch. Use -1 for auto-batch.
        project: Root output directory for all run artefacts.
        name:    Sub-directory name for this specific run.
        device:  Compute device — "auto", "cpu", "0" (GPU), "0,1" (multi-GPU).
    """

    # ── Step 1: validate dataset ──────────────────────────────────────────────
    print(f"\n[train] Validating dataset: {data}")
    _validate_dataset(data)

    # ── Step 2: load pretrained base model ────────────────────────────────────
    # YOLOv8m (medium) offers a strong accuracy/speed balance.
    # The weights are downloaded from the Ultralytics CDN on first use.
    print(f"[train] Loading base model  : {model}")
    yolo = YOLO(model)

    # ── Step 3: run training ──────────────────────────────────────────────────
    # Ultralytics handles the full loop: forward pass, loss, backprop,
    # LR scheduling, augmentation, and checkpointing.
    print(
        f"[train] Starting training\n"
        f"         epochs = {epochs}\n"
        f"         imgsz  = {imgsz}\n"
        f"         batch  = {batch}\n"
        f"         device = {device}\n"
    )

    results = yolo.train(
        data     = data,
        epochs   = epochs,
        imgsz    = imgsz,
        batch    = batch,
        project  = project,
        name     = name,
        device   = device,

        # ── Reproducibility ───────────────────────────────────────────────────
        seed     = 42,

        # ── Augmentation — tuned for broadcast football footage ───────────────
        hsv_h    = 0.015,   # small hue shift — handles varied pitch lighting
        hsv_s    = 0.7,     # saturation jitter — overcast vs. sunny conditions
        hsv_v    = 0.4,     # brightness jitter — shadows / floodlights
        fliplr   = 0.5,     # horizontal flip — pitch is laterally symmetric
        mosaic   = 1.0,     # mosaic mix — boosts small-object (ball) detection
        scale    = 0.5,     # random scale ±50% — player size variation by depth

        # ── Run management ────────────────────────────────────────────────────
        exist_ok = True,    # allow resuming / overwriting existing run folder
        verbose  = True,    # print per-epoch progress to the terminal
    )

    # ── Step 4: print evaluation metrics ─────────────────────────────────────
    # Ultralytics stores final val metrics in results.results_dict
    metrics = results.results_dict

    print("\n" + "=" * 60)
    print("  TRAINING COMPLETE")
    print("=" * 60)
    print(f"  mAP@50          : {metrics.get('metrics/mAP50(B)',    0):.4f}")
    print(f"  mAP@50-95       : {metrics.get('metrics/mAP50-95(B)', 0):.4f}")
    print(f"  Precision       : {metrics.get('metrics/precision(B)', 0):.4f}")
    print(f"  Recall          : {metrics.get('metrics/recall(B)',    0):.4f}")

    # ── Step 5: report saved weights location ─────────────────────────────────
    best = Path(project) / name / "weights" / "best.pt"
    print(f"\n  Best weights    : {best.resolve()}")
    print(
        "\n  To run detection with the trained model, set in detect.py:\n"
        f'    WEIGHTS = "{best}"\n'
    )
    print("=" * 60)


# ── CLI wrapper ───────────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train YOLOv8 for football player detection.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--data",    default=DATA,    help="Dataset YAML path")
    parser.add_argument("--model",   default=MODEL,   help="YOLOv8 checkpoint")
    parser.add_argument("--epochs",  default=EPOCHS,  type=int, help="Training epochs")
    parser.add_argument("--imgsz",   default=IMGSZ,   type=int, help="Input image size")
    parser.add_argument("--batch",   default=BATCH,   type=int, help="Batch size (-1 = auto)")
    parser.add_argument("--project", default=PROJECT, help="Output directory")
    parser.add_argument("--name",    default=NAME,    help="Run sub-directory name")
    parser.add_argument("--device",  default="cpu",   help='Device: "cpu", "0" (GPU), "0,1" (multi-GPU)')
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    train_model(
        data    = args.data,
        model   = args.model,
        epochs  = args.epochs,
        imgsz   = args.imgsz,
        batch   = args.batch,
        project = args.project,
        name    = args.name,
        device  = args.device,
    )
