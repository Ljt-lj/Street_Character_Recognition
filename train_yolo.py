"""
Train Ultralytics YOLO on prepared mchar digit detection data.

Prerequisites:
  pip install ultralytics
  python prepare_yolo_dataset.py

Models download automatically from Ultralytics (same ecosystem as
https://github.com/ultralytics/ultralytics ). Pick a larger variant for higher accuracy if VRAM allows.

Usage:
  python train_yolo.py
  python train_yolo.py --model yolo11m.pt --epochs 120 --imgsz 640
  python train_yolo.py --device cpu          # force CPU
  python train_yolo.py --device 0            # first GPU
"""

from __future__ import annotations

import argparse
from pathlib import Path

from ultralytics import YOLO

from yolo_device import device_is_cpu, resolve_yolo_device


ROOT = Path(__file__).resolve().parent


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train YOLO digit detector for mchar.")
    p.add_argument(
        "--model",
        default="yolo11m.pt",
        help="Checkpoint name or path (e.g. yolo11n.pt, yolo11m.pt, yolo11l.pt, yolov8m.pt).",
    )
    p.add_argument("--data", default=str(ROOT / "svhn_digits.yaml"), help="Dataset YAML.")
    p.add_argument("--epochs", type=int, default=120)
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--batch", type=int, default=16, help="Use -1 for auto batch.")
    p.add_argument(
        "--device",
        default="auto",
        help='Computation device: "auto" (GPU if available else CPU), "cpu", or GPU id like 0.',
    )
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--project", default=str(ROOT / "runs/detect"))
    p.add_argument("--name", default="mchar_digits")
    p.add_argument("--patience", type=int, default=40)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--fraction",
        type=float,
        default=1.0,
        help="Use a fraction of training images (e.g. 0.01) for quick CPU smoke tests.",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    data_yaml = Path(args.data)
    if not data_yaml.is_file():
        raise SystemExit(f"Missing {data_yaml}. Run prepare_yolo_dataset.py first.")

    device = resolve_yolo_device(args.device)
    print(f"Using device: {device}")

    model = YOLO(args.model)
    train_kw = dict(
        data=str(data_yaml),
        epochs=args.epochs,
        imgsz=args.imgsz,
        patience=args.patience,
        seed=args.seed,
        workers=args.workers,
        project=args.project,
        name=args.name,
        exist_ok=True,
        verbose=True,
        cos_lr=True,
        warmup_epochs=3,
        close_mosaic=10,
        amp=not device_is_cpu(device),
        fraction=args.fraction,
        device=device,
    )
    if args.batch > 0:
        train_kw["batch"] = args.batch

    model.train(**train_kw)
    best = Path(args.project) / args.name / "weights" / "best.pt"
    print(f"Training finished. Best weights: {best}")


if __name__ == "__main__":
    main()
