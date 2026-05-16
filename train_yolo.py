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
  python train_yolo.py --resume              # same run: PROJECT/NAME/weights/last.pt
  python train_yolo.py --resume runs/detect/<run>/weights/last.pt
  python train_yolo.py --continue-from runs/detect/<run>/weights/last.pt   # same as --resume path
  python train_yolo.py --continue-from runs/detect/<旧run>/weights/best.pt --name mchar_r2 --epochs 80
  python train_yolo.py --cheat --model yolo11m.pt --device 0   # train+val in training split (leakage)
"""

from __future__ import annotations

import argparse
from pathlib import Path

from ultralytics import YOLO

from yolo_device import device_is_cpu, resolve_yolo_device


ROOT = Path(__file__).resolve().parent
CHEAT_YAML = ROOT / "svhn_digits_cheat.yaml"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train YOLO digit detector for mchar.")
    p.add_argument(
        "--model",
        default="yolo11m.pt",
        help="Checkpoint name or path. Ignored when --resume or when --continue-from points to last.pt.",
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
    p.add_argument(
        "--resume",
        nargs="?",
        const="__auto__",
        default=None,
        metavar="LAST_PT",
        help=(
            "Resume same Ultralytics run from last.pt (optimizer + epoch). "
            "Alone: --project/--name/weights/last.pt. Or pass path. Mutually exclusive with --continue-from."
        ),
    )
    p.add_argument(
        "--continue-from",
        dest="continue_from",
        default=None,
        metavar="WEIGHTS",
        help=(
            "If path is last.pt: same as --resume (same run). "
            "Otherwise (e.g. best.pt): load as pretrained weights and train a new run under --project/--name "
            "(use a new --name, e.g. mchar_r2). Mutually exclusive with --resume."
        ),
    )
    p.add_argument(
        "--cheat",
        action="store_true",
        help="Use svhn_digits_cheat.yaml: training split includes val images (severe label leakage). "
        "For informal experiments only; invalid for fair val/test reporting.",
    )
    return p.parse_args()


def _resolve_data_yaml(args: argparse.Namespace) -> Path:
    if args.cheat:
        if not CHEAT_YAML.is_file():
            raise SystemExit(f"Missing {CHEAT_YAML}. It should ship with the repo next to svhn_digits.yaml.")
        print(
            "\n*** CHEAT MODE ***\n"
            "Training data includes BOTH mchar_train and mchar_val images (see svhn_digits_cheat.yaml).\n"
            "Validation-set metrics and eval_yolo_sequence on val are NOT fair — do not report as clean scores.\n"
        )
        return CHEAT_YAML
    data_yaml = Path(args.data)
    if not data_yaml.is_file():
        raise SystemExit(f"Missing {data_yaml}. Run prepare_yolo_dataset.py first.")
    return data_yaml


def main() -> None:
    args = parse_args()
    if args.resume is not None and args.continue_from is not None:
        raise SystemExit("Use only one of --resume and --continue-from.")

    data_yaml = _resolve_data_yaml(args)

    device = resolve_yolo_device(args.device)
    print(f"Using device: {device}")
    print(f"Dataset YAML: {data_yaml}")

    last_path: Path | None = None

    if args.continue_from is not None:
        w = Path(args.continue_from).expanduser().resolve()
        if not w.is_file():
            raise SystemExit(f"--continue-from not found: {w}")
        if w.name == "last.pt":
            last_path = w
        else:
            print(f"Warm-start from weights: {w}")
            model = YOLO(str(w))
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
            return

    if args.resume is not None:
        if args.resume == "__auto__":
            last_path = Path(args.project) / args.name / "weights" / "last.pt"
        else:
            last_path = Path(args.resume).expanduser().resolve()

    if last_path is not None:
        if not last_path.is_file():
            raise SystemExit(
                f"Cannot resume: missing {last_path}\n"
                "Use the same --project and --name as the interrupted run, or pass the full path to last.pt."
            )
        print(f"Resume from: {last_path}")
        model = YOLO(str(last_path))
        train_kw = dict(
            resume=True,
            data=str(data_yaml),
            epochs=args.epochs,
            imgsz=args.imgsz,
            patience=args.patience,
            workers=args.workers,
            verbose=True,
            amp=not device_is_cpu(device),
            device=device,
        )
        if args.batch > 0:
            train_kw["batch"] = args.batch
        model.train(**train_kw)
        run_root = last_path.parent.parent
        best = run_root / "weights" / "best.pt"
        print(f"Training finished. Best weights: {best}")
        return

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
