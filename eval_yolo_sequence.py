"""
Sequence accuracy on val set: left-to-right digit order vs JSON ground truth.

Uses trained detector; counts exact string matches (same metric spirit as classification baseline).

Usage:
  python eval_yolo_sequence.py --weights runs/detect/mchar_digits/weights/best.pt
  python eval_yolo_sequence.py --weights ... --device cpu
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from tqdm import tqdm

from yolo_device import resolve_yolo_device
from yolo_metrics import pred_string_from_result


ROOT = Path(__file__).resolve().parent


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--weights", required=True, help="path to best.pt")
    p.add_argument("--val-json", default=str(ROOT / "dataset/mchar_val.json"))
    p.add_argument("--val-img", default=str(ROOT / "dataset/mchar_val"))
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--conf", type=float, default=0.25)
    p.add_argument(
        "--device",
        default="auto",
        help='"auto", "cpu", or GPU id (e.g. 0).',
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    from ultralytics import YOLO

    marks = json.loads(Path(args.val_json).read_text(encoding="utf-8"))
    device = resolve_yolo_device(args.device)
    print(f"Using device: {device}")

    model = YOLO(args.weights)
    device_kw = {"device": device}

    correct = 0
    total = 0
    img_root = Path(args.val_img)

    for name, ann in tqdm(marks.items(), desc="val"):
        path = img_root / name
        if not path.is_file():
            continue
        gt = "".join(str(int(x)) for x in ann["label"])
        results = model.predict(
            source=str(path),
            imgsz=args.imgsz,
            conf=args.conf,
            verbose=False,
            **device_kw,
        )
        pred = pred_string_from_result(results[0])
        total += 1
        if pred == gt:
            correct += 1

    acc = correct / total if total else 0.0
    print(f"Sequence accuracy: {acc:.4f} ({correct}/{total})")


if __name__ == "__main__":
    main()
