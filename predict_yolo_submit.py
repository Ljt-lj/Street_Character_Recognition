"""
Generate competition submission CSV from a trained YOLO detector.

Reads test images, sorts digit boxes left-to-right, concatenates class ids.

Usage:
  python predict_yolo_submit.py --weights runs/detect/mchar_digits/weights/best.pt --out yolo_submit.csv
  python predict_yolo_submit.py --weights ... --device cpu
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from yolo_device import resolve_yolo_device


ROOT = Path(__file__).resolve().parent


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--weights", required=True)
    p.add_argument("--test-dir", default=str(ROOT / "dataset/mchar_test_a"))
    p.add_argument("--out", default=str(ROOT / "yolo_submit.csv"))
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--conf", type=float, default=0.25)
    p.add_argument(
        "--device",
        default="auto",
        help='"auto", "cpu", or GPU id (e.g. 0).',
    )
    return p.parse_args()


def pred_string_from_result(result) -> str:
    if result.boxes is None or len(result.boxes) == 0:
        return ""
    xyxy = result.boxes.xyxy.cpu().numpy()
    cls = result.boxes.cls.cpu().numpy().astype(int)
    xc = (xyxy[:, 0] + xyxy[:, 2]) / 2.0
    order = xc.argsort()
    return "".join(str(int(cls[i])) for i in order)


def main() -> None:
    args = parse_args()
    from ultralytics import YOLO

    test_dir = Path(args.test_dir)
    paths = sorted(test_dir.glob("*.png"))
    if not paths:
        raise SystemExit(f"No PNG under {test_dir}")

    device = resolve_yolo_device(args.device)
    print(f"Using device: {device}")

    model = YOLO(args.weights)
    device_kw = {"device": device}

    rows = []
    for p in tqdm(paths, desc="predict"):
        results = model.predict(
            source=str(p),
            imgsz=args.imgsz,
            conf=args.conf,
            verbose=False,
            **device_kw,
        )
        code = pred_string_from_result(results[0])
        rows.append({"file_name": p.name, "file_code": code})

    df = pd.DataFrame(rows).sort_values("file_name")
    df.to_csv(args.out, index=False)
    print(f"Wrote {len(df)} rows to {args.out}")


if __name__ == "__main__":
    main()
