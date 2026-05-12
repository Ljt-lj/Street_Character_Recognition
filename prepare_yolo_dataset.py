"""
Convert mchar_train.json / mchar_val.json + PNG folders into Ultralytics YOLO layout.

Layout under dataset/yolo/:
  images/train, images/val  — hardlinks (or copies) to original PNGs
  labels/train, labels/val  — one .txt per image: class cx cy w h (normalized)

Classes 0..9 correspond to digits '0'..'9'.

Usage:
  python prepare_yolo_dataset.py
"""

from __future__ import annotations

import json
import os
import shutil
from pathlib import Path

from PIL import Image
from tqdm import tqdm


ROOT = Path(__file__).resolve().parent
DATASET = ROOT / "dataset"
SRC_TRAIN_IMG = DATASET / "mchar_train"
SRC_VAL_IMG = DATASET / "mchar_val"
JSON_TRAIN = DATASET / "mchar_train.json"
JSON_VAL = DATASET / "mchar_val.json"
DST = DATASET / "yolo"


def link_or_copy(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        return
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def box_to_yolo_line(left: float, top: float, w: float, h: float, iw: int, ih: int) -> str:
    """Single YOLO line: cls xc yc w h (0-1)."""
    xc = (left + w / 2.0) / iw
    yc = (top + h / 2.0) / ih
    nw = w / iw
    nh = h / ih
    xc = min(max(xc, 0.0), 1.0)
    yc = min(max(yc, 0.0), 1.0)
    nw = min(max(nw, 1e-6), 1.0)
    nh = min(max(nh, 1e-6), 1.0)
    return f"{xc:.6f} {yc:.6f} {nw:.6f} {nh:.6f}"


def process_split(
    img_dir: Path,
    json_path: Path,
    dst_images: Path,
    dst_labels: Path,
) -> None:
    marks = json.loads(json_path.read_text(encoding="utf-8"))
    missing_img = 0
    for name, ann in tqdm(marks.items(), desc=f"YOLO export ({dst_images.parent.name}/{dst_images.name})"):
        src = img_dir / name
        if not src.is_file():
            missing_img += 1
            continue
        with Image.open(src) as im:
            iw, ih = im.size
        lines = []
        n = len(ann["label"])
        for i in range(n):
            digit = int(ann["label"][i])
            cls = digit  # 0..9
            left = float(ann["left"][i])
            top = float(ann["top"][i])
            w = float(ann["width"][i])
            h = float(ann["height"][i])
            rest = box_to_yolo_line(left, top, w, h, iw, ih)
            lines.append(f"{cls} {rest}")

        link_or_copy(src, dst_images / name)
        dst_lab = dst_labels / f"{Path(name).stem}.txt"
        dst_lab.parent.mkdir(parents=True, exist_ok=True)
        dst_lab.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")

    if missing_img:
        print(f"Warning: {missing_img} entries skipped (image file missing).")


def main() -> None:
    if not JSON_TRAIN.is_file() or not JSON_VAL.is_file():
        raise SystemExit(f"Missing JSON: {JSON_TRAIN} or {JSON_VAL}")
    if not SRC_TRAIN_IMG.is_dir() or not SRC_VAL_IMG.is_dir():
        raise SystemExit(f"Expect image folders: {SRC_TRAIN_IMG} and {SRC_VAL_IMG}")

    DST.mkdir(parents=True, exist_ok=True)
    process_split(
        SRC_TRAIN_IMG,
        JSON_TRAIN,
        DST / "images" / "train",
        DST / "labels" / "train",
    )
    process_split(
        SRC_VAL_IMG,
        JSON_VAL,
        DST / "images" / "val",
        DST / "labels" / "val",
    )
    print(f"Done. YOLO dataset root: {DST}")


if __name__ == "__main__":
    main()
