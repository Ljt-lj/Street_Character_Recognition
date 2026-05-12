"""Shared helpers for YOLO sequence-level validation accuracy (mchar)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from tqdm import tqdm


def pred_string_from_result(result: Any) -> str:
    if result.boxes is None or len(result.boxes) == 0:
        return ""
    xyxy = result.boxes.xyxy.cpu().numpy()
    cls = result.boxes.cls.cpu().numpy().astype(int)
    xc = (xyxy[:, 0] + xyxy[:, 2]) / 2.0
    order = xc.argsort()
    return "".join(str(int(cls[i])) for i in order)


def sequence_accuracy_one_model(
    model: Any,
    marks: dict,
    img_root: Path,
    *,
    imgsz: int,
    conf: float,
    device: str | int,
    desc: str = "val",
) -> tuple[float, int, int]:
    """Return (accuracy, correct, total) for one conf threshold; model already loaded."""
    device_kw = {"device": device}
    correct = 0
    total = 0
    for name, ann in tqdm(marks.items(), desc=desc):
        path = img_root / name
        if not path.is_file():
            continue
        gt = "".join(str(int(x)) for x in ann["label"])
        results = model.predict(
            source=str(path),
            imgsz=imgsz,
            conf=conf,
            verbose=False,
            **device_kw,
        )
        pred = pred_string_from_result(results[0])
        total += 1
        if pred == gt:
            correct += 1
    acc = correct / total if total else 0.0
    return acc, correct, total


def sweep_confidence(
    weights: str | Path,
    marks: dict,
    img_root: Path,
    confs: list[float],
    *,
    imgsz: int,
    device: str | int,
    max_val_samples: int | None = None,
) -> tuple[float, float, dict[float, float]]:
    """
    Load weights once; try each conf. Returns (best_acc, best_conf, conf_to_acc).
    If max_val_samples is set, only the first N keys (sorted by filename) are used.
    """
    from ultralytics import YOLO

    if max_val_samples is not None and len(marks) > max_val_samples:
        keys = sorted(marks.keys())[:max_val_samples]
        marks = {k: marks[k] for k in keys}

    model = YOLO(str(weights))
    table: dict[float, float] = {}
    best_acc, best_c = -1.0, confs[0] if confs else 0.25
    for c in confs:
        acc, _, _ = sequence_accuracy_one_model(
            model,
            marks,
            img_root,
            imgsz=imgsz,
            conf=c,
            device=device,
            desc=f"val@conf={c}",
        )
        table[c] = acc
        if acc > best_acc:
            best_acc, best_c = acc, c
    return best_acc, best_c, table
