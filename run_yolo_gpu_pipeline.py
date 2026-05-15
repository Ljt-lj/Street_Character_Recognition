"""
End-to-end: train YOLO on GPU (or auto device), tune confidence on val sequence accuracy,
optionally train a larger model if the primary run is below a threshold.

Usage (full GPU run, project root — use your own terminal where CUDA works):
  python run_yolo_gpu_pipeline.py --require-gpu

Or double-click / run:
  run_yolo_full_gpu.bat

Quick smoke (tiny subset):
  python run_yolo_gpu_pipeline.py --quick

Local CPU (full train possible but slow; recommended defaults):
  python run_yolo_gpu_pipeline.py --cpu-preset

Resume Stage 1 from an interrupted run (same Ultralytics run via last.pt), then conf sweep / upgrades:
  python run_yolo_gpu_pipeline.py --require-gpu --resume-train runs/detect/<run>/weights/last.pt

Requires: prepare_yolo_dataset.py already run; ultralytics installed; local .pt weights.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path

from yolo_device import resolve_yolo_device

from yolo_metrics import sweep_confidence


ROOT = Path(__file__).resolve().parent


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train YOLO + val conf sweep + optional larger model.")
    p.add_argument("--primary-model", default="yolo11m.pt", help="First backbone checkpoint (path or name under project).")
    p.add_argument("--upgrade-model", default="yolo11l.pt", help="Larger model if primary below --min-acc.")
    p.add_argument(
        "--mega-upgrade-model",
        default="yolo11x.pt",
        help="If still below --min-acc after primary+upgrade, train this model (large VRAM). Empty string to skip.",
    )
    p.add_argument("--no-mega-upgrade", action="store_true", help="Do not run the mega (e.g. x) stage.")
    p.add_argument(
        "--require-gpu",
        action="store_true",
        help="Exit immediately if torch.cuda.is_available() is False (recommended on your GPU machine).",
    )
    p.add_argument(
        "--cpu-preset",
        action="store_true",
        help="Force CPU-friendly settings (device=cpu, cap batch/workers, subsample val sweep, skip mega model). See CPU.md.",
    )
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--batch", type=int, default=16, help="0 = omit (Ultralytics default); -1 passed as -1 for auto.")
    p.add_argument("--device", default="auto")
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--fraction", type=float, default=1.0)
    p.add_argument("--min-acc", type=float, default=0.88, help="If best val seq acc below this, train upgrade model.")
    p.add_argument("--no-upgrade", action="store_true", help="Never train the larger model.")
    p.add_argument(
        "--conf-sweep",
        default="0.2,0.25,0.3,0.35,0.4",
        help="Comma-separated conf thresholds for validation tuning.",
    )
    p.add_argument("--val-json", default=str(ROOT / "dataset/mchar_val.json"))
    p.add_argument("--val-img", default=str(ROOT / "dataset/mchar_val"))
    p.add_argument("--project", default=str(ROOT / "runs/detect"))
    p.add_argument("--quick", action="store_true", help="epochs=2, fraction=0.02, skip upgrade, single conf.")
    p.add_argument(
        "--val-max-samples",
        type=int,
        default=0,
        help="If >0, cap validation images for conf sweep (default full val; quick uses 1000 unless overridden).",
    )
    p.add_argument(
        "--resume-train",
        default=None,
        metavar="LAST_PT",
        help="Stage 1: resume primary training from weights/last.pt (same run as interrupted). "
        "Then run val conf sweep and optional upgrade stages. --epochs is the target total epochs.",
    )
    return p.parse_args()


def _resolve_model_path(s: str) -> str:
    p = Path(s)
    if p.is_file():
        return str(p.resolve())
    cand = ROOT / s
    if cand.is_file():
        return str(cand.resolve())
    raise SystemExit(f"Model weight not found: {s}")


def _try_resolve_model_path(s: str) -> str | None:
    p = Path(s)
    if p.is_file():
        return str(p.resolve())
    cand = ROOT / s
    if cand.is_file():
        return str(cand.resolve())
    return None


def run_train(
    *,
    model_path: str | None = None,
    run_name: str | None = None,
    resume_last: str | None = None,
    epochs: int,
    imgsz: int,
    batch: int,
    device: str,
    workers: int,
    fraction: float,
    project: str,
) -> Path:
    if resume_last:
        last_p = Path(resume_last).expanduser().resolve()
        if not last_p.is_file():
            raise SystemExit(f"--resume-train file not found: {last_p}")
        if last_p.name != "last.pt":
            raise SystemExit(f"--resume-train must point to weights/last.pt, got: {last_p}")
        cmd = [
            sys.executable,
            str(ROOT / "train_yolo.py"),
            "--resume",
            str(last_p),
            "--epochs",
            str(epochs),
            "--imgsz",
            str(imgsz),
            "--device",
            device,
            "--workers",
            str(workers),
        ]
        if batch != 0:
            cmd.extend(["--batch", str(batch)])
        print("Running:", " ".join(cmd))
        subprocess.check_call(cmd, cwd=str(ROOT))
        best = last_p.parent / "best.pt"
        if not best.is_file():
            raise SystemExit(f"Missing best.pt after resume train: {best}")
        return best

    if not model_path or not run_name:
        raise SystemExit("run_train: need model_path and run_name when not using resume_last.")
    cmd = [
        sys.executable,
        str(ROOT / "train_yolo.py"),
        "--model",
        model_path,
        "--name",
        run_name,
        "--epochs",
        str(epochs),
        "--imgsz",
        str(imgsz),
        "--device",
        device,
        "--workers",
        str(workers),
        "--fraction",
        str(fraction),
        "--project",
        project,
    ]
    if batch != 0:
        cmd.extend(["--batch", str(batch)])
    print("Running:", " ".join(cmd))
    subprocess.check_call(cmd, cwd=str(ROOT))
    best = Path(project) / run_name / "weights" / "best.pt"
    if not best.is_file():
        raise SystemExit(f"Missing best.pt after train: {best}")
    return best


def main() -> None:
    args = parse_args()
    if args.require_gpu and args.cpu_preset:
        raise SystemExit("Use only one of --require-gpu or --cpu-preset.")
    if args.cpu_preset:
        args.device = "cpu"
        args.no_mega_upgrade = True
        args.workers = min(args.workers, 2)
        if args.batch > 4:
            args.batch = 4
        if args.val_max_samples <= 0:
            args.val_max_samples = 5000
        print(
            "[cpu-preset] device=cpu, batch=%d, workers=%d, val_max_samples=%d (conf sweep), no mega upgrade"
            % (args.batch, args.workers, args.val_max_samples)
        )
    if args.require_gpu:
        import torch

        if not torch.cuda.is_available():
            raise SystemExit(
                "CUDA is not available in this Python (torch.cuda.is_available() == False).\n"
                "Install CUDA build of PyTorch + NVIDIA driver, then run again outside a CPU-only sandbox."
            )
        print("require-gpu: OK —", torch.cuda.get_device_name(0))

    if not Path(args.val_json).is_file():
        raise SystemExit("Missing val json. Run prepare_yolo_dataset.py and check paths.")
    data_yaml = ROOT / "svhn_digits.yaml"
    if not data_yaml.is_file():
        raise SystemExit(f"Missing {data_yaml}")

    if args.quick:
        args.epochs = 2
        args.fraction = 0.02
        args.no_upgrade = True
        args.conf_sweep = "0.25"
        if args.val_max_samples <= 0:
            args.val_max_samples = 1000

    val_max = args.val_max_samples if args.val_max_samples > 0 else None
    device = resolve_yolo_device(args.device)
    print(f"Device resolved: {device}")

    marks = json.loads(Path(args.val_json).read_text(encoding="utf-8"))
    img_root = Path(args.val_img)
    confs = [float(x.strip()) for x in args.conf_sweep.split(",") if x.strip()]

    stamp = datetime.now().strftime("%m%d_%H%M")
    primary_path = _resolve_model_path(args.primary_model)
    stem = Path(primary_path).stem.replace(".", "_")

    if args.resume_train:
        last_p = Path(args.resume_train).expanduser().resolve()
        if not last_p.is_file():
            raise SystemExit(f"--resume-train not found: {last_p}")
        if last_p.name != "last.pt":
            raise SystemExit(f"--resume-train must be weights/last.pt, got: {last_p}")
        run_primary = last_p.parent.parent.name
        print("\n=== Stage 1: resume primary (from last.pt) ===\n")
        best_primary = run_train(
            resume_last=str(last_p),
            epochs=args.epochs,
            imgsz=args.imgsz,
            batch=args.batch,
            device=args.device,
            workers=args.workers,
            fraction=args.fraction,
            project=args.project,
        )
    else:
        run_primary = f"{stem}_{stamp}"
        print("\n=== Stage 1: train primary ===\n")
        best_primary = run_train(
            model_path=primary_path,
            run_name=run_primary,
            epochs=args.epochs,
            imgsz=args.imgsz,
            batch=args.batch,
            device=args.device,
            workers=args.workers,
            fraction=args.fraction,
            project=args.project,
        )

    print("\n=== Stage 2: confidence sweep (primary) ===\n")
    acc_p, conf_p, table_p = sweep_confidence(
        best_primary,
        marks,
        img_root,
        confs,
        imgsz=args.imgsz,
        device=device,
        max_val_samples=val_max,
    )
    print("Primary conf sweep:", json.dumps({str(k): round(v, 4) for k, v in sorted(table_p.items())}, ensure_ascii=False))
    print(f"PRIMARY best sequence accuracy: {acc_p:.4f} at conf={conf_p}")

    best_weights = best_primary
    best_conf = conf_p
    best_acc = acc_p
    upgrade_run = None
    best_upgrade = None
    mega_run = None
    best_mega = None

    if not args.no_upgrade and best_acc < args.min_acc:
        up_path = _try_resolve_model_path(args.upgrade_model)
        if not up_path:
            print(f"\n=== Stage 3 skipped: upgrade weights not found ({args.upgrade_model}) ===\n")
        elif Path(up_path).resolve() == Path(primary_path).resolve():
            print("Upgrade model same as primary; skip.")
        else:
            stem_u = Path(up_path).stem.replace(".", "_")
            run_u = f"{stem_u}_{stamp}_upgrade"
            print(f"\n=== Stage 3: primary acc {best_acc:.4f} < {args.min_acc}, train larger model ===\n")
            best_upgrade = run_train(
                model_path=up_path,
                run_name=run_u,
                epochs=args.epochs,
                imgsz=args.imgsz,
                batch=max(1, args.batch // 2) if args.batch > 0 else args.batch,
                device=args.device,
                workers=args.workers,
                fraction=args.fraction,
                project=args.project,
            )
            upgrade_run = run_u
            print("\n=== Stage 4: confidence sweep (upgrade) ===\n")
            acc_u, conf_u, table_u = sweep_confidence(
                best_upgrade,
                marks,
                img_root,
                confs,
                imgsz=args.imgsz,
                device=device,
                max_val_samples=val_max,
            )
            print("Upgrade conf sweep:", json.dumps({str(k): round(v, 4) for k, v in sorted(table_u.items())}, ensure_ascii=False))
            print(f"UPGRADE best sequence accuracy: {acc_u:.4f} at conf={conf_u}")
            if acc_u > best_acc:
                best_acc, best_conf, best_weights = acc_u, conf_u, best_upgrade
                print("Chose UPGRADE weights as global best.")
            else:
                print("Chose PRIMARY weights as global best.")

    mega_spec = (args.mega_upgrade_model or "").strip()
    if (
        (not args.no_upgrade)
        and (not args.no_mega_upgrade)
        and mega_spec
        and best_acc < args.min_acc
    ):
        mega_path = _try_resolve_model_path(mega_spec)
        if not mega_path:
            print(f"\n=== Mega stage skipped: weights not found ({mega_spec}) ===\n")
        elif Path(mega_path).resolve() == Path(best_weights).resolve():
            print("\n=== Mega stage skipped: same checkpoint as current best ===\n")
        else:
            stem_m = Path(mega_path).stem.replace(".", "_")
            run_m = f"{stem_m}_{stamp}_mega"
            print(f"\n=== Stage 5: best acc {best_acc:.4f} still < {args.min_acc}, train mega model ===\n")
            mega_batch = args.batch
            if mega_batch > 0:
                mega_batch = max(1, mega_batch // 4)
            best_mega = run_train(
                model_path=mega_path,
                run_name=run_m,
                epochs=args.epochs,
                imgsz=args.imgsz,
                batch=mega_batch,
                device=args.device,
                workers=args.workers,
                fraction=args.fraction,
                project=args.project,
            )
            mega_run = run_m
            print("\n=== Stage 6: confidence sweep (mega) ===\n")
            acc_m, conf_m, table_m = sweep_confidence(
                best_mega,
                marks,
                img_root,
                confs,
                imgsz=args.imgsz,
                device=device,
                max_val_samples=val_max,
            )
            print("Mega conf sweep:", json.dumps({str(k): round(v, 4) for k, v in sorted(table_m.items())}, ensure_ascii=False))
            print(f"MEGA best sequence accuracy: {acc_m:.4f} at conf={conf_m}")
            if acc_m > best_acc:
                best_acc, best_conf, best_weights = acc_m, conf_m, best_mega
                print("Chose MEGA weights as global best.")
            else:
                print("Keeping previous global best.")

    out = {
        "best_weights": str(best_weights),
        "best_conf": best_conf,
        "best_sequence_accuracy": round(best_acc, 6),
        "primary_run": run_primary,
        "primary_weights": str(best_primary),
        "resume_train": str(Path(args.resume_train).resolve()) if args.resume_train else None,
        "primary_best_conf": conf_p,
        "primary_best_acc": round(acc_p, 6),
        "upgrade_run": upgrade_run,
        "upgrade_weights": str(best_upgrade) if best_upgrade else None,
        "mega_run": mega_run,
        "mega_weights": str(best_mega) if best_mega else None,
        "min_acc_threshold": args.min_acc,
        "device": str(device),
        "val_max_samples_for_sweep": val_max,
    }
    out_path = ROOT / "runs" / "yolo_pipeline_summary.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    print("\n=== FINAL ===")
    print(json.dumps(out, indent=2, ensure_ascii=False))
    print(f"\nSummary written to: {out_path}")


if __name__ == "__main__":
    main()
