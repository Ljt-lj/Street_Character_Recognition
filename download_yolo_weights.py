"""Download Ultralytics YOLO11 detection checkpoints into this project folder."""

from __future__ import annotations

import argparse
import http.client
import os
import shutil
import sys
import tempfile
import time
import urllib.error
import urllib.request

ROOT = os.path.dirname(os.path.abspath(__file__))

WEIGHTS = ["yolo11n.pt", "yolo11s.pt", "yolo11m.pt", "yolo11l.pt", "yolo11x.pt"]

# Sizes from GitHub release v8.4.0 assets (minus small margin).
MIN_BYTES = {
    "yolo11n.pt": 5_400_000,
    "yolo11s.pt": 18_500_000,
    "yolo11m.pt": 39_000_000,
    "yolo11l.pt": 49_000_000,
    "yolo11x.pt": 110_000_000,
}

CHUNK = 1024 * 1024
ATTEMPTS = 4


def mirror_urls(filename: str) -> list[str]:
    """Try HF first (often more stable than GitHub raw release from CN networks)."""
    return [
        f"https://huggingface.co/Ultralytics/YOLO11/resolve/main/{filename}",
        f"https://hf-mirror.com/Ultralytics/YOLO11/resolve/main/{filename}",
        f"https://github.com/ultralytics/assets/releases/download/v8.4.0/{filename}",
        f"https://ghfast.top/https://github.com/ultralytics/assets/releases/download/v8.4.0/{filename}",
        f"https://mirror.ghproxy.com/https://github.com/ultralytics/assets/releases/download/v8.4.0/{filename}",
    ]


def _finalize_tmp_to_dest(tmp_path: str, dest: str) -> None:
    """Move completed download into place; handle Windows locks / antivirus."""
    for attempt in range(12):
        try:
            if os.path.isfile(dest):
                os.remove(dest)
        except OSError:
            pass
        try:
            shutil.move(tmp_path, dest)
            return
        except OSError as e:
            if attempt == 11:
                alt = dest + ".new"
                shutil.move(tmp_path, alt)
                raise OSError(
                    f"could not write {dest} ({e!r}); saved complete file as {alt} — "
                    f"close programs using the old file, then rename {os.path.basename(alt)}."
                ) from e
            time.sleep(1.25)


def stream_save(url: str, dest: str) -> int:
    req = urllib.request.Request(
        url,
        headers={"User-Agent": "Mozilla/5.0 (compatible; Street_Character_Recognition)"},
    )
    fd, tmp_path = tempfile.mkstemp(prefix="yolo_dl_", suffix=".pt")
    os.close(fd)
    total = 0
    try:
        with urllib.request.urlopen(req, timeout=900) as resp:
            with open(tmp_path, "wb") as f:
                while True:
                    try:
                        chunk = resp.read(CHUNK)
                    except http.client.IncompleteRead as e:
                        raise ValueError(f"IncompleteRead ({len(e.partial)} partial)") from e
                    if not chunk:
                        break
                    f.write(chunk)
                    total += len(chunk)
        _finalize_tmp_to_dest(tmp_path, dest)
        tmp_path = ""
        return total
    finally:
        if tmp_path and os.path.isfile(tmp_path):
            try:
                os.remove(tmp_path)
            except OSError:
                pass


def download_one(name: str, *, force: bool = False) -> bool:
    dest = os.path.join(ROOT, name)
    need = MIN_BYTES[name]

    if force and os.path.isfile(dest):
        print(f"[force] removing existing {name}")
        try:
            os.remove(dest)
        except OSError:
            bak = dest + ".bak"
            try:
                if os.path.isfile(bak):
                    os.remove(bak)
                os.replace(dest, bak)
                print(f"[force] moved old file to {os.path.basename(bak)}")
            except OSError as e:
                print(f"[warn] could not remove/rename locked {dest}: {e}")

    if os.path.isfile(dest) and os.path.getsize(dest) >= need:
        mb = os.path.getsize(dest) // 1024 // 1024
        print(f"[skip] {name} OK ({mb} MB)")
        return True

    if os.path.isfile(dest):
        print(f"[redo] {name} too small ({os.path.getsize(dest)} bytes), re-downloading")
        try:
            os.remove(dest)
        except OSError:
            pass

    last_err: Exception | None = None
    for url in mirror_urls(name):
        for attempt in range(1, ATTEMPTS + 1):
            print(f"[try] {name} attempt {attempt}/{ATTEMPTS}\n      {url}")
            try:
                n = stream_save(url, dest)
                if n < need:
                    raise ValueError(f"only {n} bytes (need >= {need})")
                mb = n // 1024 // 1024
                print(f"[ok]  {name} ({mb} MB)")
                return True
            except (urllib.error.URLError, TimeoutError, ValueError, OSError, ConnectionResetError) as e:
                last_err = e
                print(f"[fail] {e}")
                for p in (dest + ".part", dest + ".new"):
                    if os.path.isfile(p):
                        try:
                            os.remove(p)
                        except OSError:
                            pass

    print(f"[error] all mirrors failed for {name}: {last_err}")
    print(
        f"        Manual: save into project folder from browser —\n"
        f"        https://huggingface.co/Ultralytics/YOLO11/resolve/main/{name}\n"
        f"        or https://github.com/ultralytics/assets/releases/download/v8.4.0/{name}"
    )
    return False


def main() -> None:
    parser = argparse.ArgumentParser(description="Download YOLO11 .pt weights into this folder.")
    parser.add_argument(
        "weights",
        nargs="*",
        metavar="FILE.pt",
        help="Which files to fetch (default: all standard detection weights).",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Delete existing file(s) and download again.",
    )
    args = parser.parse_args()

    todo = args.weights if args.weights else WEIGHTS
    unknown = [w for w in todo if w not in MIN_BYTES]
    if unknown:
        raise SystemExit(f"Unknown weight(s): {unknown}. Known: {list(MIN_BYTES)}")

    ok = True
    for w in todo:
        ok = download_one(w, force=args.force) and ok
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
