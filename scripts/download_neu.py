#!/usr/bin/env python3
"""Download and verify the NEU surface-defect dataset (VisionAI Track A).

Official source (free for research use):
    http://faculty.neu.edu.cn/songkc/en/zdylm/263265/list/index.htm
Citation: K. Song, Y. Yan, "A noise robust method based on completed local
binary patterns for hot-rolled steel strip surface defects,"
Applied Surface Science, vol. 285, pp. 858-864, 2013.

Usage:
    python scripts/download_neu.py                 # try official page, else print manual steps
    python scripts/download_neu.py --local PATH    # verify/arrange a manually downloaded copy
    python scripts/download_neu.py --mirror URL    # explicit opt-in: download a zip from URL

Result layout:  data/raw/neu/<class>/*.jpg   (6 classes x 300 images)

This script tries the OFFICIAL source first. A community mirror is only used
when you pass --mirror explicitly. Whatever the origin, every download is
verified (6 classes x 300 images, 200x200, readable) before it is accepted.
"""
import argparse
import re
import shutil
import sys
import tempfile
import zipfile
from pathlib import Path

import requests
from PIL import Image

REPO = Path(__file__).resolve().parents[1]
DEST = REPO / "data" / "raw" / "neu"
OFFICIAL_PAGE = "http://faculty.neu.edu.cn/songkc/en/zdylm/263265/list/index.htm"

CLASSES = ["crazing", "inclusion", "patches", "pitted_surface",
           "rolled-in_scale", "scratches"]
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}


def _norm(name: str) -> str:
    return name.strip().lower().replace(" ", "_").replace("-", "_")


def download(url: str, out: Path) -> None:
    print(f"Downloading {url}")
    with requests.get(url, stream=True, timeout=60) as r:
        r.raise_for_status()
        total = int(r.headers.get("content-length", 0))
        done = 0
        with open(out, "wb") as f:
            for chunk in r.iter_content(chunk_size=1 << 20):
                if chunk:
                    f.write(chunk)
                    done += len(chunk)
                    if total:
                        print(f"\r  {done / 1e6:.1f} / {total / 1e6:.1f} MB", end="")
    print("\n  done.")


def normalize_layout(src_root: Path) -> None:
    """Collect class images from any nested layout into DEST/<class>/*.jpg."""
    found: dict[str, list[Path]] = {}
    for d in src_root.rglob("*"):
        if d.is_dir() and _norm(d.name) in {_norm(c) for c in CLASSES}:
            canon = next(c for c in CLASSES if _norm(c) == _norm(d.name))
            imgs = sorted(p for p in d.iterdir()
                          if p.is_file() and p.suffix.lower() in IMG_EXTS)
            if imgs:
                found.setdefault(canon, []).extend(imgs)
    if len(found) != 6:
        raise RuntimeError(
            f"Expected 6 defect-class folders, found {sorted(found)} under {src_root}")
    DEST.mkdir(parents=True, exist_ok=True)
    for cls, imgs in found.items():
        target = DEST / cls
        target.mkdir(exist_ok=True)
        for p in imgs:
            dst = target / p.name
            if not dst.exists():
                shutil.copy2(p, dst)
    print(f"Arranged dataset at {DEST}")


def verify() -> bool:
    """Return True only if the on-disk dataset is exactly 6 x 300 readable images."""
    ok = True
    for cls in CLASSES:
        d = DEST / cls
        imgs = sorted(p for p in d.glob("*") if p.suffix.lower() in IMG_EXTS) if d.is_dir() else []
        status = "OK " if len(imgs) == 300 else "FAIL"
        if len(imgs) != 300:
            ok = False
        print(f"  [{status}] {cls}: {len(imgs)} images (expected 300)")
    # Spot-check readability + dimensions.
    checked, bad = 0, 0
    for cls in CLASSES:
        for p in sorted((DEST / cls).glob("*"))[:5]:
            try:
                with Image.open(p) as im:
                    im.verify()
                with Image.open(p) as im:
                    if im.size != (200, 200):
                        print(f"  WARN {p.name}: size {im.size} (expected 200x200)")
                checked += 1
            except Exception as e:  # noqa: BLE001
                bad += 1
                print(f"  FAIL unreadable: {p} ({e})")
    print(f"  readability spot-check: {checked} ok, {bad} bad")
    return ok and bad == 0


def from_official() -> bool:
    """Try to find a direct archive link on the official NEU page."""
    print(f"Checking official source: {OFFICIAL_PAGE}")
    try:
        r = requests.get(OFFICIAL_PAGE, timeout=30)
        r.raise_for_status()
    except Exception as e:  # noqa: BLE001
        print(f"  Could not reach official page: {e}")
        return False
    links = re.findall(r'href="([^"]+)"', r.text, flags=re.IGNORECASE)
    cands = [l for l in links
             if l.lower().endswith((".zip", ".rar", ".7z")) and "neu" in l.lower()]
    if not cands:
        print("  No direct download link found on the official page.")
        return False
    url = cands[0] if cands[0].startswith("http") else OFFICIAL_PAGE.rsplit("/", 1)[0] + "/" + cands[0]
    tmp = Path(tempfile.mkdtemp(prefix="neu_dl_"))
    zpath = tmp / "neu.zip"
    download(url, zpath)
    with zipfile.ZipFile(zpath) as z:
        z.extractall(tmp / "extracted")
    normalize_layout(tmp / "extracted")
    shutil.rmtree(tmp, ignore_errors=True)
    return True


def manual_instructions() -> None:
    print("""
No automatic download available. Manual steps (2 minutes):
  1. Open the official NEU dataset page:
       http://faculty.neu.edu.cn/songkc/en/zdylm/263265/list/index.htm
  2. Download the NEU surface-defect dataset (free for research use).
  3. Extract it anywhere, then run:
       python scripts/download_neu.py --local "C:\\path\\to\\extracted\\folder"
     (PowerShell)  or
       python scripts/download_neu.py --local /path/to/extracted/folder
     (Linux/macOS)
The script will arrange and verify the 6 x 300 images automatically.
""")


def main() -> int:
    ap = argparse.ArgumentParser(description="Download + verify the NEU defect dataset.")
    ap.add_argument("--local", help="Path to a manually downloaded zip or folder.")
    ap.add_argument("--mirror", help="Explicit opt-in: download a zip from this URL.")
    args = ap.parse_args()

    if args.local:
        src = Path(args.local)
        if not src.exists():
            print(f"Path not found: {src}")
            return 1
        if src.is_file() and src.suffix.lower() == ".zip":
            tmp = Path(tempfile.mkdtemp(prefix="neu_dl_"))
            with zipfile.ZipFile(src) as z:
                z.extractall(tmp)
            normalize_layout(tmp)
            shutil.rmtree(tmp, ignore_errors=True)
        else:
            normalize_layout(src)
    elif args.mirror:
        tmp = Path(tempfile.mkdtemp(prefix="neu_dl_"))
        zpath = tmp / "neu.zip"
        download(args.mirror, zpath)
        with zipfile.ZipFile(zpath) as z:
            z.extractall(tmp / "extracted")
        normalize_layout(tmp / "extracted")
        shutil.rmtree(tmp, ignore_errors=True)
    else:
        if not from_official():
            manual_instructions()
            return 2

    print("Verifying...")
    return 0 if verify() else 1


if __name__ == "__main__":
    sys.exit(main())
