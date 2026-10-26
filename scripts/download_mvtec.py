#!/usr/bin/env python3
"""Download one MVTec AD category for VisionAI Track B.

Official source: https://www.mvtec.com/company/research/datasets/mvtec-ad
  The official download requires a request form (name / email / affiliation);
  there is no unauthenticated direct link. If you have official access, place
  the category folder at data/raw/mvtec/<category>/ and skip this script.

Default mirror (explicit, verified for this build):
  Hugging Face `foersben/mvtec-ad` — community mirror of the official release,
  same CC BY-NC-SA 4.0 license, original folder layout preserved.
  Verified 2026-10-21: toothbrush = 60 train/good, 12 test/good,
  30 test/defective + 30 ground-truth masks, all 1024x1024.

Usage:
    python scripts/download_mvtec.py --category toothbrush
    python scripts/download_mvtec.py --category bottle --mirror hf-mirror

Result layout:  data/raw/mvtec/<category>/{train,test,ground_truth}/...
"""
import argparse
import shutil
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
DEST = REPO / "data" / "raw" / "mvtec"

HF_MIRROR = "foersben/mvtec-ad"  # huggingface.co/datasets/foersben/mvtec-ad


def download_hf(category: str) -> Path:
    """Fetch a category from the Hugging Face mirror via plain HTTPS."""
    try:
        from huggingface_hub import HfApi
    except ImportError:
        print("This downloader needs the 'huggingface_hub' package:")
        print("  pip install huggingface_hub")
        return None
    api = HfApi()
    files = [f.rfilename for f in api.list_repo_files(HF_MIRROR, repo_type="dataset")
             if f.rfilename.startswith(category + "/")]
    if not files:
        print(f"Category '{category}' not found in {HF_MIRROR}.")
        print("Available categories: bottle, cable, capsule, carpet, grid, hazelnut,")
        print("  leather, metal_nut, pill, screw, tile, toothbrush, transistor, wood, zipper")
        return None
    tmp = Path(tempfile.mkdtemp(prefix="mvtec_dl_"))
    base = f"https://huggingface.co/datasets/{HF_MIRROR}/resolve/main"
    import requests
    for i, rf in enumerate(sorted(files), 1):
        rel = Path(rf).relative_to(category)
        out = tmp / rel
        out.parent.mkdir(parents=True, exist_ok=True)
        url = f"{base}/{rf}"
        print(f"  [{i}/{len(files)}] {rel}")
        with requests.get(url, stream=True, timeout=120) as r:
            r.raise_for_status()
            with open(out, "wb") as f:
                for chunk in r.iter_content(chunk_size=1 << 20):
                    if chunk:
                        f.write(chunk)
    return tmp


def verify(category: str) -> bool:
    cat = DEST / category
    checks = {
        "train/good": 1,   # at least 1; exact counts vary by category
        "test/good": 1,
    }
    ok = True
    for sub, minimum in checks.items():
        d = cat / sub
        n = len(list(d.glob("*.png"))) if d.is_dir() else 0
        good = n >= minimum
        ok &= good
        print(f"  [{'OK ' if good else 'FAIL'}] {sub}: {n} images")
    # mask pairing for every defective test image
    for defect_dir in sorted((cat / "test").glob("*")):
        if not defect_dir.is_dir() or defect_dir.name == "good":
            continue
        gt_dir = cat / "ground_truth" / defect_dir.name
        paired, total = 0, 0
        for p in sorted(defect_dir.glob("*.png")):
            total += 1
            if (gt_dir / (p.stem + ".png")).exists() or \
               (gt_dir / (p.stem + "_mask.png")).exists():
                paired += 1
        good = total > 0 and paired == total
        ok &= good
        print(f"  [{'OK ' if good else 'FAIL'}] test/{defect_dir.name}: "
              f"{total} images, {paired}/{total} masks paired")
    return ok


def main() -> int:
    ap = argparse.ArgumentParser(description="Download an MVTec AD category.")
    ap.add_argument("--category", default="toothbrush")
    ap.add_argument("--mirror", default="hf-mirror", choices=["hf-mirror"],
                    help="Download mirror (explicit opt-in).")
    args = ap.parse_args()

    dest = DEST / args.category
    if dest.exists() and any(dest.iterdir()):
        print(f"Already present: {dest} — verifying instead.")
        return 0 if verify(args.category) else 1

    print(f"Downloading MVTec AD '{args.category}' via {args.mirror} ...")
    tmp = download_hf(args.category)
    if tmp is None:
        return 1
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        shutil.rmtree(dest)
    shutil.move(str(tmp), str(dest))
    print(f"Saved to {dest}\nVerifying...")
    return 0 if verify(args.category) else 1


if __name__ == "__main__":
    sys.exit(main())
