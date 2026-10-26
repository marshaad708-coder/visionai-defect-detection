#!/usr/bin/env python3
"""Train (fit) the PatchCore memory bank on NORMAL MVTec AD images (Track B).

Usage:
    python src/train_patchcore.py --config configs/mvtec_patchcore.yaml

Steps:
  1. Load train split (normal images only).
  2. Extract patch embeddings with a frozen WideResNet-50.
  3. Greedy k-center coreset -> compact memory bank.
  4. Calibrate an operating threshold from train image scores
     (99.5th percentile; an operating point, NOT a calibrated probability).
  5. Save the bank to checkpoints/patchcore_<category>.pt.

Outputs: checkpoints/patchcore_<category>.pt  (bank, grid, threshold, settings)
"""
import argparse
import sys
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from src.anomaly import PatchCore  # noqa: E402
from src.common.config import load_config  # noqa: E402
from src.common.seed import set_seed  # noqa: E402
from src.data_mvtec import MVTecDataset  # noqa: E402


def parse_args():
    ap = argparse.ArgumentParser(description="Fit PatchCore on MVTec AD normals.")
    ap.add_argument("--config", default="configs/mvtec_patchcore.yaml")
    ap.add_argument("--device", default=None)
    return ap.parse_args()


def main() -> int:
    args = parse_args()
    cfg = load_config(REPO / args.config)
    set_seed(cfg.get("seed", 42))
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    dcfg, mcg = cfg["data"], cfg["model"]
    train_ds = MVTecDataset(dcfg["root"], dcfg["category"], split="train",
                            image_size=dcfg.get("image_size", 224))
    train_loader = DataLoader(train_ds, batch_size=8, shuffle=False, num_workers=0)
    print(f"Train images (normal only): {len(train_ds)}")

    pc = PatchCore(backbone=mcg.get("backbone", "wide_resnet50_2"),
                   patch=3, knn_k=mcg.get("knn_k", 9), device=device)
    t0 = time.time()
    info = pc.fit(train_loader,
                  patches_per_image=mcfg_extra(cfg, "patches_per_image", 128),
                  coreset_ratio=mcg.get("coreset_ratio", 0.01),
                  coreset_max=mcg.get("coreset_max", 20000),
                  coreset_batch=mcfg_extra(cfg, "coreset_batch", 64),
                  seed=cfg.get("seed", 42))
    print(f"Bank built in {time.time() - t0:.1f}s: "
          f"{info['bank_size']} embeddings x {info['embed_dim']} "
          f"from {info['n_train_images']} images (grid {info['patch_grid']})")

    thr = pc.calibrate(train_loader,
                       quantile=mcfg_extra(cfg, "threshold_quantile", 0.995))
    print(f"Operating threshold (99.5th pct of train scores): {thr:.4f}")

    bank_path = REPO / cfg["paths"]["bank"]
    pc.save(bank_path)
    print(f"Saved: {bank_path}")
    return 0


def mcfg_extra(cfg, key, default):
    return cfg["model"].get(key, default)


if __name__ == "__main__":
    sys.exit(main())
