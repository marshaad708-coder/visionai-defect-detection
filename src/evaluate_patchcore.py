#!/usr/bin/env python3
"""Evaluate PatchCore on the MVTec AD test split (Track B).

Usage:
    python src/evaluate_patchcore.py --config configs/mvtec_patchcore.yaml [--save-maps N]

Compares predictions against the ground-truth masks:
  - image-level: AUROC, plus precision/recall/F1 at the saved operating threshold
  - pixel-level:  AUROC over all test pixels (normal vs defect)
Saves results/mvtec_<category>_results.json and N example overlay PNGs.
"""
import argparse
import json
import sys
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import (auc, f1_score, precision_recall_fscore_support,
                             roc_auc_score, roc_curve)
from torch.utils.data import DataLoader

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from src.anomaly import PatchCore  # noqa: E402
from src.common.config import load_config  # noqa: E402
from src.common.seed import set_seed  # noqa: E402
from src.data_mvtec import MVTecDataset  # noqa: E402


def parse_args():
    ap = argparse.ArgumentParser(description="Evaluate PatchCore on MVTec AD.")
    ap.add_argument("--config", default="configs/mvtec_patchcore.yaml")
    ap.add_argument("--save-maps", type=int, default=6,
                    help="Save N example heatmap-vs-mask overlays.")
    ap.add_argument("--device", default=None)
    return ap.parse_args()


def overlay(image_np: np.ndarray, heat: np.ndarray) -> np.ndarray:
    colored = cv2.applyColorMap((heat * 255).astype(np.uint8), cv2.COLORMAP_JET)
    colored = cv2.cvtColor(colored, cv2.COLOR_BGR2RGB)
    return (0.45 * colored + 0.55 * (image_np * 255)).astype(np.uint8)


def main() -> int:
    args = parse_args()
    cfg = load_config(REPO / args.config)
    set_seed(cfg.get("seed", 42))
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    bank_path = REPO / cfg["paths"]["bank"]
    if not bank_path.exists():
        print(f"Memory bank not found: {bank_path}\n"
              f"Fit it first: python src/train_patchcore.py --config {args.config}")
        return 1
    pc = PatchCore.load(bank_path, device=device)
    print(f"Loaded bank: {len(pc.bank)} embeddings | threshold {pc.threshold:.4f}")

    dcfg = cfg["data"]
    test_ds = MVTecDataset(dcfg["root"], dcfg["category"], split="test",
                           image_size=dcfg.get("image_size", 224))
    loader = DataLoader(test_ds, batch_size=1, shuffle=False, num_workers=0)

    y_true, y_score, defects = [], [], []
    pix_true, pix_score = [], []
    saved = 0
    out_dir = REPO / "results" / f"mvtec_{dcfg['category']}_maps"
    out_dir.mkdir(parents=True, exist_ok=True)

    for batch in loader:
        img = batch["image"][0]
        mask = batch["mask"][0].numpy()
        label = int(batch["label"][0])
        score, heat = pc.anomaly_map(img)
        y_true.append(label)
        y_score.append(score)
        defects.append(batch["defect"][0])
        pix_true.append(mask.ravel())
        pix_score.append(heat.ravel())
        if label == 1 and saved < args.save_maps:
            fig, ax = plt.subplots(1, 3, figsize=(12, 4))
            rgb = img.permute(1, 2, 0).numpy()
            rgb = (rgb - rgb.min()) / (rgb.max() - rgb.min() + 1e-8)
            ax[0].imshow(rgb); ax[0].set_title("input"); ax[0].axis("off")
            ax[1].imshow(mask, cmap="gray"); ax[1].set_title("ground truth"); ax[1].axis("off")
            ax[2].imshow(overlay(rgb, heat)); ax[2].set_title(f"heatmap (score {score:.3f})")
            ax[2].axis("off")
            fig.suptitle(f"{dcfg['category']} / {batch['defect'][0]}")
            fig.tight_layout()
            fig.savefig(out_dir / f"map_{saved:02d}.png", dpi=120)
            plt.close(fig)
            saved += 1

    y_true = np.array(y_true)
    y_score = np.array(y_score)
    img_auroc = float(roc_auc_score(y_true, y_score))
    pix_true = np.concatenate(pix_true)
    pix_score = np.concatenate(pix_score)
    pix_auroc = float(roc_auc_score(pix_true, pix_score)) if pix_true.max() > 0 else None

    thr = pc.threshold if pc.threshold is not None else float(np.median(y_score))
    y_pred = (y_score >= thr).astype(int)
    prec, rec, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="binary", zero_division=0)

    results = {
        "category": dcfg["category"],
        "bank": str(bank_path),
        "n_test": len(y_true),
        "n_anomalous": int(y_true.sum()),
        "image_auroc": img_auroc,
        "pixel_auroc": pix_auroc,
        "operating_threshold": float(thr),
        "at_threshold": {"precision": float(prec), "recall": float(rec), "f1": float(f1)},
        "note": ("Threshold is an operating point from train scores, not a calibrated "
                 "probability. Pixel AUROC is computed over all test pixels."),
    }
    res_path = REPO / cfg["paths"]["results"]
    res_path.parent.mkdir(parents=True, exist_ok=True)
    with open(res_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print(f"Test images: {len(y_true)} ({int(y_true.sum())} anomalous)")
    print(f"Image AUROC: {img_auroc:.4f} | Pixel AUROC: {pix_auroc}")
    print(f"At threshold {thr:.4f}: P {prec:.4f} R {rec:.4f} F1 {f1:.4f}")
    print(f"Saved: {res_path}  (+{saved} overlays in {out_dir})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
