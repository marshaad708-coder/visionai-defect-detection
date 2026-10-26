#!/usr/bin/env python3
"""Evaluate a trained VisionAI Track A classifier on the held-out test split.

Usage:
    python src/evaluate.py --config configs/neu_compactcnn.yaml

Metrics (all computed on the test split, never on train/val):
  accuracy, per-class precision/recall/F1, macro/micro/weighted averages,
  AUROC one-vs-rest (overall + per class), confusion matrix.

Outputs:
  results/<name>_results.json   all metrics
  results/<name>_confusion_matrix.png
"""
import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # headless-safe
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import (accuracy_score, auc, confusion_matrix,
                             f1_score, precision_recall_fscore_support,
                             roc_auc_score, roc_curve)
from torch.utils.data import DataLoader

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from src.common.config import load_config  # noqa: E402
from src.common.seed import set_seed  # noqa: E402
from src.data_neu import NEUDataset, build_transforms, discover_samples  # noqa: E402
from src.models import get_model  # noqa: E402


def parse_args():
    ap = argparse.ArgumentParser(description="Evaluate VisionAI Track A classifier.")
    ap.add_argument("--config", default="configs/neu_compactcnn.yaml")
    ap.add_argument("--checkpoint", default=None, help="Override checkpoint path.")
    ap.add_argument("--device", default=None)
    return ap.parse_args()


def build_test_loader(cfg, class_names):
    dcfg = cfg["data"]
    paths, labels, _ = discover_samples(dcfg["root"])
    with open(REPO / cfg["paths"]["splits"], encoding="utf-8") as f:
        splits = json.load(f)
    assert splits["class_names"] == class_names, "checkpoint classes != split classes"
    test_idx = splits["test"]
    tf = build_transforms(dcfg["image_size"], dcfg["in_channels"],
                          dcfg.get("imagenet_norm", False), augment=False)
    ds = NEUDataset([paths[i] for i in test_idx], [labels[i] for i in test_idx], tf)
    return DataLoader(ds, batch_size=dcfg["batch_size"], shuffle=False,
                      num_workers=dcfg.get("num_workers", 0))


@torch.no_grad()
def predict_all(model, loader, device):
    model.eval()
    ys, ps, probas = [], [], []
    for xb, yb in loader:
        logits = model(xb.to(device))
        proba = torch.softmax(logits, dim=1)
        ys.extend(yb.tolist())
        ps.extend(proba.argmax(1).cpu().tolist())
        probas.extend(proba.cpu().tolist())
    return np.array(ys), np.array(ps), np.array(probas)


def plot_confusion_matrix(cm, classes, out_path):
    fig, ax = plt.subplots(figsize=(8, 6.5))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(classes)), classes, rotation=30, ha="right")
    ax.set_yticks(range(len(classes)), classes)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Confusion matrix (test set)")
    for i in range(len(classes)):
        for j in range(len(classes)):
            ax.text(j, i, cm[i, j], ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black")
    fig.colorbar(im, ax=ax, label="count")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main() -> int:
    args = parse_args()
    cfg = load_config(REPO / args.config)
    set_seed(cfg.get("seed", 42))
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))

    ckpt_path = Path(args.checkpoint or cfg["paths"]["checkpoint"])
    if not ckpt_path.exists():
        print(f"Checkpoint not found: {ckpt_path}\nTrain first: python src/train_classifier.py --config {args.config}")
        return 1
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    class_names = ckpt["class_names"]

    mcfg = ckpt["config"]["model"]
    model = get_model(mcfg["name"], num_classes=mcfg["num_classes"],
                      **{k: v for k, v in mcfg.items() if k not in ("name", "num_classes")})
    model.load_state_dict(ckpt["model_state"])
    model.to(device)

    loader = build_test_loader(ckpt["config"], class_names)
    y_true, y_pred, y_proba = predict_all(model, loader, device)
    n = len(y_true)

    acc = accuracy_score(y_true, y_pred)
    prec, rec, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=list(range(len(class_names))), zero_division=0)
    macro = {"precision": float(prec.mean()), "recall": float(rec.mean()),
             "f1": float(f1.mean())}
    micro_p, micro_r, micro_f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="micro", zero_division=0)
    weighted_p, weighted_r, weighted_f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="weighted", zero_division=0)

    # AUROC one-vs-rest from softmax outputs (model scores, not calibrated probabilities).
    auroc_ovr = float(roc_auc_score(y_true, y_proba, multi_class="ovr"))
    auroc_per_class = {}
    for i, c in enumerate(class_names):
        binary = (y_true == i).astype(int)
        fpr, tpr, _ = roc_curve(binary, y_proba[:, i])
        auroc_per_class[c] = float(auc(fpr, tpr))

    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(class_names))))

    results = {
        "config": str(args.config),
        "checkpoint": str(ckpt_path),
        "checkpoint_epoch": ckpt.get("epoch"),
        "n_test": n,
        "accuracy": float(acc),
        "per_class": {
            c: {"precision": float(prec[i]), "recall": float(rec[i]), "f1": float(f1[i]),
                "support": int((y_true == i).sum()), "auroc_ovr": auroc_per_class[c]}
            for i, c in enumerate(class_names)
        },
        "macro_avg": macro,
        "micro_avg": {"precision": float(micro_p), "recall": float(micro_r), "f1": float(micro_f1)},
        "weighted_avg": {"precision": float(weighted_p), "recall": float(weighted_r), "f1": float(weighted_f1)},
        "auroc_ovr": auroc_ovr,
        "confusion_matrix": cm.tolist(),
        "note": ("Softmax outputs and AUROC are computed from raw model scores. "
                 "Softmax values are NOT calibrated probabilities."),
    }

    res_path = REPO / cfg["paths"]["results"]
    res_path.parent.mkdir(parents=True, exist_ok=True)
    with open(res_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    cm_path = res_path.with_name(res_path.stem + "_confusion_matrix.png")
    plot_confusion_matrix(cm, class_names, cm_path)

    print(f"Test samples: {n}")
    print(f"Accuracy: {acc:.4f} | Macro P/R/F1: {macro['precision']:.4f}/{macro['recall']:.4f}/{macro['f1']:.4f}")
    print(f"AUROC (OvR): {auroc_ovr:.4f}")
    print("Per-class P/R/F1:")
    for i, c in enumerate(class_names):
        print(f"  {c:15s} P {prec[i]:.4f}  R {rec[i]:.4f}  F1 {f1[i]:.4f}  AUROC {auroc_per_class[c]:.4f}")
    print(f"Saved: {res_path}\nSaved: {cm_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
