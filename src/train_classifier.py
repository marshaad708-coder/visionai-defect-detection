#!/usr/bin/env python3
"""Train the VisionAI Track A defect classifier (NEU-CLS).

Usage:
    python src/train_classifier.py --config configs/neu_compactcnn.yaml
    python src/train_classifier.py --config configs/neu_resnet18.yaml --batch-size 16

What it does:
  1. Loads config, sets all RNG seeds (reproducible).
  2. Builds stratified train/val/test splits (saved to results/*.json).
  3. Trains with AdamW + cosine LR schedule, monitoring validation macro-F1.
  4. Early-stops and keeps the best checkpoint (model + config + class names).

Outputs:
  checkpoints/<name>.pt      best model (state_dict, config, class names, metrics)
  results/<name>_log.csv     per-epoch train/val metrics
"""
import argparse
import csv
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
from sklearn.metrics import f1_score

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from src.common.config import load_config  # noqa: E402
from src.common.seed import set_seed  # noqa: E402
from src.data_neu import make_dataloaders  # noqa: E402
from src.models import count_parameters, get_model  # noqa: E402


def parse_args():
    ap = argparse.ArgumentParser(description="Train VisionAI Track A classifier.")
    ap.add_argument("--config", default="configs/neu_compactcnn.yaml")
    ap.add_argument("--batch-size", type=int, default=None,
                    help="Override config batch size (use 16 on low-RAM machines).")
    ap.add_argument("--epochs", type=int, default=None, help="Override config epochs.")
    ap.add_argument("--device", default=None, help="cpu | cuda (default: auto)")
    ap.add_argument("--resume", action="store_true",
                    help="Resume from the existing best checkpoint "
                         "(restores model, optimizer, scheduler, epoch).")
    return ap.parse_args()


def run_epoch(model, loader, criterion, optimizer, device, train: bool):
    model.train(train)
    total_loss, correct, n = 0.0, 0, 0
    all_pred, all_true = [], []
    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)
        if train:
            optimizer.zero_grad()
        with torch.set_grad_enabled(train):
            logits = model(xb)
            loss = criterion(logits, yb)
            if train:
                loss.backward()
                optimizer.step()
        total_loss += loss.item() * len(xb)
        pred = logits.argmax(1)
        correct += (pred == yb).sum().item()
        n += len(xb)
        all_pred.extend(pred.cpu().tolist())
        all_true.extend(yb.cpu().tolist())
    macro_f1 = f1_score(all_true, all_pred, average="macro", zero_division=0)
    return total_loss / n, correct / n, macro_f1


def main() -> int:
    args = parse_args()
    cfg = load_config(REPO / args.config)
    if args.batch_size:
        cfg["data"]["batch_size"] = args.batch_size
    if args.epochs:
        cfg["train"]["epochs"] = args.epochs
    seed = cfg.get("seed", 42)
    set_seed(seed)

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    print(f"Device: {device} | seed: {seed}")

    loaders, info = make_dataloaders(cfg, seed=seed)
    print(f"Data: {info['n_train']} train / {info['n_val']} val / {info['n_test']} test")
    print(f"Classes: {info['class_names']}")

    mcfg = cfg["model"]
    model = get_model(mcfg["name"], num_classes=mcfg["num_classes"],
                      **{k: v for k, v in mcfg.items()
                         if k not in ("name", "num_classes")})
    print(f"Model: {mcfg['name']} ({count_parameters(model):,} trainable params)")
    model.to(device)

    tcfg = cfg["train"]
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=tcfg["lr"],
                                  weight_decay=tcfg.get("weight_decay", 1e-4))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=tcfg["epochs"])

    ckpt_path = REPO / cfg["paths"]["checkpoint"]
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)
    log_path = REPO / cfg["paths"]["log"]
    log_path.parent.mkdir(parents=True, exist_ok=True)

    best_f1, best_epoch = -1.0, -1
    patience = tcfg.get("early_stopping_patience", 8)
    patience_left = patience
    start_epoch = 1
    if args.resume and ckpt_path.exists():
        prev = torch.load(ckpt_path, map_location=device, weights_only=False)
        model.load_state_dict(prev["model_state"])
        if "optimizer_state" in prev:
            optimizer.load_state_dict(prev["optimizer_state"])
        if "scheduler_state" in prev:
            scheduler.load_state_dict(prev["scheduler_state"])
        best_f1 = float(prev.get("val_macro_f1", -1.0))
        best_epoch = int(prev.get("epoch", 0))
        start_epoch = best_epoch + 1
        print(f"Resumed from checkpoint epoch {best_epoch} "
              f"(val macro-F1 {best_f1:.4f}); continuing at epoch {start_epoch}.")

    log_exists = log_path.exists()
    with open(log_path, "a" if (args.resume and log_exists) else "w",
              newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if not (args.resume and log_exists):
            writer.writerow(["epoch", "train_loss", "train_acc", "val_loss",
                             "val_acc", "val_macro_f1", "lr", "seconds"])
        for epoch in range(start_epoch, tcfg["epochs"] + 1):
            t0 = time.time()
            tr_loss, tr_acc, _ = run_epoch(model, loaders["train"], criterion,
                                           optimizer, device, train=True)
            va_loss, va_acc, va_f1 = run_epoch(model, loaders["val"], criterion,
                                               optimizer, device, train=False)
            scheduler.step()
            lr = optimizer.param_groups[0]["lr"]
            secs = time.time() - t0
            writer.writerow([epoch, f"{tr_loss:.4f}", f"{tr_acc:.4f}", f"{va_loss:.4f}",
                             f"{va_acc:.4f}", f"{va_f1:.4f}", f"{lr:.2e}", f"{secs:.1f}"])
            f.flush()
            improved = va_f1 > best_f1 + 1e-4
            tag = "  <-- best" if improved else ""
            print(f"epoch {epoch:3d}/{tcfg['epochs']}  "
                  f"train loss {tr_loss:.4f} acc {tr_acc:.4f} | "
                  f"val loss {va_loss:.4f} acc {va_acc:.4f} macro-F1 {va_f1:.4f}{tag}")
            if improved:
                best_f1, best_epoch, patience_left = va_f1, epoch, patience
                torch.save({
                    "model_state": model.state_dict(),
                    "optimizer_state": optimizer.state_dict(),
                    "scheduler_state": scheduler.state_dict(),
                    "config": cfg,
                    "class_names": info["class_names"],
                    "epoch": epoch,
                    "val_macro_f1": va_f1,
                    "val_acc": va_acc,
                }, ckpt_path)
            else:
                patience_left -= 1
                if patience_left <= 0:
                    print(f"Early stopping at epoch {epoch} "
                          f"(best val macro-F1 {best_f1:.4f} @ epoch {best_epoch}).")
                    break
    print(f"Best checkpoint: {ckpt_path}  (val macro-F1 {best_f1:.4f} @ epoch {best_epoch})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
