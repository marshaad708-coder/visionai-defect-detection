# VisionAI: AI-Powered Defect Detection System for Smart Manufacturing

A research-grade computer vision project that detects and localizes manufacturing
defects in product images — built with free and open-source tools only, runnable
on a CPU-only Windows PC.

![Python](https://img.shields.io/badge/Python-3.10%E2%80%933.12-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-CPU-orange)
![License](https://img.shields.io/badge/License-MIT-green)

## Overview

VisionAI has two complementary tracks, the way real industrial inspection works:

| | **Track A** — Supervised classification | **Track B** — Unsupervised anomaly detection |
|---|---|---|
| Dataset | NEU steel surface defects (1,800 images, 6 classes) | MVTec AD (15 industrial categories) |
| Model | Compact CNN from scratch (+ optional ResNet-18 baseline) | PatchCore (Roth et al., CVPR 2022) |
| Learns from | Labeled defect examples | **Normal** images only |
| Output | Defect class + Grad-CAM heatmap | Anomaly score + pixel-level heatmap |
| Use when | Defect types are known and labeled | Defects are rare / unseen / unlabeled |

```
                    ┌─────────────────────────────────────────┐
                    │           VisionAI Dashboard            │
                    │   (Flask: upload → prediction → heatmap) │
                    └───────────┬───────────────┬─────────────┘
                                │               │
              ┌─────────────────▼───┐   ┌───────▼──────────────────┐
              │  TRACK A            │   │  TRACK B                 │
              │  Compact CNN        │   │  PatchCore               │
              │  6-class defect     │   │  anomaly score +         │
              │  classification +   │   │  pixel heatmap           │
              │  Grad-CAM           │   │                          │
              └─────────────────────┘   └──────────────────────────┘
```

## Features

- Real trained models — no mock predictions anywhere; every score is a model forward pass
- Grad-CAM visual explanations for every Track A prediction
- Honest reporting: softmax outputs and anomaly scores are labeled as **model scores,
  not calibrated probabilities** (see below)
- Reproducible: fixed seeds, saved data splits, configs, per-epoch logs, locked dependency versions
- Clean Flask dashboard: upload, input preview, prediction, heatmap, metrics page
- Full test-set evaluation: accuracy, precision, recall, F1, AUROC, confusion matrix (Track A);
  image- and pixel-level AUROC vs ground-truth masks (Track B)

## Project structure

```
visionai-defect-detection/
├── README.md  requirements.txt  requirements-lock.txt  LICENSE (MIT)
├── configs/                  # neu_compactcnn.yaml, neu_resnet18.yaml, mvtec_patchcore.yaml
├── data/
│   ├── raw/  (gitignored)    # datasets live here after download
│   └── DATASET_STATUS.md     # provenance + verification log
├── scripts/
│   ├── download_neu.py       # official-first NEU downloader + verifier
│   └── download_mvtec.py     # MVTec AD category downloader
├── src/
│   ├── common/               # seeding, config loading
│   ├── data_neu.py           # NEU loader, stratified splits
│   ├── data_mvtec.py         # MVTec AD loader (train/test/masks)
│   ├── models.py             # CompactCNN, ResNet-18
│   ├── anomaly/patchcore.py  # PatchCore implementation
│   ├── train_classifier.py   # Track A training (early stopping, CSV log)
│   ├── train_patchcore.py    # Track B memory-bank fitting
│   ├── evaluate.py           # Track A test metrics + confusion matrix
│   ├── evaluate_patchcore.py # Track B image/pixel metrics vs masks
│   ├── infer.py              # single-image inference + Grad-CAM
│   └── gradcam.py            # Grad-CAM (Selvaraju et al., ICCV 2017)
├── app/                      # Flask dashboard (templates + static)
├── tests/test_smoke.py       # zero-dependency smoke tests
├── results/                  # metrics JSON, plots, logs (generated)
└── checkpoints/ (gitignored) # trained weights (generated)
```

Training, evaluation, inference, and the web app are separate modules.

## Installation (Windows, PowerShell)

Requirements: Windows 10/11, Python 3.11 or 3.12 (from
[python.org](https://www.python.org/downloads/); tick **"Add python.exe to PATH"**),
~4 GB free disk. No GPU needed. 8 GB RAM is enough; 16–32 GB is comfortable.

```powershell
# 1. Open PowerShell in the project folder and create a virtual environment
python -m venv .venv
.\.venv\Scripts\Activate.ps1
# If Windows blocks the script, run once:
#   Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser

# 2. Upgrade pip
python -m pip install --upgrade pip

# 3. Install PyTorch (CPU-only build: smaller download, no NVIDIA GPU needed)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

# 4. Install the remaining dependencies
pip install -r requirements.txt

# 5. Verify — must print versions with no errors
python -c "import torch, torchvision, cv2, flask, sklearn; print('torch', torch.__version__)"
```

Exact tested versions are pinned in `requirements-lock.txt`
(e.g. `torch==2.14.1+cpu`, `torchvision==0.29.1+cpu`).

## Datasets

### NEU surface defect database (Track A)
- Official source: <http://faculty.neu.edu.cn/songkc/en/zdylm/263265/list/index.htm>
  (Northeastern University, China) — free for research use
- Citation: K. Song, Y. Yan, "A noise robust method based on completed local binary
  patterns for hot-rolled steel strip surface defects," *Applied Surface Science*,
  vol. 285, pp. 858–864, 2013.
- 1,800 grayscale images (200×200), 6 defect classes × 300:
  `crazing`, `inclusion`, `patches`, `pitted_surface`, `rolled-in_scale`, `scratches`

```powershell
python scripts/download_neu.py            # tries official source, verifies 6x300
python scripts/download_neu.py --local "C:\path\to\manual\download"   # verify your own copy
```

### MVTec AD (Track B)
- Official source: <https://www.mvtec.com/company/research/datasets/mvtec-ad>
- License: **CC BY-NC-SA 4.0** (research/non-commercial, share-alike)
- Citation: P. Bergmann, M. Fauser, D. Sattlegger, C. Steger,
  "MVTec AD — A Comprehensive Real-World Dataset for Unsupervised Anomaly Detection,"
  *Proc. CVPR*, 2019.

```powershell
python scripts/download_mvtec.py --category bottle
```

See `data/DATASET_STATUS.md` for the provenance and verification log of every
dataset copy used.

## Usage

All commands run from the project root with the virtual environment activated.

```powershell
# ---- Track A: train the defect classifier (CPU; ~30-60 min on a modern PC) ----
python src/train_classifier.py --config configs/neu_compactcnn.yaml

# Optional ResNet-18 transfer-learning baseline (same data splits, comparable numbers)
python src/train_classifier.py --config configs/neu_resnet18.yaml

# ---- Track A: evaluate on the held-out test split ----
python src/evaluate.py --config configs/neu_compactcnn.yaml
# writes results/neu_results.json and results/neu_results_confusion_matrix.png

# ---- Single-image inference with Grad-CAM overlay ----
python src/infer.py --checkpoint checkpoints/neu_compactcnn_best.pt --image path\to\steel.jpg --out overlay.png

# ---- Web dashboard ----
python app/app.py
# open http://127.0.0.1:5000  →  Inspect | Metrics | About

# ---- Track B: fit PatchCore on normal images, then evaluate vs ground-truth masks ----
python src/train_patchcore.py --config configs/mvtec_patchcore.yaml
python src/evaluate_patchcore.py --config configs/mvtec_patchcore.yaml

# ---- Smoke tests (no extra test framework needed) ----
python tests/test_smoke.py
```

## Results

### Track A (NEU-CLS) — measured 2026-10-21

Reference run: CompactCNN trained from scratch (seed 42, stratified
1260/270/270 split, AdamW + cosine schedule, early stopping on val macro-F1,
best @ epoch 14), CPU-only. Metrics computed on the **held-out test set**
(270 images) — never on training data.

| Metric | Value |
|---|---|
| Test accuracy | 0.989 |
| Macro precision / recall / F1 | 0.989 / 0.989 / 0.989 |
| AUROC (one-vs-rest) | 0.999 |

Per-class (P/R/F1/AUROC): crazing 1.000/1.000/1.000/1.000,
inclusion 0.957/1.000/0.978/0.998, patches 1.000/0.978/0.989/1.000,
pitted_surface 1.000/0.978/0.989/1.000, rolled-in_scale 1.000/1.000/1.000/1.000,
scratches 0.978/0.978/0.978/0.998.

Per-class precision/recall/F1/AUROC and the confusion matrix are in
`results/neu_results.json` and `results/neu_results_confusion_matrix.png`,
and are displayed on the dashboard's Metrics page.

### Track B (MVTec AD, toothbrush) — measured 2026-10-21

Reference run: PatchCore (WideResNet-50 layer2+layer3 features, 768-embedding
coreset bank fitted on 60 normal images, operating threshold 2.05 from the
99.5th percentile of train scores). Evaluated on 42 test images (30 anomalous)
against the ground-truth masks.

| Metric | Value |
|---|---|
| Image-level AUROC | 0.917 |
| Pixel-level AUROC (vs ground-truth masks) | 0.927 |
| Precision / recall / F1 @ operating threshold | 1.000 / 0.667 / 0.800 |

Full details: `results/mvtec_toothbrush_results.json`; example heatmap-vs-mask
overlays: `results/mvtec_toothbrush_maps/`.

## Scores are not calibrated probabilities

- Track A "confidence" bars are **raw softmax outputs**. They rank classes but
  are typically overconfident and are *not* calibrated probabilities.
- Track B anomaly scores are **distances in feature space**; the operating
  threshold is the 99.5th percentile of normal training scores — an operating
  point, not a probability.
- Proper calibration (temperature scaling, conformal prediction) is listed as
  future work. The dashboard and docs never present these numbers as true
  probabilities.

## Reproducibility

- `seed: 42` in every config; `src/common/seed.py` fixes Python, NumPy, and
  PyTorch RNGs and enables deterministic algorithms.
- `results/neu_splits.json` stores the exact train/val/test indices; evaluation
  rebuilds the test set from this file, so train and eval can never disagree.
- `results/neu_training_log.csv` records every epoch (loss, accuracy, macro-F1, LR).
- `requirements-lock.txt` pins every tested dependency version.

## Roadmap

- NEU-DET bounding-box detection track (Faster R-CNN / YOLO) reusing the 1,800 XML annotations
- Score calibration (temperature scaling) and conformal thresholds
- ONNX export for deployment without a Python runtime
- Multi-category MVTec AD sweep with a summary table

## License

Code: MIT (see `LICENSE`). Datasets are **not** covered by this license —
NEU is free for research use per its lab page; MVTec AD is CC BY-NC-SA 4.0.
