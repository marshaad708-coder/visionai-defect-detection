"""MVTec AD dataset loading for Track B (unsupervised anomaly detection).

Expected layout for one category, e.g. data/raw/mvtec/bottle/:
    train/good/*.png
    test/good/*.png
    test/<defect_type>/*.png
    ground_truth/<defect_type>/*.png        (binary masks, same stem as image)

Reference: P. Bergmann et al., "MVTec AD — A Comprehensive Real-World Dataset
for Unsupervised Anomaly Detection", CVPR 2019.
"""
from pathlib import Path
from typing import List, Tuple

import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

_IMG_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}


def _list_images(d: Path) -> List[Path]:
    return sorted(p for p in d.iterdir()
                  if p.is_file() and p.suffix.lower() in _IMG_EXTS) if d.is_dir() else []


class MVTecDataset(Dataset):
    """split='train' -> normal images only (label 0, empty mask).
    split='test'  -> normal + anomalous images with pixel masks."""

    def __init__(self, root: str | Path, category: str, split: str = "train",
                 image_size: int = 224):
        self.cat_dir = Path(root) / category
        if not (self.cat_dir / "train" / "good").is_dir():
            raise FileNotFoundError(
                f"MVTec AD category not found: {self.cat_dir}\n"
                "Run: python scripts/download_mvtec.py --category "
                f"{category}  (see README.md)"
            )
        if split not in ("train", "test"):
            raise ValueError("split must be 'train' or 'test'")
        self.split = split
        self.samples: List[Tuple[Path, Path | None, int, str]] = []  # img, mask, label, defect

        if split == "train":
            for p in _list_images(self.cat_dir / "train" / "good"):
                self.samples.append((p, None, 0, "good"))
        else:
            test_dir = self.cat_dir / "test"
            for defect_dir in sorted(d for d in test_dir.iterdir() if d.is_dir()):
                defect = defect_dir.name
                gt_dir = self.cat_dir / "ground_truth" / defect
                for p in _list_images(defect_dir):
                    label = 0 if defect == "good" else 1
                    mask = None
                    if label == 1 and gt_dir.is_dir():
                        # Official MVTec AD: <stem>.png ; some mirrors: <stem>_mask.png
                        for cand in (gt_dir / (p.stem + ".png"),
                                     gt_dir / (p.stem + "_mask.png")):
                            if cand.exists():
                                mask = cand
                                break
                    self.samples.append((p, mask, label, defect))
        if not self.samples:
            raise FileNotFoundError(f"No images found for {category}/{split}")

        self.transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])
        self.mask_size = image_size

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, i: int):
        img_path, mask_path, label, defect = self.samples[i]
        img = Image.open(img_path).convert("RGB")
        img = self.transform(img)
        if mask_path is not None:
            mask = Image.open(mask_path).convert("L").resize(
                (self.mask_size, self.mask_size))
            mask = (np.array(mask) > 127).astype(np.float32)
        else:
            mask = np.zeros((self.mask_size, self.mask_size), dtype=np.float32)
        return {"image": img, "mask": mask, "label": label,
                "defect": defect, "path": str(img_path)}


def summarize_category(root: str | Path, category: str) -> dict:
    """Quick on-disk audit used by scripts and tests."""
    out = {"category": category, "train_good": 0, "test": {}}
    base = Path(root) / category
    out["train_good"] = len(_list_images(base / "train" / "good"))
    test_dir = base / "test"
    if test_dir.is_dir():
        for d in sorted(x for x in test_dir.iterdir() if x.is_dir()):
            out["test"][d.name] = len(_list_images(d))
    return out
