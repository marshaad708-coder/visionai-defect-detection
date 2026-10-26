"""NEU surface-defect dataset loading with reproducible stratified splits.

Expected layout (either works):
    data/raw/neu/<class_name>/*.jpg            (flat, ImageFolder-style)
    data/raw/neu/<anything>/<class_name>/*.jpg (nested mirrors)

Class folders are matched case-insensitively against the 6 known NEU-CLS
classes; unknown folders are reported and ignored.
"""
import json
from pathlib import Path
from typing import Dict, List, Tuple

from PIL import Image
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

NEU_CLASSES = [
    "crazing",
    "inclusion",
    "patches",
    "pitted_surface",
    "rolled-in_scale",
    "scratches",
]

_IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def _norm(name: str) -> str:
    return name.strip().lower().replace(" ", "_").replace("-", "_")


_NORM_CLASSES = {_norm(c): c for c in NEU_CLASSES}


def discover_samples(root: str | Path) -> Tuple[List[Path], List[int], List[str]]:
    """Return (image_paths, labels, class_names) found under root."""
    root = Path(root)
    if not root.exists():
        raise FileNotFoundError(
            f"NEU dataset root not found: {root}\n"
            "Run: python scripts/download_neu.py  (see README.md)"
        )
    # Find every directory whose normalized name matches a known class.
    class_dirs: Dict[str, Path] = {}
    for d in root.rglob("*"):
        if d.is_dir() and _norm(d.name) in _NORM_CLASSES:
            class_dirs.setdefault(_NORM_CLASSES[_norm(d.name)], d)

    missing = [c for c in NEU_CLASSES if c not in class_dirs]
    if missing:
        raise FileNotFoundError(
            f"NEU class folders missing under {root}: {missing}. "
            f"Found: {sorted(class_dirs)}"
        )

    paths, labels = [], []
    for idx, cls in enumerate(NEU_CLASSES):
        imgs = sorted(
            p for p in class_dirs[cls].iterdir()
            if p.is_file() and p.suffix.lower() in _IMG_EXTS
        )
        if not imgs:
            raise FileNotFoundError(f"No images found in {class_dirs[cls]}")
        paths.extend(imgs)
        labels.extend([idx] * len(imgs))
    return paths, labels, NEU_CLASSES


class NEUDataset(Dataset):
    def __init__(self, paths: List[Path], labels: List[int], transform=None):
        assert len(paths) == len(labels)
        self.paths = paths
        self.labels = labels
        self.transform = transform

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, i: int):
        img = Image.open(self.paths[i]).convert("L")  # NEU is grayscale
        if self.transform:
            img = self.transform(img)
        return img, self.labels[i]


def build_transforms(image_size: int, in_channels: int, imagenet_norm: bool,
                     augment: bool, aug_cfg: dict | None = None):
    aug_cfg = aug_cfg or {}
    t = []
    if augment:
        t += [
            transforms.RandomRotation(aug_cfg.get("rotation_deg", 10)),
            transforms.RandomHorizontalFlip(p=0.5 if aug_cfg.get("hflip", True) else 0.0),
            transforms.RandomAffine(0, translate=(aug_cfg.get("translate", 0.05),) * 2),
        ]
    t += [transforms.Resize((image_size, image_size))]
    if in_channels == 3:
        t.append(transforms.Grayscale(num_output_channels=3))
    t.append(transforms.ToTensor())
    if imagenet_norm and in_channels == 3:
        t.append(transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                      std=[0.229, 0.224, 0.225]))
    else:
        t.append(transforms.Normalize(mean=[0.5] * in_channels,
                                      std=[0.5] * in_channels))
    return transforms.Compose(t)


def make_dataloaders(cfg: dict, seed: int = 42):
    """Stratified train/val/test split; saves split indices to cfg['paths']['splits']."""
    dcfg = cfg["data"]
    paths, labels, class_names = discover_samples(dcfg["root"])

    # Split off test, then split the remainder into train/val (stratified, fixed seed).
    idx = list(range(len(paths)))
    trainval_idx, test_idx = train_test_split(
        idx, test_size=dcfg["test_ratio"], random_state=seed, stratify=labels)
    val_ratio_adj = dcfg["val_ratio"] / (1.0 - dcfg["test_ratio"])
    train_idx, val_idx = train_test_split(
        trainval_idx, test_size=val_ratio_adj, random_state=seed,
        stratify=[labels[i] for i in trainval_idx])

    aug_cfg = cfg.get("augment", {})
    tf_train = build_transforms(dcfg["image_size"], dcfg["in_channels"],
                                dcfg.get("imagenet_norm", False),
                                augment=True, aug_cfg=aug_cfg)
    tf_eval = build_transforms(dcfg["image_size"], dcfg["in_channels"],
                               dcfg.get("imagenet_norm", False), augment=False)

    def subset(idxs, tf):
        return NEUDataset([paths[i] for i in idxs], [labels[i] for i in idxs], tf)

    loaders = {
        "train": DataLoader(subset(train_idx, tf_train), batch_size=dcfg["batch_size"],
                            shuffle=True, num_workers=dcfg.get("num_workers", 0)),
        "val": DataLoader(subset(val_idx, tf_eval), batch_size=dcfg["batch_size"],
                          shuffle=False, num_workers=dcfg.get("num_workers", 0)),
        "test": DataLoader(subset(test_idx, tf_eval), batch_size=dcfg["batch_size"],
                           shuffle=False, num_workers=dcfg.get("num_workers", 0)),
    }
    # Persist the split so training and evaluation always use identical sets.
    split_path = Path(cfg["paths"]["splits"])
    split_path.parent.mkdir(parents=True, exist_ok=True)
    with open(split_path, "w", encoding="utf-8") as f:
        json.dump({"train": train_idx, "val": val_idx, "test": test_idx,
                   "seed": seed, "class_names": class_names}, f)
    info = {
        "class_names": class_names,
        "n_total": len(paths),
        "n_train": len(train_idx), "n_val": len(val_idx), "n_test": len(test_idx),
        "counts": {c: labels.count(i) for i, c in enumerate(class_names)},
    }
    return loaders, info
