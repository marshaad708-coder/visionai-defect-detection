"""VisionAI smoke tests — no extra test framework needed.

Run from the repo root (with the virtual environment activated):
    python tests/test_smoke.py

Every test either PASSES, FAILS, or SKIPS (with a reason, e.g. dataset not
downloaded yet). A FAIL means something is genuinely broken; a SKIP is not
a failure. Exit code 0 = all good, 1 = at least one failure.
"""
import sys
import traceback
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

_RESULTS = []


def check(name):
    def deco(fn):
        def wrapper():
            try:
                fn()
            except Exception as e:  # noqa: BLE001
                if isinstance(e, Skipped):
                    _RESULTS.append((name, "SKIP", str(e)))
                else:
                    _RESULTS.append((name, "FAIL", f"{e}\n{traceback.format_exc(limit=3)}"))
            else:
                _RESULTS.append((name, "PASS", ""))
        wrapper.__name__ = fn.__name__
        return wrapper
    return deco


class Skipped(Exception):
    pass


def require_data(path: Path, what: str) -> Path:
    if not path.exists():
        raise Skipped(f"{what} not present at {path} (run scripts/download_*.py)")
    return path


@check("config: neu_compactcnn.yaml loads")
def t_config():
    from src.common.config import load_config
    cfg = load_config(REPO / "configs" / "neu_compactcnn.yaml")
    assert cfg["model"]["name"] == "compact_cnn"
    assert cfg["data"]["batch_size"] == 32
    assert cfg["seed"] == 42


@check("reproducibility: set_seed fixes RNG streams")
def t_seed():
    import torch
    from src.common.seed import set_seed
    set_seed(123)
    a = torch.randn(8)
    set_seed(123)
    b = torch.randn(8)
    assert torch.equal(a, b), "RNG streams diverged after identical seeds"


@check("models: CompactCNN forward + Grad-CAM hook point")
def t_compact_cnn():
    import torch
    from src import models
    m = models.get_model("compact_cnn", num_classes=6)
    m.eval()
    with torch.no_grad():
        y = m(torch.randn(2, 1, 224, 224))
    assert y.shape == (2, 6), f"unexpected output shape {y.shape}"
    assert hasattr(m, "last_conv"), "missing last_conv for Grad-CAM"
    n = models.count_parameters(m)
    assert 500_000 < n < 2_000_000, f"param count {n} outside expected range"


@check("models: ResNet-18 builder (no pretrained download)")
def t_resnet18():
    import torch
    from src import models
    m = models.get_model("resnet18", num_classes=6, pretrained=False)
    m.eval()
    with torch.no_grad():
        y = m(torch.randn(1, 3, 224, 224))
    assert y.shape == (1, 6)
    assert hasattr(m, "last_conv")


@check("data_neu: discovers 6 classes x 300 images")
def t_neu_discovery():
    from src import data_neu
    root = require_data(REPO / "data" / "raw" / "neu", "NEU dataset")
    paths, labels, classes = data_neu.discover_samples(root)
    assert classes == data_neu.NEU_CLASSES, f"class list mismatch: {classes}"
    assert len(paths) == 1800, f"expected 1800 images, found {len(paths)}"
    from collections import Counter
    counts = Counter(labels)
    assert all(counts[i] == 300 for i in range(6)), f"per-class counts: {dict(counts)}"


@check("data_neu: stratified splits + transform shapes")
def t_neu_splits():
    from src.common.config import load_config
    from src import data_neu
    cfg = load_config(REPO / "configs" / "neu_compactcnn.yaml")
    require_data(REPO / cfg["data"]["root"], "NEU dataset")
    loaders, info = data_neu.make_dataloaders(cfg, seed=42)
    # 70/15/15 of 1800 -> 1260/270/270
    assert info["n_train"] == 1260, info
    assert info["n_val"] == 270, info
    assert info["n_test"] == 270, info
    xb, yb = next(iter(loaders["train"]))
    assert xb.shape[1:] == (1, 224, 224), f"batch shape {xb.shape}"
    assert (REPO / cfg["paths"]["splits"]).exists(), "splits JSON not saved"


@check("data_mvtec: category audit (toothbrush)")
def t_mvtec():
    from src import data_mvtec
    from torch.utils.data import DataLoader
    cat = require_data(REPO / "data" / "raw" / "mvtec" / "toothbrush",
                       "MVTec AD toothbrush")
    summary = data_mvtec.summarize_category(cat.parent, "toothbrush")
    assert summary["train_good"] == 60, summary
    assert summary["test"].get("good") == 12, summary
    assert summary["test"].get("defective") == 30, summary
    # Loader returns masks paired to defective images.
    ds = data_mvtec.MVTecDataset(cat.parent, "toothbrush", split="test", image_size=224)
    assert len(ds) == 42, len(ds)
    n_masked = 0
    for i in range(len(ds)):
        s = ds[i]
        assert s["image"].shape == (3, 224, 224), s["image"].shape
        assert s["mask"].shape == (224, 224), s["mask"].shape
        if s["label"] == 1:
            assert s["mask"].sum() > 0, f"empty mask for {s['path']}"
            n_masked += 1
    assert n_masked == 30, n_masked
    tr = data_mvtec.MVTecDataset(cat.parent, "toothbrush", split="train", image_size=224)
    assert len(tr) == 60 and all(s["label"] == 0 for s in
                                (tr[i] for i in range(0, 60, 15)))


def main() -> int:
    for fn in [t_config, t_seed, t_compact_cnn, t_resnet18,
               t_neu_discovery, t_neu_splits, t_mvtec]:
        fn()
    print("\n=== VisionAI smoke tests ===")
    failed = 0
    for name, status, detail in _RESULTS:
        mark = {"PASS": "PASS", "FAIL": "FAIL", "SKIP": "SKIP"}[status]
        print(f"[{mark}] {name}")
        if detail and status != "PASS":
            print(f"       {detail.splitlines()[0]}")
        if status == "FAIL":
            failed += 1
    print(f"\n{len(_RESULTS) - failed} passed-or-skipped, {failed} failed.")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
