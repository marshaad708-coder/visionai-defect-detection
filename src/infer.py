"""Inference for VisionAI Track A: real model predictions + Grad-CAM heatmaps.

No mock data anywhere: every score comes from a forward pass of the trained
checkpoint. Softmax outputs are reported as *model scores* and are explicitly
NOT calibrated probabilities (see README "Scores vs calibrated probabilities").
"""
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from src.data_neu import build_transforms  # noqa: E402
from src.gradcam import GradCAM, overlay_heatmap  # noqa: E402
from src.models import get_model  # noqa: E402

CALIBRATION_NOTE = ("Model scores are raw softmax outputs, not calibrated "
                    "probabilities. See README for details.")


class DefectPredictor:
    """Loads a trained Track A checkpoint and runs real inference."""

    def __init__(self, checkpoint_path: str | Path, device: str | None = None):
        ckpt_path = Path(checkpoint_path)
        if not ckpt_path.exists():
            raise FileNotFoundError(
                f"Checkpoint not found: {ckpt_path}. "
                "Train first: python src/train_classifier.py --config configs/neu_compactcnn.yaml")
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        ckpt = torch.load(ckpt_path, map_location=self.device, weights_only=False)
        self.class_names = ckpt["class_names"]
        self.train_config = ckpt["config"]
        mcfg = self.train_config["model"]
        self.model = get_model(mcfg["name"], num_classes=mcfg["num_classes"],
                               **{k: v for k, v in mcfg.items()
                                  if k not in ("name", "num_classes")})
        self.model.load_state_dict(ckpt["model_state"])
        self.model.to(self.device).eval()
        dcfg = self.train_config["data"]
        self.transform = build_transforms(dcfg["image_size"], dcfg["in_channels"],
                                          dcfg.get("imagenet_norm", False), augment=False)
        self.image_size = dcfg["image_size"]
        self._gradcam = GradCAM(self.model, self.model.last_conv)

    def _load_image(self, image_path: str | Path) -> Image.Image:
        p = Path(image_path)
        if not p.exists():
            raise FileNotFoundError(f"Image not found: {p}")
        try:
            img = Image.open(p)
            img.verify()
            img = Image.open(p).convert("RGB")
        except Exception as e:  # noqa: BLE001
            raise ValueError(f"Not a readable image: {p} ({e})")
        return img

    @torch.no_grad()
    def predict(self, image_path: str | Path) -> dict:
        """Run the real model. Returns predicted class, scores, top-3."""
        img = self._load_image(image_path)
        x = self.transform(img.convert("L")).unsqueeze(0).to(self.device)
        logits = self.model(x)
        scores = torch.softmax(logits, dim=1)[0].cpu().numpy()
        order = np.argsort(scores)[::-1]
        return {
            "predicted_class": self.class_names[int(order[0])],
            "predicted_index": int(order[0]),
            "scores": {self.class_names[int(i)]: float(scores[i]) for i in order},
            "top3": [(self.class_names[int(i)], float(scores[i])) for i in order[:3]],
            "calibration_note": CALIBRATION_NOTE,
        }

    def predict_with_heatmap(self, image_path: str | Path):
        """Returns (prediction dict, overlay PIL image, heatmap HxW in [0,1])."""
        pred = self.predict(image_path)
        img = self._load_image(image_path).resize((self.image_size, self.image_size))
        x = self.transform(img.convert("L")).unsqueeze(0).to(self.device)
        heat = self._gradcam(x, class_idx=pred["predicted_index"])
        overlay = overlay_heatmap(np.array(img), heat)
        return pred, Image.fromarray(overlay), heat


def main() -> int:
    import argparse
    ap = argparse.ArgumentParser(description="Run Track A inference on one image.")
    ap.add_argument("--checkpoint", default="checkpoints/neu_compactcnn_best.pt")
    ap.add_argument("--image", required=True)
    ap.add_argument("--out", default=None, help="Save Grad-CAM overlay PNG here.")
    args = ap.parse_args()

    predictor = DefectPredictor(args.checkpoint)
    pred, overlay, _ = predictor.predict_with_heatmap(args.image)
    print(f"Predicted: {pred['predicted_class']}")
    for cls, s in pred["top3"]:
        print(f"  {cls:15s} {s:.4f}")
    print(pred["calibration_note"])
    if args.out:
        overlay.save(args.out)
        print(f"Overlay saved: {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
