"""Grad-CAM visualization for VisionAI Track A classifiers.

Reference: R. R. Selvaraju et al., "Grad-CAM: Visual Explanations from Deep
Networks via Gradient-based Localization," ICCV 2017.

Given a trained classifier and an input image, Grad-CAM produces a heatmap
highlighting the image regions that most influenced the predicted (or a
chosen) class. This is a *post-hoc explanation* of the model, not an
independent defect detector.
"""
import cv2
import numpy as np
import torch


class GradCAM:
    def __init__(self, model: torch.nn.Module, target_layer: torch.nn.Module):
        self.model = model
        self.target_layer = target_layer
        self._activations = None
        self._gradients = None
        target_layer.register_forward_hook(self._save_activation)
        # full backward hook: correct API for nn.Module targets.
        target_layer.register_full_backward_hook(self._save_gradient)

    def _save_activation(self, module, inp, out):
        self._activations = out.detach()

    def _save_gradient(self, module, grad_in, grad_out):
        self._gradients = grad_out[0].detach()

    def heatmap(self, x: torch.Tensor, class_idx: int | None = None) -> np.ndarray:
        """Return a HxW heatmap in [0, 1] for input batch x (uses x[0])."""
        self.model.eval()
        self.model.zero_grad()
        logits = self.model(x)
        if class_idx is None:
            class_idx = int(logits[0].argmax())
        score = logits[0, class_idx]
        score.backward()

        # Channel weights = global average of gradients; weighted sum of activations.
        weights = self._gradients[0].mean(dim=(1, 2), keepdim=True)  # C x 1 x 1
        cam = (weights * self._activations[0]).sum(dim=0)            # H x W
        cam = torch.relu(cam)
        cam -= cam.min()
        if cam.max() > 0:
            cam /= cam.max()
        h, w = x.shape[2], x.shape[3]
        cam = cam.cpu().numpy()
        cam = cv2.resize(cam, (w, h))
        return cam.astype(np.float32)

    def __call__(self, x: torch.Tensor, class_idx: int | None = None) -> np.ndarray:
        return self.heatmap(x, class_idx)


def overlay_heatmap(image: np.ndarray, heatmap: np.ndarray,
                    alpha: float = 0.45) -> np.ndarray:
    """Blend a [0,1] heatmap (HxW) over an RGB uint8 image -> RGB uint8 overlay."""
    if image.dtype != np.uint8:
        raise ValueError("image must be uint8")
    h, w = image.shape[:2]
    if heatmap.shape != (h, w):
        heatmap = cv2.resize(heatmap, (w, h))
    colored = cv2.applyColorMap((heatmap * 255).astype(np.uint8),
                                cv2.COLORMAP_JET)          # BGR
    colored = cv2.cvtColor(colored, cv2.COLOR_BGR2RGB)      # -> RGB
    blended = (alpha * colored + (1 - alpha) * image).astype(np.uint8)
    return blended
