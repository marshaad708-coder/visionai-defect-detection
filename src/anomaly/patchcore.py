"""PatchCore for VisionAI Track B (unsupervised industrial anomaly detection).

Reference: T. Roth et al., "Towards Total Recall in Industrial Anomaly
Detection," CVPR 2022.

Pipeline (all steps are real computation, no mock data):
  1. Patch embeddings: mid-level features (layer2 + layer3) of an
     ImageNet-pretrained WideResNet-50, locally aggregated with 3x3
     average pooling and concatenated -> one vector per image patch.
  2. Memory bank: greedy k-center coreset subsampling of the patch
     embeddings of NORMAL training images only.
  3. Inference: for each test patch, distance to its nearest bank
     neighbour. Image score = max patch distance with neighbourhood-density
     reweighting (Eq. 7 of the paper); pixel map = per-patch distances
     upsampled to image size and Gaussian-smoothed.

Documented deviations from the reference implementation (practical, CPU-first):
  - Coreset selection picks `coreset_batch` points per greedy step instead of
    one; with batch=1 this is exactly the paper's algorithm.
  - Before coreset selection, each training image contributes
    `patches_per_image` uniformly random patches, bounding peak RAM on
    CPU-only machines. The paper uses all patches.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class PatchCoreBackbone(nn.Module):
    """WideResNet-50 trunk returning (layer2, layer3) feature maps."""

    def __init__(self, weights: str = "IMAGENET1K_V1"):
        super().__init__()
        from torchvision.models import Wide_ResNet50_2_Weights, wide_resnet50_2
        w = Wide_ResNet50_2_Weights.IMAGENET1K_V1 if weights == "IMAGENET1K_V1" else None
        net = wide_resnet50_2(weights=w)
        self.conv1, self.bn1, self.relu, self.maxpool = \
            net.conv1, net.bn1, net.relu, net.maxpool
        self.layer1, self.layer2, self.layer3 = net.layer1, net.layer2, net.layer3
        self.eval()
        for p in self.parameters():
            p.requires_grad_(False)

    def forward(self, x: torch.Tensor):
        x = self.maxpool(self.relu(self.bn1(self.conv1(x))))
        x = self.layer1(x)
        f2 = self.layer2(x)
        f3 = self.layer3(f2)
        return f2, f3


def embed_patches(f2: torch.Tensor, f3: torch.Tensor, patch: int = 3):
    """Aggregate + concat layer2/layer3 maps -> (B, P, D) patch embeddings."""
    p = patch // 2
    f2 = F.avg_pool2d(f2, patch, stride=1, padding=p)
    f3 = F.avg_pool2d(f3, patch, stride=1, padding=p)
    f3 = F.interpolate(f3, size=f2.shape[-2:], mode="bilinear", align_corners=False)
    emb = torch.cat([f2, f3], dim=1)
    b, c, h, w = emb.shape
    return emb.permute(0, 2, 3, 1).reshape(b, h * w, c), (h, w)


def greedy_coreset(embeddings: torch.Tensor, n_sample: int,
                   batch: int = 64, seed: int = 42) -> torch.Tensor:
    """Greedy k-center coreset (Roth et al., Alg. 1), batched for CPU speed.

    embeddings: (N, D) CPU tensor. Returns (n_sample, D).
    """
    assert embeddings.dim() == 2 and embeddings.shape[0] > 0
    n_sample = min(n_sample, embeddings.shape[0])
    g = torch.Generator().manual_seed(seed)
    start = int(torch.randint(0, embeddings.shape[0], (1,), generator=g))
    selected = [start]
    with torch.no_grad():
        dists = ((embeddings - embeddings[[start]]) ** 2).sum(1)
        dists[start] = -1.0  # never re-select
        while len(selected) < n_sample:
            k = min(batch, n_sample - len(selected))
            idx = torch.topk(dists, k).indices.tolist()
            selected.extend(idx)
            d2 = torch.cdist(embeddings, embeddings[idx]).pow(2).min(1).values
            dists = torch.minimum(dists, d2)
            dists[idx] = -1.0
    return embeddings[selected]


class PatchCore:
    """Unsupervised anomaly detector. Train on normal images only."""

    def __init__(self, backbone: str = "wide_resnet50_2", patch: int = 3,
                 knn_k: int = 9, device: str = "cpu"):
        if backbone != "wide_resnet50_2":
            raise ValueError("Only 'wide_resnet50_2' is currently supported.")
        self.device = torch.device(device)
        self.patch = patch
        self.knn_k = knn_k
        self.net = PatchCoreBackbone().to(self.device)
        self.bank: torch.Tensor | None = None   # (M, D)
        self.grid: tuple[int, int] | None = None
        self.threshold: float | None = None
        self._nn = None

    @torch.no_grad()
    def _embed_batch(self, x: torch.Tensor):
        f2, f3 = self.net(x.to(self.device))
        emb, grid = embed_patches(f2, f3, self.patch)
        return emb.cpu(), grid  # (B, P, D)

    @torch.no_grad()
    def fit(self, loader, patches_per_image: int = 128,
            coreset_ratio: float = 0.01, coreset_max: int = 20000,
            coreset_batch: int = 64, seed: int = 42) -> dict:
        """Build the memory bank from a loader of NORMAL images. Two passes."""
        pool, self.grid = [], None
        n_images = 0
        for batch in loader:
            imgs = batch["image"]
            emb, self.grid = self._embed_batch(imgs)
            b, p, _ = emb.shape
            keep = min(patches_per_image, p)
            idx = torch.randperm(p)[:keep]
            pool.append(emb[:, idx, :].reshape(-1, emb.shape[-1]))
            n_images += b
        pool = torch.cat(pool, dim=0)
        n_bank = min(coreset_max, max(1, int(len(pool) * coreset_ratio)))
        self.bank = greedy_coreset(pool, n_bank, batch=coreset_batch, seed=seed)
        del pool
        from sklearn.neighbors import NearestNeighbors
        self._nn = NearestNeighbors(n_neighbors=self.knn_k, algorithm="auto")
        self._nn.fit(self.bank.numpy())
        return {"n_train_images": n_images, "bank_size": len(self.bank),
                "embed_dim": self.bank.shape[1], "patch_grid": self.grid}

    @torch.no_grad()
    def calibrate(self, loader, quantile: float = 0.995) -> float:
        """Set the operating threshold from NORMAL training image scores.

        threshold = `quantile`-quantile of train image scores. This is an
        operating point, not a calibrated probability.
        """
        scores = []
        for batch in loader:
            emb, _ = self._embed_batch(batch["image"])
            for i in range(emb.shape[0]):
                scores.append(self._image_score(emb[i]))
        self.threshold = float(torch.tensor(scores).quantile(quantile))
        return self.threshold

    def _patch_distances(self, emb: torch.Tensor):
        """emb: (P, D) -> (nearest distances (P,), all k distances (P, k))."""
        dists, _ = self._nn.kneighbors(emb.numpy(), n_neighbors=self.knn_k)
        return (torch.from_numpy(dists[:, 0]),
                torch.from_numpy(dists))  # (P,), (P, k)

    def _image_score(self, emb: torch.Tensor) -> float:
        """Max patch distance with neighbourhood-density reweighting (Eq. 7)."""
        d1, dk = self._patch_distances(emb)
        m = int(torch.argmax(d1))
        d_star = float(d1[m])
        neigh = dk[m]  # (k,) distances from test patch m to its k neighbours
        w = 1.0 - float(torch.exp(torch.tensor(-d_star)) /
                        torch.exp(-neigh).sum())
        return w * d_star

    @torch.no_grad()
    def anomaly_map(self, image: torch.Tensor):
        """image: (3, H, W) tensor -> (image_score, heatmap HxW in [0,1])."""
        if self.bank is None:
            raise RuntimeError("Call fit() before inference.")
        emb, (gh, gw) = self._embed_batch(image.unsqueeze(0))
        emb = emb[0]
        score = self._image_score(emb)
        d1, _ = self._patch_distances(emb)
        pmap = d1.reshape(gh, gw)
        pmap = (pmap - pmap.min()) / (pmap.max() - pmap.min() + 1e-8)
        heat = F.interpolate(pmap.unsqueeze(0).unsqueeze(0),
                             size=image.shape[-2:], mode="bilinear",
                             align_corners=False)[0, 0].numpy()
        # Gaussian smoothing (as in the paper's post-processing).
        import cv2
        heat = cv2.GaussianBlur(heat, (0, 0), sigmaX=4)
        heat = (heat - heat.min()) / (heat.max() - heat.min() + 1e-8)
        return float(score), heat.astype("float32")

    def save(self, path) -> None:
        import pathlib
        path = pathlib.Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"bank": self.bank, "grid": self.grid,
                    "threshold": self.threshold, "patch": self.patch,
                    "knn_k": self.knn_k}, path)

    @classmethod
    def load(cls, path, device: str = "cpu") -> "PatchCore":
        data = torch.load(path, map_location="cpu", weights_only=False)
        obj = cls(patch=data["patch"], knn_k=data["knn_k"], device=device)
        obj.bank, obj.grid, obj.threshold = data["bank"], data["grid"], data["threshold"]
        from sklearn.neighbors import NearestNeighbors
        obj._nn = NearestNeighbors(n_neighbors=obj.knn_k, algorithm="auto")
        obj._nn.fit(obj.bank.numpy())
        return obj
