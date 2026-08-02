"""Appearance embedding extractor for DeepSORT.

Uses a frozen, ImageNet-pretrained ResNet18's global-average-pooled
penultimate-layer features (512-d, L2-normalized) as the appearance
descriptor for each detected person crop.

This is a deliberate simplification, documented in docs/limitations.md:
the original DeepSORT paper (and most production systems since) uses a
dedicated person re-identification network trained with a metric-learning
loss (triplet/contrastive) on person-ReID datasets like Market-1501,
which produces embeddings specifically optimized so that "same person"
pairs are close and "different person" pairs are far apart. A generic
ImageNet classifier's features are a weaker but real and legitimate
off-the-shelf substitute -- exactly the approach used in DeepSORT's own
predecessors before dedicated ReID nets became standard -- and avoids
pulling in an extra pretrained-weights dependency for a CPU-only laptop
project. The gap this leaves is measured directly by comparing DeepSORT's
ID-switch count against ByteTrack's in docs/results.md.
"""

from __future__ import annotations

import numpy as np
import torch
from torch import nn
from torchvision.models import ResNet18_Weights, resnet18
from torchvision.transforms import functional as F

EMBEDDING_DIM = 512


class AppearanceEmbedder:
    def __init__(self, device: str = "cpu"):
        self.device = device
        backbone = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        backbone.fc = nn.Identity()
        backbone.eval()
        self.model = backbone.to(device)
        self.mean = torch.tensor(ResNet18_Weights.IMAGENET1K_V1.transforms().mean).view(3, 1, 1)
        self.std = torch.tensor(ResNet18_Weights.IMAGENET1K_V1.transforms().std).view(3, 1, 1)

    @torch.no_grad()
    def embed_crops(self, crops: list[np.ndarray]) -> np.ndarray:
        """`crops` are HxWx3 RGB uint8 arrays (already cropped to each
        detection's box). Returns an `(N, 512)` L2-normalized embedding
        array; empty input returns a correctly-shaped empty array."""
        if not crops:
            return np.zeros((0, EMBEDDING_DIM))

        tensors = []
        for crop in crops:
            tensor = torch.from_numpy(crop).permute(2, 0, 1).float() / 255.0
            tensor = F.resize(tensor, [128, 64], antialias=True)
            tensor = (tensor - self.mean) / self.std
            tensors.append(tensor)
        batch = torch.stack(tensors).to(self.device)

        features = self.model(batch)
        features = features / features.norm(dim=1, keepdim=True).clamp_min(1e-8)
        return features.cpu().numpy()
