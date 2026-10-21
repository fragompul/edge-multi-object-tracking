"""Crop -> L2-normalised appearance embedding, behind one small interface.

Two embedders are provided so the benchmark can *measure* what a real
re-identification network buys over a generic classifier:

- `resnet18`: frozen ImageNet ResNet18 features (the original, deliberately
  weak baseline -- features optimised for "which of 1000 classes" rather
  than "which of these two similar pedestrians").
- `osnet_x0_25` / `osnet_x0_5`: OSNet trained with identity supervision on
  MSMT17 (4,101 identities), i.e. features optimised for exactly the
  same-person / different-person decision the tracker has to make.
"""

from __future__ import annotations

import hashlib
import logging
import urllib.request
from pathlib import Path
from typing import Protocol

import numpy as np
import torch
from torch import nn
from torchvision.models import ResNet18_Weights, resnet18
from torchvision.transforms import functional as F

from emot.reid.osnet import FEATURE_DIM, OSNet

logger = logging.getLogger(__name__)

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

WEIGHTS_DIR = Path(__file__).resolve().parents[3] / "weights"
_HF = "https://huggingface.co/kaiyangzhou/osnet/resolve/main"
_MSMT = "combineall_256x128_amsgrad_ep150_stp60_lr0.0015_b64_fb10_softmax_labelsmooth_flip_jitter"
# name -> (width, remote filename, sha256 of the checkpoint)
OSNET_CHECKPOINTS: dict[str, tuple[str, str, str]] = {
    "osnet_x0_25": (
        "x0_25",
        f"osnet_x0_25_msmt17_{_MSMT}.pth",
        "cf55163d78fc44c62c82f85ab62d39f10438679b5abe8c698ae08cfa84aa6e18",
    ),
    "osnet_x0_5": (
        "x0_5",
        f"osnet_x0_5_msmt17_{_MSMT}.pth",
        "e96cbd20ee9cc3c6dcc0e8f4fbba8c8069d47647a5a96a59ce381fb785c54f68",
    ),
}
EMBEDDER_NAMES = ["resnet18", *OSNET_CHECKPOINTS]


class Embedder(Protocol):
    name: str
    dim: int

    def embed_crops(self, crops: list[np.ndarray]) -> np.ndarray: ...


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def fetch_osnet_checkpoint(name: str, weights_dir: Path = WEIGHTS_DIR) -> Path:
    """Downloads the official checkpoint on first use and verifies its hash."""
    _, remote, expected = OSNET_CHECKPOINTS[name]
    dest = weights_dir / f"{name}_msmt17.pth"
    if not dest.exists():
        weights_dir.mkdir(parents=True, exist_ok=True)
        logger.info("downloading %s ...", remote)
        urllib.request.urlretrieve(f"{_HF}/{remote}", dest)  # noqa: S310 - fixed HF host
    if _sha256(dest) != expected:
        raise RuntimeError(f"checksum mismatch for {dest}; delete it and retry")
    return dest


def _preprocess(crops: list[np.ndarray], size: tuple[int, int]) -> torch.Tensor:
    tensors = []
    mean = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
    std = torch.tensor(IMAGENET_STD).view(3, 1, 1)
    for crop in crops:
        tensor = torch.from_numpy(np.ascontiguousarray(crop)).permute(2, 0, 1).float() / 255.0
        tensor = F.resize(tensor, list(size), antialias=True)
        tensors.append((tensor - mean) / std)
    return torch.stack(tensors)


class _TorchEmbedder:
    name: str
    dim: int
    input_size: tuple[int, int]

    def __init__(self, model: nn.Module, device: str, batch_size: int = 64):
        self.model = model.eval().to(device)
        self.device = device
        self.batch_size = batch_size

    @torch.no_grad()
    def embed_crops(self, crops: list[np.ndarray]) -> np.ndarray:
        """`crops` are HxWx3 **RGB** uint8 arrays. Returns `(N, dim)` L2-normalised
        float32 embeddings; empty input yields a correctly shaped empty array."""
        if not crops:
            return np.zeros((0, self.dim), dtype=np.float32)
        out = []
        for start in range(0, len(crops), self.batch_size):
            batch = _preprocess(crops[start : start + self.batch_size], self.input_size)
            feats = self.model(batch.to(self.device))
            feats = feats / feats.norm(dim=1, keepdim=True).clamp_min(1e-8)
            out.append(feats.cpu().numpy())
        return np.concatenate(out).astype(np.float32)


class ResNet18Embedder(_TorchEmbedder):
    name = "resnet18"
    dim = 512
    input_size = (128, 64)

    def __init__(self, device: str = "cpu"):
        backbone = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        backbone.fc = nn.Identity()
        super().__init__(backbone, device)


class OSNetEmbedder(_TorchEmbedder):
    dim = FEATURE_DIM
    input_size = (256, 128)

    def __init__(self, name: str = "osnet_x0_25", device: str = "cpu"):
        width, _, _ = OSNET_CHECKPOINTS[name]
        model = OSNet(width)
        state = torch.load(fetch_osnet_checkpoint(name), map_location="cpu", weights_only=True)
        model.load_state_dict(state, strict=True)
        self.name = name
        super().__init__(model, device)


def build_embedder(name: str = "osnet_x0_25", device: str = "cpu") -> Embedder:
    if name == "resnet18":
        return ResNet18Embedder(device)
    if name in OSNET_CHECKPOINTS:
        return OSNetEmbedder(name, device)
    raise ValueError(f"unknown embedder {name!r}; choose from {EMBEDDER_NAMES}")
