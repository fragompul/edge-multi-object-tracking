"""Appearance (re-identification) embedders used by the appearance-aware trackers."""

from emot.reid.embedders import (
    EMBEDDER_NAMES,
    Embedder,
    OSNetEmbedder,
    ResNet18Embedder,
    build_embedder,
)

__all__ = ["EMBEDDER_NAMES", "Embedder", "OSNetEmbedder", "ResNet18Embedder", "build_embedder"]
