import hashlib

import numpy as np
import pytest
import torch

from emot.reid.embedders import OSNET_CHECKPOINTS, _sha256, build_embedder
from emot.reid.osnet import FEATURE_DIM, OSNET_CHANNELS, OSNet


@pytest.mark.parametrize("width", list(OSNET_CHANNELS))
def test_osnet_outputs_a_512d_feature_for_every_width(width):
    model = OSNet(width).eval()
    out = model(torch.zeros(2, 3, 256, 128))
    assert out.shape == (2, FEATURE_DIM)


def test_osnet_parameter_names_match_torchreid_checkpoint_layout():
    keys = set(OSNet("x0_25").state_dict())
    for expected in (
        "conv1.conv.weight",
        "conv2.0.conv2b.0.conv1.weight",
        "conv2.0.gate.fc1.weight",
        "conv2.0.downsample.conv.weight",
        "fc.0.weight",
        "fc.1.running_mean",
        "classifier.weight",
    ):
        assert expected in keys


def test_unknown_embedder_is_rejected():
    with pytest.raises(ValueError, match="unknown embedder"):
        build_embedder("clip")


def test_checkpoint_registry_has_pinned_hashes():
    for _, _, digest in OSNET_CHECKPOINTS.values():
        assert len(digest) == 64


def test_sha256_helper(tmp_path):
    f = tmp_path / "x.bin"
    f.write_bytes(b"emot")
    assert _sha256(f) == hashlib.sha256(b"emot").hexdigest()


def test_embedder_returns_unit_vectors_and_handles_empty_input():
    torch.manual_seed(0)
    from emot.reid.embedders import _TorchEmbedder

    class Tiny(_TorchEmbedder):
        name, dim, input_size = "tiny", FEATURE_DIM, (64, 32)

    e = Tiny(OSNet("x0_25"), "cpu")
    crops = [
        np.random.default_rng(i).integers(0, 255, (80 + 10 * i, 30, 3), dtype=np.uint8)
        for i in range(3)
    ]
    out = e.embed_crops(crops)
    assert out.shape == (3, FEATURE_DIM) and np.allclose(np.linalg.norm(out, axis=1), 1, atol=1e-4)
    assert e.embed_crops([]).shape == (0, FEATURE_DIM)
