"""OSNet (Zhou et al., "Omni-Scale Feature Learning for Person
Re-Identification", ICCV 2019) -- written from the paper so that the
parameter names match the official `torchreid` checkpoints exactly, which
lets us load the authors' MSMT17-trained weights with `strict=True`
without installing `torchreid` (an unmaintained, build-heavy dependency).

OSNet's trick is an *omni-scale* residual block: four parallel streams of
1, 2, 3 and 4 stacked light 3x3 convolutions (receptive fields 3, 5, 7 and
9) fused by a *shared* channel gate, so each block sees both fine texture
(a bag strap) and coarse shape (body silhouette) at a tiny parameter
budget -- 0.2M parameters for the x0.25 variant used by default.
"""

from __future__ import annotations

import torch
from torch import nn

# width multiplier -> channel widths of (stem, stage2, stage3, stage4)
OSNET_CHANNELS: dict[str, tuple[int, int, int, int]] = {
    "x0_25": (16, 64, 96, 128),
    "x0_5": (32, 128, 192, 256),
    "x0_75": (48, 192, 288, 384),
    "x1_0": (64, 256, 384, 512),
}
FEATURE_DIM = 512


class ConvLayer(nn.Module):
    def __init__(self, cin: int, cout: int, k: int, stride: int = 1, padding: int = 0):
        super().__init__()
        self.conv = nn.Conv2d(cin, cout, k, stride=stride, padding=padding, bias=False)
        self.bn = nn.BatchNorm2d(cout)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.relu(self.bn(self.conv(x)))


class Conv1x1(nn.Module):
    def __init__(self, cin: int, cout: int):
        super().__init__()
        self.conv = nn.Conv2d(cin, cout, 1, bias=False)
        self.bn = nn.BatchNorm2d(cout)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.relu(self.bn(self.conv(x)))


class Conv1x1Linear(nn.Module):
    """1x1 conv + BN, no activation (the residual branch's output)."""

    def __init__(self, cin: int, cout: int):
        super().__init__()
        self.conv = nn.Conv2d(cin, cout, 1, bias=False)
        self.bn = nn.BatchNorm2d(cout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.bn(self.conv(x))


class LightConv3x3(nn.Module):
    """Pointwise 1x1 followed by depthwise 3x3: a 3x3 conv's receptive field
    at a fraction of its parameters."""

    def __init__(self, cin: int, cout: int):
        super().__init__()
        self.conv1 = nn.Conv2d(cin, cout, 1, bias=False)
        self.conv2 = nn.Conv2d(cout, cout, 3, padding=1, bias=False, groups=cout)
        self.bn = nn.BatchNorm2d(cout)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.relu(self.bn(self.conv2(self.conv1(x))))


class ChannelGate(nn.Module):
    """Squeeze-and-excitation style unified aggregation gate."""

    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Conv2d(channels, channels // reduction, 1, bias=True)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Conv2d(channels // reduction, channels, 1, bias=True)
        self.gate = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate = self.gate(self.fc2(self.relu(self.fc1(self.pool(x)))))
        return x * gate


class OSBlock(nn.Module):
    def __init__(self, cin: int, cout: int, bottleneck_reduction: int = 4):
        super().__init__()
        mid = cout // bottleneck_reduction
        self.conv1 = Conv1x1(cin, mid)
        self.conv2a = LightConv3x3(mid, mid)
        self.conv2b = nn.Sequential(LightConv3x3(mid, mid), LightConv3x3(mid, mid))
        self.conv2c = nn.Sequential(*[LightConv3x3(mid, mid) for _ in range(3)])
        self.conv2d = nn.Sequential(*[LightConv3x3(mid, mid) for _ in range(4)])
        self.gate = ChannelGate(mid)
        self.conv3 = Conv1x1Linear(mid, cout)
        self.downsample = Conv1x1Linear(cin, cout) if cin != cout else None
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        x1 = self.conv1(x)
        x2 = (
            self.gate(self.conv2a(x1))
            + self.gate(self.conv2b(x1))
            + self.gate(self.conv2c(x1))
            + self.gate(self.conv2d(x1))
        )
        out = self.conv3(x2)
        if self.downsample is not None:
            identity = self.downsample(identity)
        return self.relu(out + identity)


class OSNet(nn.Module):
    def __init__(self, width: str = "x0_25", num_classes: int = 4101):
        super().__init__()
        c = OSNET_CHANNELS[width]
        self.conv1 = ConvLayer(3, c[0], 7, stride=2, padding=3)
        self.maxpool = nn.MaxPool2d(3, stride=2, padding=1)
        self.conv2 = self._stage(c[0], c[1], reduce=True)
        self.conv3 = self._stage(c[1], c[2], reduce=True)
        self.conv4 = self._stage(c[2], c[3], reduce=False)
        self.conv5 = Conv1x1(c[3], c[3])
        self.global_avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(c[3], FEATURE_DIM), nn.BatchNorm1d(FEATURE_DIM), nn.ReLU(inplace=True)
        )
        # Identity-classification head: only present so official checkpoints load
        # with strict=True; unused at inference (we want the 512-d embedding).
        self.classifier = nn.Linear(FEATURE_DIM, num_classes)

    @staticmethod
    def _stage(cin: int, cout: int, reduce: bool) -> nn.Sequential:
        layers: list[nn.Module] = [OSBlock(cin, cout), OSBlock(cout, cout)]
        if reduce:
            layers.append(nn.Sequential(Conv1x1(cout, cout), nn.AvgPool2d(2, stride=2)))
        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.maxpool(self.conv1(x))
        x = self.conv5(self.conv4(self.conv3(self.conv2(x))))
        return self.fc(self.global_avgpool(x).flatten(1))
