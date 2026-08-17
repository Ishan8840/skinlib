"""BiSeNet architecture, matching the CelebAMask-HQ pretrained checkpoint.

This is a definition, not a training artefact — no model is trained here. The
layer names reproduce ``zllrunning/face-parsing.PyTorch`` exactly, because the
published checkpoint is a state dict keyed by those names and it will not load
against anything else.

Kept private (leading underscore): callers use ``skinlib.parse``, which owns
preprocessing, the class-id mapping and the mask cleanup. Nothing here should
be imported directly.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ["BiSeNet"]


class ConvBNReLU(nn.Module):
    def __init__(self, in_chan: int, out_chan: int, ks: int = 3, stride: int = 1, padding: int = 1):
        super().__init__()
        self.conv = nn.Conv2d(
            in_chan, out_chan, kernel_size=ks, stride=stride, padding=padding, bias=False
        )
        self.bn = nn.BatchNorm2d(out_chan)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.relu(self.bn(self.conv(x)))


def _conv3x3(in_planes: int, out_planes: int, stride: int = 1) -> nn.Conv2d:
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)


class BasicBlock(nn.Module):
    def __init__(self, in_chan: int, out_chan: int, stride: int = 1):
        super().__init__()
        self.conv1 = _conv3x3(in_chan, out_chan, stride)
        self.bn1 = nn.BatchNorm2d(out_chan)
        self.conv2 = _conv3x3(out_chan, out_chan)
        self.bn2 = nn.BatchNorm2d(out_chan)
        self.relu = nn.ReLU(inplace=True)
        self.downsample: nn.Module | None = None
        if in_chan != out_chan or stride != 1:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_chan, out_chan, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_chan),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.relu(self.bn1(self.conv1(x)))
        residual = self.bn2(self.conv2(residual))
        shortcut = x if self.downsample is None else self.downsample(x)
        return self.relu(shortcut + residual)


def _create_layer_basic(in_chan: int, out_chan: int, bnum: int, stride: int = 1) -> nn.Sequential:
    layers = [BasicBlock(in_chan, out_chan, stride=stride)]
    layers += [BasicBlock(out_chan, out_chan, stride=1) for _ in range(bnum - 1)]
    return nn.Sequential(*layers)


class Resnet18(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = _create_layer_basic(64, 64, bnum=2, stride=1)
        self.layer2 = _create_layer_basic(64, 128, bnum=2, stride=2)
        self.layer3 = _create_layer_basic(128, 256, bnum=2, stride=2)
        self.layer4 = _create_layer_basic(256, 512, bnum=2, stride=2)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        x = self.maxpool(F.relu(self.bn1(self.conv1(x))))
        x = self.layer1(x)
        feat8 = self.layer2(x)  # 1/8
        feat16 = self.layer3(feat8)  # 1/16
        feat32 = self.layer4(feat16)  # 1/32
        return feat8, feat16, feat32


class AttentionRefinementModule(nn.Module):
    def __init__(self, in_chan: int, out_chan: int):
        super().__init__()
        self.conv = ConvBNReLU(in_chan, out_chan, ks=3, stride=1, padding=1)
        self.conv_atten = nn.Conv2d(out_chan, out_chan, kernel_size=1, bias=False)
        self.bn_atten = nn.BatchNorm2d(out_chan)
        self.sigmoid_atten = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.conv(x)
        atten = F.avg_pool2d(feat, feat.size()[2:])
        atten = self.sigmoid_atten(self.bn_atten(self.conv_atten(atten)))
        return torch.mul(feat, atten)


class ContextPath(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.resnet = Resnet18()
        self.arm16 = AttentionRefinementModule(256, 128)
        self.arm32 = AttentionRefinementModule(512, 128)
        self.conv_head32 = ConvBNReLU(128, 128, ks=3, stride=1, padding=1)
        self.conv_head16 = ConvBNReLU(128, 128, ks=3, stride=1, padding=1)
        self.conv_avg = ConvBNReLU(512, 128, ks=1, stride=1, padding=0)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        feat8, feat16, feat32 = self.resnet(x)
        h8, w8 = feat8.size()[2:]
        h16, w16 = feat16.size()[2:]
        h32, w32 = feat32.size()[2:]

        avg = self.conv_avg(F.avg_pool2d(feat32, feat32.size()[2:]))
        avg_up = F.interpolate(avg, (h32, w32), mode="nearest")

        feat32_sum = self.arm32(feat32) + avg_up
        feat32_up = self.conv_head32(F.interpolate(feat32_sum, (h16, w16), mode="nearest"))

        feat16_sum = self.arm16(feat16) + feat32_up
        feat16_up = self.conv_head16(F.interpolate(feat16_sum, (h8, w8), mode="nearest"))

        return feat8, feat16_up, feat32_up


class FeatureFusionModule(nn.Module):
    def __init__(self, in_chan: int, out_chan: int):
        super().__init__()
        self.convblk = ConvBNReLU(in_chan, out_chan, ks=1, stride=1, padding=0)
        self.conv1 = nn.Conv2d(out_chan, out_chan // 4, kernel_size=1, stride=1, padding=0, bias=False)
        self.conv2 = nn.Conv2d(out_chan // 4, out_chan, kernel_size=1, stride=1, padding=0, bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, fsp: torch.Tensor, fcp: torch.Tensor) -> torch.Tensor:
        feat = self.convblk(torch.cat([fsp, fcp], dim=1))
        atten = F.avg_pool2d(feat, feat.size()[2:])
        atten = self.sigmoid(self.conv2(self.relu(self.conv1(atten))))
        return feat + torch.mul(feat, atten)


class BiSeNetOutput(nn.Module):
    def __init__(self, in_chan: int, mid_chan: int, n_classes: int):
        super().__init__()
        self.conv = ConvBNReLU(in_chan, mid_chan, ks=3, stride=1, padding=1)
        self.conv_out = nn.Conv2d(mid_chan, n_classes, kernel_size=1, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv_out(self.conv(x))


class BiSeNet(nn.Module):
    """CelebAMask-HQ face parser. ``n_classes`` is 19 for the public checkpoint."""

    def __init__(self, n_classes: int = 19):
        super().__init__()
        self.cp = ContextPath()
        # The spatial path is the context path's 1/8 feature map in this
        # implementation; the checkpoint has no separate spatial-path weights.
        self.ffm = FeatureFusionModule(256, 256)
        self.conv_out = BiSeNetOutput(256, 256, n_classes)
        self.conv_out16 = BiSeNetOutput(128, 64, n_classes)
        self.conv_out32 = BiSeNetOutput(128, 64, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Returns the main logits only.

        The two auxiliary heads (``conv_out16``/``conv_out32``) exist to load
        the checkpoint intact but are deep-supervision outputs used during
        training; evaluating them here would be wasted compute.
        """
        height, width = x.size()[2:]
        feat_res8, feat_cp8, _feat_cp16 = self.cp(x)
        feat_fuse = self.ffm(feat_res8, feat_cp8)
        out = self.conv_out(feat_fuse)
        return F.interpolate(out, (height, width), mode="bilinear", align_corners=True)
