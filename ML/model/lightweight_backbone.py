import torch
import torch.nn as nn
from typing import Tuple


def _make_divisible(v: int, divisor: int = 8) -> int:
    return int((v + divisor - 1) // divisor * divisor)


class SEModule(nn.Module):
    def __init__(self, channels: int, reduction: int = 4):
        super().__init__()
        hidden = max(channels // reduction, 4)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(channels, hidden, 1, bias=True),
            nn.SiLU(),
            nn.Conv2d(hidden, channels, 1, bias=True)
        )
        self.act = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        scale = self.avg_pool(x)
        scale = self.fc(scale)
        return x * self.act(scale)


class DepthwiseSeparableBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int = 1,
        expand_ratio: float = 2.0,
        use_se: bool = True
    ):
        super().__init__()
        hidden = _make_divisible(int(in_channels * expand_ratio))
        self.use_residual = stride == 1 and in_channels == out_channels
        self.depthwise = nn.Conv2d(in_channels, in_channels, 3, stride=stride, padding=1, groups=in_channels, bias=False)
        self.dw_bn = nn.BatchNorm2d(in_channels)
        self.dw_act = nn.SiLU()
        self.pointwise_exp = nn.Sequential(
            nn.Conv2d(in_channels, hidden, 1, bias=False),
            nn.BatchNorm2d(hidden),
            nn.SiLU()
        )
        self.se = SEModule(hidden) if use_se else nn.Identity()
        self.pointwise_proj = nn.Sequential(
            nn.Conv2d(hidden, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels)
        )
        self.out_act = nn.SiLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.depthwise(x)
        x = self.dw_bn(x)
        x = self.dw_act(x)
        x = self.pointwise_exp(x)
        x = self.se(x)
        x = self.pointwise_proj(x)
        if self.use_residual:
            x = x + residual
        x = self.out_act(x)
        return x


class LightweightBackbone(nn.Module):
    def __init__(
        self,
        img_size: Tuple[int, int] = (512, 1024),
        dims: Tuple[int, int, int, int] = (48, 96, 160, 224),
        depths: Tuple[int, int, int, int] = (2, 2, 3, 2),
        expand_ratio: float = 2.0,
        width_mult: float = 1.0,
        use_se: bool = True
    ):
        super().__init__()
        dims = tuple(_make_divisible(int(d * width_mult), 8) for d in dims)
        self.stem = nn.Sequential(
            nn.Conv2d(3, dims[0], kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(dims[0]),
            nn.SiLU()
        )
        self.stages = nn.ModuleList()
        in_channels = dims[0]
        for stage_idx, (out_channels, depth) in enumerate(zip(dims, depths)):
            blocks = []
            for block_idx in range(depth):
                stride = 1
                if stage_idx > 0 and block_idx == 0:
                    stride = 2
                blocks.append(
                    DepthwiseSeparableBlock(
                        in_channels if block_idx == 0 else out_channels,
                        out_channels,
                        stride=stride,
                        expand_ratio=expand_ratio,
                        use_se=use_se
                    )
                )
                in_channels = out_channels
            self.stages.append(nn.Sequential(*blocks))
        self.final_conv = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.SiLU()
        )
        self.num_features = in_channels

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        for stage in self.stages:
            x = stage(x)
        x = self.final_conv(x)
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward_features(x)


def lightweight_backbone(width_mult: float = 1.0) -> LightweightBackbone:
    return LightweightBackbone(width_mult=width_mult)
