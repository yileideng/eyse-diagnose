import math
from typing import List
import torch
import torch.nn as nn


def _make_divisible(v, divisor=8):
    return int(math.ceil(v / divisor) * divisor)


class DropPath(nn.Module):
    def __init__(self, drop_prob: float = 0.0):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.training and self.drop_prob > 0.0:
            keep_prob = 1 - self.drop_prob
            shape = (x.shape[0],) + (1,) * (x.ndim - 1)
            random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
            random_tensor.floor_()
            x = x.div(keep_prob) * random_tensor
        return x


class PatchEmbed(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.GELU()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x)


class DownsampleLayer(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.layer = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(out_channels)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layer(x)


class FusedMBConv(nn.Module):
    def __init__(self, dim: int, expand_ratio: float = 4.0):
        super().__init__()
        hidden_dim = _make_divisible(dim * expand_ratio)
        self.block = nn.Sequential(
            nn.Conv2d(dim, hidden_dim, 1, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.GELU(),
            nn.Conv2d(hidden_dim, hidden_dim, 3, stride=1, padding=1, groups=hidden_dim, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.GELU(),
            nn.Conv2d(hidden_dim, dim, 1, bias=False),
            nn.BatchNorm2d(dim)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.block(x)


class FeedForward(nn.Module):
    def __init__(self, dim: int, mlp_ratio: float):
        super().__init__()
        hidden_dim = _make_divisible(dim * mlp_ratio)
        self.fc1 = nn.Linear(dim, hidden_dim)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_dim, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        return x


class AttentionBlock(nn.Module):
    def __init__(self, dim: int, num_heads: int, mlp_ratio: float, drop_path: float):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        self.drop_path = DropPath(drop_path)
        self.norm2 = nn.LayerNorm(dim)
        self.ffn = FeedForward(dim, mlp_ratio)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        x_flat = x.permute(0, 2, 3, 1).reshape(b, h * w, c)
        residual = x_flat
        x_flat = self.norm1(x_flat)
        attn_out, _ = self.attn(x_flat, x_flat, x_flat, need_weights=False)
        x_flat = residual + self.drop_path(attn_out)
        residual = x_flat
        x_flat = self.norm2(x_flat)
        x_flat = residual + self.drop_path(self.ffn(x_flat))
        x = x_flat.reshape(b, h, w, c).permute(0, 3, 1, 2)
        return x


class EfficientFormer(nn.Module):
    def __init__(
        self,
        depths: List[int],
        embed_dims: List[int],
        num_heads: List[int],
        attn_blocks: List[int],
        mlp_ratios: List[float],
        num_classes: int = 1000,
        drop_path_rate: float = 0.1
    ):
        super().__init__()
        self.num_classes = num_classes
        self.embed_dims = embed_dims
        self.num_features = embed_dims[-1]
        self.downsample_layers = nn.ModuleList()
        self.downsample_layers.append(PatchEmbed(3, embed_dims[0]))
        for i in range(1, len(embed_dims)):
            self.downsample_layers.append(DownsampleLayer(embed_dims[i - 1], embed_dims[i]))

        total_blocks = sum(depths)
        drop_rates = torch.linspace(0, drop_path_rate, total_blocks).tolist()
        self.stages = nn.ModuleList()
        block_idx = 0
        for stage_idx, depth in enumerate(depths):
            blocks = nn.ModuleList()
            dim = embed_dims[stage_idx]
            attn_start = depth - attn_blocks[stage_idx]
            for i in range(depth):
                if i < attn_start:
                    blocks.append(FusedMBConv(dim))
                else:
                    drop = drop_rates[block_idx + i]
                    blocks.append(AttentionBlock(dim, num_heads[stage_idx], mlp_ratios[stage_idx], drop))
            self.stages.append(blocks)
            block_idx += depth
        self.norm = nn.LayerNorm(embed_dims[-1])
        self.head = nn.Linear(embed_dims[-1], num_classes)

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        for down, stage in zip(self.downsample_layers, self.stages):
            x = down(x)
            for block in stage:
                x = block(x)
        return x

    def forward_head(self, x: torch.Tensor) -> torch.Tensor:
        x = x.mean(dim=(2, 3))
        x = self.norm(x)
        return self.head(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feats = self.forward_features(x)
        return self.forward_head(feats)


def efficientformer_l1(num_classes: int = 8, drop_path_rate: float = 0.1) -> EfficientFormer:
    return EfficientFormer(
        depths=[3, 2, 6, 4],
        embed_dims=[32, 48, 96, 176],
        num_heads=[1, 2, 4, 8],
        attn_blocks=[0, 0, 2, 4],
        mlp_ratios=[4.0, 4.0, 4.0, 4.0],
        num_classes=num_classes,
        drop_path_rate=drop_path_rate
    )


def efficientformer_l3(num_classes: int = 8, drop_path_rate: float = 0.2) -> EfficientFormer:
    return EfficientFormer(
        depths=[4, 4, 12, 6],
        embed_dims=[40, 64, 128, 256],
        num_heads=[1, 2, 4, 8],
        attn_blocks=[0, 0, 4, 6],
        mlp_ratios=[4.0, 4.0, 4.0, 4.0],
        num_classes=num_classes,
        drop_path_rate=drop_path_rate
    )


def efficientformer_l7(num_classes: int = 8, drop_path_rate: float = 0.3) -> EfficientFormer:
    return EfficientFormer(
        depths=[6, 6, 18, 8],
        embed_dims=[64, 96, 160, 288],
        num_heads=[1, 2, 5, 10],
        attn_blocks=[0, 0, 6, 8],
        mlp_ratios=[4.0, 4.0, 4.0, 4.0],
        num_classes=num_classes,
        drop_path_rate=drop_path_rate
    )


def count_params(model, trainable_only=False):
    if trainable_only:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    return sum(p.numel() for p in model.parameters())

if __name__ == "__main__":
    device = "cuda:0"

   
    model = efficientformer_l1().to(device=device)

    total = count_params(model)
    trainable = count_params(model, True)

    inputs = torch.rand(1, 3, 224, 448).to(device)
    out = model(inputs)
    print(out)
    print(out[0].size())
    print(f"Params: {total/1e6:.3f} M (all), {trainable/1e6:.3f} M (trainable)")
