import math
from typing import List, Optional

import torch
import torch.nn as nn
import timm


class EfficientViTMultiLabel(nn.Module):
    def __init__(
        self,
        backbone: str = "lightweight",
        num_classes: int = 8,
        pretrained: bool = True,
        dropout: float = 0.0,
        img_size: Optional[tuple] = None,
        light_width_mult: float = 0.75
    ):
        super().__init__()
        if img_size is None:
            img_size = (512, 1024)
        if backbone.startswith('efficientformer'):
            from model import efficientformer as local_eff
            factory = {
                'efficientformer_l1': local_eff.efficientformer_l1,
                'efficientformer_l3': local_eff.efficientformer_l3,
                'efficientformer_l7': local_eff.efficientformer_l7,
            }
            if backbone not in factory:
                raise ValueError(f'Unsupported efficientformer variant: {backbone}')
            self.backbone = factory[backbone](num_classes=num_classes, drop_path_rate=0.1)
        elif backbone == 'lightweight':
            from model.lightweight_backbone import LightweightBackbone
            self.backbone = LightweightBackbone(img_size=img_size, width_mult=light_width_mult)
        else:
            self.backbone = timm.create_model(backbone, pretrained=pretrained, num_classes=0, global_pool="")
        self.num_features = getattr(self.backbone, 'num_features', None)
        if self.num_features is None:
            raise ValueError('Backbone must expose num_features attribute')
        self.head = nn.Sequential(
            nn.Conv2d(self.num_features, self.num_features, 1, bias=False),
            nn.BatchNorm2d(self.num_features),
            nn.GELU(),
        )
        self.attn_map = nn.Sequential(
            nn.Conv2d(self.num_features, self.num_features // 2, 1),
            nn.GELU(),
            nn.Conv2d(self.num_features // 2, num_classes, 1),
            nn.Sigmoid(),
        )
        self.attn_head = nn.Linear(self.num_features, num_classes)
        self.eye_head = nn.Linear(self.num_features, num_classes)
        self.temperature = nn.Parameter(torch.tensor(1.0))
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        if hasattr(self.backbone, 'forward_features'):
            feat = self.backbone.forward_features(x)
        else:
            feat = self.backbone(x)
        return feat

    def forward(self, x: torch.Tensor, return_maps: bool = False):
        feat = self.forward_features(x)
        feat = self.head(feat)
        pooled = feat.mean(dim=(2, 3))
        logits_gap = self.dropout(pooled)
        logits_gap = self.attn_head(logits_gap)
        maps = self.attn_map(feat)
        maps_flat = maps.flatten(2)
        weights_sum = maps_flat.sum(-1, keepdim=True).clamp_min(1e-6)
        weights = maps_flat / weights_sum
        feat_flat = feat.flatten(2).transpose(1, 2)
        attn_feat = torch.bmm(weights, feat_flat)
        logits_attn = self.attn_head(attn_feat.mean(dim=1))
        mid = feat.shape[-1] // 2
        left_feat = feat[:, :, :, :mid].mean(dim=(2, 3))
        right_feat = feat[:, :, :, mid:].mean(dim=(2, 3))
        eye_left = self.eye_head(left_feat)
        eye_right = self.eye_head(right_feat)
        eye_logits = torch.max(torch.stack([eye_left, eye_right], dim=1), dim=1).values
        logits = logits_gap + logits_attn + 0.2 * eye_logits
        logits = logits * self.temperature.clamp_min(0.5)
        if return_maps:
            return logits, maps
        return logits

def count_params(model, trainable_only=False):
    if trainable_only:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    return sum(p.numel() for p in model.parameters())

if __name__ == "__main__":
    device = "cuda:0"

   
    model = EfficientViTMultiLabel().to(device=device)

    total = count_params(model)
    trainable = count_params(model, True)

    inputs = torch.rand(1, 3, 224, 448).to(device)
    out = model(inputs)
    print(out)
    print(out[0].size())
    print(f"Params: {total/1e6:.3f} M (all), {trainable/1e6:.3f} M (trainable)")
