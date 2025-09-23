import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
import numpy as np

from pdb import set_trace as stop
from model.transformer_layers import SelfAttnLayer
from model.utils import custom_replace, weights_init
from model.position_enc import PositionEmbeddingSine, positionalencoding2d


class CTranModel(nn.Module):
    def __init__(
        self,
        num_labels,
        use_lmt,
        device,
        backbone_model,
        pos_emb=False,
        layers=3,
        heads=4,
        dropout=0.6,
        int_loss=0,
        no_x_features=False,
        grad_cam=False,
    ):
        super().__init__()

        self.use_lmt = use_lmt
        self.no_x_features = no_x_features
        self.backbone = backbone_model

        self.hidden = hidden = 1024

        self.downsample = False
        if self.downsample:
            self.conv_downsample = nn.Conv2d(hidden, hidden, kernel_size=1)

        self.register_buffer("label_ids", torch.arange(num_labels).long())
        self.label_lt = nn.Embedding(num_labels, hidden)
        self.known_label_lt = nn.Embedding(3, hidden, padding_idx=0)

        self.use_pos_enc = pos_emb
        self.pos_sine = PositionEmbeddingSine(num_pos_feats=hidden // 2, normalize=True) if self.use_pos_enc else None

        self.self_attn_layers = nn.ModuleList([SelfAttnLayer(hidden, heads, dropout) for _ in range(layers)])
        self.output_linear = nn.Linear(hidden, num_labels)

        self.LayerNorm = nn.LayerNorm(hidden)
        self.dropout = nn.Dropout(dropout)
        self.grad_cam = grad_cam

        self.label_lt.apply(weights_init)
        self.known_label_lt.apply(weights_init)
        self.LayerNorm.apply(weights_init)
        self.self_attn_layers.apply(weights_init)
        self.output_linear.apply(weights_init)

        self.device = torch.device(device) if not isinstance(device, torch.device) else device

    def forward(
        self,
        images,
        mask=None,
        grad_cam: bool = False,
        apply_health_constraint: bool = True,
        return_feat: bool = False,
    ):
        B = images.size(0)

        const_label_input = self.label_ids.unsqueeze(0).expand(B, -1).to(images.device)      # (B, L)
        init_label_embeddings = self.label_lt(const_label_input)                               # (B, L, C)

        if self.use_lmt:
            if mask is None:
                mask = torch.zeros(B, const_label_input.size(1), device=images.device)
                if self.training:
                    for i in range(B):
                        r = torch.empty(1, device=images.device).uniform_(0.25, 1.0).item()
                        k = int(mask.size(1) * r)
                        idx = torch.randperm(mask.size(1), device=images.device)[:k]
                        mask[i, idx] = 1
            label_feat_vec = custom_replace(mask, 0, 1, 2).long()                              # (B, L)
            state_embeddings = self.known_label_lt(label_feat_vec)                             # (B, L, C)
            init_label_embeddings = init_label_embeddings + state_embeddings                   # (B, L, C)

        features = self.backbone(images)                                                       # (B, C, h, w)
        if self.downsample:
            features = self.conv_downsample(features)                                          # (B, C, h, w)

        if self.use_pos_enc:
            pad_mask = torch.zeros(B, features.shape[-2], features.shape[-1], dtype=torch.bool, device=features.device)
            pos_encoding = self.pos_sine(pad_mask).to(features.dtype)                          # (B, C, h, w)
            features = features + pos_encoding

        original_features = features                                                           # (B, C, h, w)

        feats_seq = features.flatten(2).permute(0, 2, 1)                                       # (B, hw, C)
        embeddings = init_label_embeddings if self.no_x_features else torch.cat([feats_seq, init_label_embeddings], dim=1)  # (B, hw+L, C)
        embeddings = self.LayerNorm(embeddings)                                                # (B, hw+L, C)

        attn_list = []
        for layer in self.self_attn_layers:
            embeddings, attn = layer(embeddings, mask=None)                                    # (B, hw+L, C), (heads, hw+L, hw+L)
            attn_list.append(attn.detach())

        label_embeddings = embeddings[:, -init_label_embeddings.size(1):, :]                   # (B, L, C)
        out_full = self.output_linear(label_embeddings)                                        # (B, L, L)
        output = out_full.diagonal(dim1=1, dim2=2)                                            # (B, L)

        if apply_health_constraint:
            health_mask = (output[:, 0] > 0.5).unsqueeze(1)                                   # (B, 1)
            output = torch.where(
                health_mask,
                torch.cat([output[:, :1], torch.zeros_like(output[:, 1:])], dim=1),           # (B, L)
                output,
            )

        if grad_cam or return_feat:
            return output, original_features                                                   # (B, L), (B, C, h, w)
        return output                                                                          # (B, L)

    def get_last_conv_layer(self):
        last = None
        for m in self.backbone.modules():
            if isinstance(m, nn.Conv2d):
                last = m
        if last is None:
            raise RuntimeError("No Conv2d found in backbone")
        return last

    @torch.no_grad()
    def extract_backbone_feature(self, images):
        self.eval()
        return self.backbone(images)                                                           # (B, C, h, w)


def count_params(model, trainable_only=False):
    if trainable_only:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    return sum(p.numel() for p in model.parameters())


if __name__ == "__main__":
    device = "cuda:0"

    try:
        backbone = models.densenet121(weights=models.DenseNet121_Weights.IMAGENET1K_V1).features
    except Exception:
        backbone = models.densenet121(pretrained=True).features

    model = CTranModel(num_labels=8, use_lmt=True, device=device, backbone_model=backbone, pos_emb=False).to(device=device)

    total = count_params(model)
    trainable = count_params(model, True)

    inputs = torch.rand(1, 3, 224, 448).to(device)
    out = model(inputs)
    print(out)
    print(out[0].size())
    print(f"Params: {total/1e6:.3f} M (all), {trainable/1e6:.3f} M (trainable)")
