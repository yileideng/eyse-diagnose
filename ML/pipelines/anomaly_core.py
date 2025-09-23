from tqdm import tqdm
import math
from typing import Iterable, Optional, Tuple
import torch
import torch.nn.functional as F

@torch.no_grad()
def _chunked_cdist(a: torch.Tensor, b: torch.Tensor, chunk: int = 4096) -> torch.Tensor:

    assert a.dim() == 2 and b.dim() == 2 and a.size(1) == b.size(1), "Shapes must be (M,d) and (N,d)"
    M = a.size(0)
    outs = []
    for i in tqdm(range(0, M, chunk), total=math.ceil(M/chunk), desc="[CDist]", ncols=100, leave=False):
        aa = a[i:i+chunk]
        outs.append(torch.cdist(aa, b))
    return torch.cat(outs, dim=0)

@torch.no_grad()
def build_feature_bank(
    backbone: torch.nn.Module,
    normal_loader: Iterable,
    device: torch.device,
    coreset_ratio: float = 0.1,
    downsample: int = 2,
    take_layers: Optional[Tuple[int, ...]] = None,
):

    backbone.eval().to(device)
    feats_all = []
    for batch in tqdm(normal_loader, total=len(normal_loader), desc="[bank]bulid", ncols=100):
        x = batch[0] if isinstance(batch, (list, tuple)) else batch
        x = x.to(device, non_blocking=True)
        f = backbone(x)
        # If model returns a list of feature maps (e.g., feature pyramid), pick & upsample/concat
        if isinstance(f, (list, tuple)):
            selected = [f[i] for i in (take_layers if take_layers is not None else range(len(f)))]
            # unify spatial via interpolation to largest map, then cat along C
            hs = [t.shape[-2] for t in selected]
            ws = [t.shape[-1] for t in selected]
            Ht, Wt = max(hs), max(ws)
            selected = [F.interpolate(t, size=(Ht, Wt), mode='bilinear', align_corners=False) for t in selected]
            f = torch.cat(selected, dim=1)
        if downsample and downsample > 1:
            f = F.avg_pool2d(f, downsample, downsample)
        # (B,C,h,w) -> (B*h*w,C)
        B, C, h, w = f.shape
        feats_all.append(f.permute(0, 2, 3, 1).reshape(-1, C).cpu())
    bank = torch.cat(feats_all, dim=0)
    # random coreset
    keep = max(1, int(len(bank) * coreset_ratio))
    perm = torch.randperm(len(bank))[:keep]
    bank = bank[perm].to(device)
    meta = {"downsample": downsample}
    return bank, meta

@torch.no_grad()
def anomaly_map_from_bank(
    backbone: torch.nn.Module,
    x: torch.Tensor,
    bank: torch.Tensor,
    k: int = 5,
    downsample: int = 2,
    chunk: int = 8192,
):

    device = bank.device
    backbone.eval().to(device)
    x = x.to(device, non_blocking=True)
    f = backbone(x)
    if isinstance(f, (list, tuple)):
        # default: use the last feature map
        f = f[-1]
    if downsample and downsample > 1:
        f = F.avg_pool2d(f, downsample, downsample)
    _, C, h, w = f.shape
    q = f.permute(0, 2, 3, 1).reshape(-1, C)  # (h*w, C)
    dmat = _chunked_cdist(q.float(), bank.float(), chunk=chunk)  # (h*w, K)
    topk = torch.topk(dmat, k=min(k, dmat.size(1)), dim=1).values.mean(1)  # (h*w,)
    amap = topk.view(h, w)
    # normalize to [0,1]
    amap = (amap - amap.min()) / (amap.max() - amap.min() + 1e-8)
    # upsample back to input size
    amap = F.interpolate(amap.unsqueeze(0).unsqueeze(0), size=x.shape[2:], mode='bilinear', align_corners=False)
    return amap.squeeze().detach().cpu().numpy()
