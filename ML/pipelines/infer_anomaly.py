import argparse, os, os.path as osp, numpy as np, pandas as pd, cv2, torch
from torch.utils.data import DataLoader, Subset
from torchvision import transforms, models
from datasets.ODIR_Dataset import ODIR_Dataset
from anomaly_core import build_feature_bank, anomaly_map_from_bank
from tqdm import tqdm
from itertools import islice

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

def denorm_img(x: torch.Tensor):
    # x: (1,3,H,W) tensor in normalized space -> uint8 RGB image
    x = x[0].cpu().numpy().transpose(1, 2, 0)
    x = x * np.array(IMAGENET_STD)[None, None, :] + np.array(IMAGENET_MEAN)[None, None, :]
    x = np.clip(x, 0, 1)
    return (x * 255).astype(np.uint8)  # RGB

def save_overlay(rgb: np.ndarray, amap: np.ndarray, out_path: str, alpha: float = 0.5):
    # rgb: (H,W,3) uint8 in RGB; amap: (H,W) float in [0,1]
    heat = cv2.applyColorMap((amap * 255).astype(np.uint8), cv2.COLORMAP_JET)
    overlay = cv2.addWeighted(rgb[..., ::-1], alpha, heat, 1 - alpha, 0)  # RGB->BGR for cv2
    cv2.imwrite(out_path, overlay)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--img_dir', default='C:/Code/Fc25_07/Training_data', help='Directory of ODIR images')
    ap.add_argument('--label_csv', default='C:/Code/Fc25_07/fuchuang_uestc/total_data.csv', help='CSV with labels; must contain columns: N,D,G,C,A,H,M,O')
    ap.add_argument('--out_dir', default='runs/unsup_loc', help='Where to save overlays and (optional) graymaps')
    ap.add_argument('--batch', type=int, default=8)
    ap.add_argument('--coreset', type=float, default=0.10, help='Fraction of spatial features to keep for the bank')
    ap.add_argument('--downsample', type=int, default=2, help='AvgPool factor for features to save memory (1/2/4)')
    ap.add_argument('--k', type=int, default=5, help='Top-k neighbor average for anomaly score')
    ap.add_argument('--num_visual', type=int, default=200, help='Limit number of saved images; -1 for all')
    ap.add_argument('--seed', type=int, default=42)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    torch.manual_seed(args.seed); np.random.seed(args.seed)

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])

    # Full dataset
    ds = ODIR_Dataset(img_dir=args.img_dir, label_dir=args.label_csv,
                      transform=transform, use_resampling=False)

    # Build indices for NORMAL samples: N==1 and others==0
    # ODIR label order is [N,D,G,C,A,H,M,O]
    if hasattr(ds, 'labels'):
        labels = ds.labels.cpu().numpy()
    else:
        df = pd.read_csv(args.label_csv)
        labels = df[['N','D','G','C','A','H','M','O']].values.astype(np.float32)

    normal_mask = (labels[:,0] == 1) & (labels[:,1:].sum(axis=1) == 0)
    normal_idxs = np.where(normal_mask)[0].tolist()
    if len(normal_idxs) == 0:
        raise RuntimeError("No pure NORMAL samples found to build the feature bank. Check your CSV columns.")

    normal_set = Subset(ds, normal_idxs)
    normal_loader = DataLoader(normal_set, batch_size=args.batch, shuffle=False, num_workers=0, pin_memory=True)

    # Backbone: torchvision DenseNet121.features
    try:
        backbone = models.densenet121(weights=models.DenseNet121_Weights.IMAGENET1K_V1).features
    except Exception:
        backbone = models.densenet121(pretrained=True).features
    backbone.eval()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Build or reuse bank
    bank_path = osp.join(args.out_dir, 'feature_bank.pt')
    if osp.exists(bank_path):
        ckpt = torch.load(bank_path, map_location=device)
        bank = ckpt['bank'].to(device).float()
        meta = ckpt.get('meta', {'downsample': args.downsample})
        print(f"[Bank] Reused saved bank from {bank_path} | shape={tuple(bank.shape)}")
    else:
        print(f"[Bank] Using {len(normal_idxs)} NORMAL images…")
        bank, meta = build_feature_bank(backbone, normal_loader, device,
                                        coreset_ratio=args.coreset, downsample=args.downsample)
    print(f"[Bank] Using {len(normal_idxs)} NORMAL images…")
    bank, meta = build_feature_bank(backbone, normal_loader, device,
                                    coreset_ratio=args.coreset, downsample=args.downsample)

    # Inference on the whole dataset (or a subset for speed)
    loader = DataLoader(ds, batch_size=1, shuffle=False, num_workers=0, pin_memory=True)
    max_to_save = len(loader) if args.num_visual < 0 else min(args.num_visual, len(loader))

    for i, batch in tqdm(enumerate(islice(loader, max_to_save)),
                     total=max_to_save,
                     desc="[Infer] saving overlays",
                     ncols=100):
        x = batch[0] if isinstance(batch, (list, tuple)) else batch
        amap = anomaly_map_from_bank(backbone, x, bank, k=args.k, downsample=args.downsample)
        rgb = denorm_img(x)
        save_overlay(rgb, amap, osp.join(args.out_dir, f"{i:05d}.jpg"))


    # Optionally save bank for reuse
    # torch.save({"bank": bank.half().cpu(), "meta": meta}, osp.join(args.out_dir, "feature_bank.pt"))
    # print(f"[Done] Saved {n_save} overlays to {args.out_dir}")

if __name__ == "__main__":
    main()
