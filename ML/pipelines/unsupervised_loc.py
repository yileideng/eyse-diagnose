import argparse
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
import yaml
from PIL import Image, ImageDraw, ImageFont
from iterstrat.ml_stratifiers import MultilabelStratifiedShuffleSplit
from torch.utils.data import DataLoader, Subset
from torchvision import models
from tqdm import tqdm

from datasets import ODIRDataset
from model.Ctran import CTranModel
from model.grad_cam import GradCAM
from pipelines.anomaly.anomaly_core import build_feature_bank, anomaly_map_from_bank

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
CLASSES = ['N', 'D', 'G', 'C', 'A', 'H', 'M', 'O']
CLASS_LABELS_ZH = {
    'N': '正常',
    'D': '糖尿病',
    'G': '青光眼',
    'C': '白内障',
    'A': '黄斑变性',
    'H': '高血压相关',
    'M': '近视相关',
    'O': '其他'
}

CLASS_LABELS_EN = {
    'N': 'Normal',
    'D': 'Diabetic',
    'G': 'Glaucoma',
    'C': 'Cataract',
    'A': 'AMD',
    'H': 'Hypertensive',
    'M': 'Myopic',
    'O': 'Other'
}

FONT_CANDIDATE_PATHS = [
    'C:/Windows/Fonts/msyh.ttc',
    'C:/Windows/Fonts/simhei.ttf',
    '/System/Library/Fonts/STHeiti Light.ttc',
    '/System/Library/Fonts/STHeiti Medium.ttc',
    '/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.otf',
    '/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttf'
]

def load_chinese_font(size: int = 24):
    for candidate in FONT_CANDIDATE_PATHS:
        font_path = Path(candidate)
        if font_path.exists():
            try:
                return ImageFont.truetype(str(font_path), size)
            except OSError:
                continue
    return None

def annotate_overlay_with_labels(bgr_img: np.ndarray, classes, font_size: int = 24) -> np.ndarray:
    if not classes:
        return bgr_img
    font = load_chinese_font(font_size)
    use_chinese = font is not None
    if not use_chinese:
        font = ImageFont.load_default()
    if use_chinese:
        texts = [f"{CLASS_LABELS_ZH.get(name, name)} ({name}): {prob:.2f}" for name, prob in classes]
    else:
        texts = [f"{CLASS_LABELS_EN.get(name, name)} ({name}): {prob:.2f}" for name, prob in classes]
    img = Image.fromarray(cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img)
    x, y = 16, 32
    for text in texts:
        if hasattr(draw, 'textbbox'):
            bbox = draw.textbbox((x, y), text, font=font)
            text_w = bbox[2] - bbox[0]
            text_h = bbox[3] - bbox[1]
        else:
            text_w, text_h = font.getsize(text)
            bbox = (x, y, x + text_w, y + text_h)
        pad = 6
        draw.rectangle([bbox[0] - pad, bbox[1] - pad, bbox[2] + pad, bbox[3] + pad], fill=(0, 0, 0, 180))
        draw.text((x, y), text, font=font, fill=(255, 255, 255))
        y = bbox[3] + pad + 6
    return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)




def set_seed(seed: int):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def create_splits(labels, train_ratio, val_ratio, seed):
    test_ratio = max(0.0, 1.0 - train_ratio - val_ratio)
    msss = MultilabelStratifiedShuffleSplit(n_splits=1, test_size=val_ratio + test_ratio, random_state=seed)
    idx = np.arange(len(labels))
    train_idx, temp_idx = next(msss.split(idx, labels))
    if test_ratio <= 0:
        val_idx = temp_idx
        test_idx = None
    else:
        val_size = val_ratio / (val_ratio + test_ratio)
        msss_val = MultilabelStratifiedShuffleSplit(n_splits=1, test_size=val_size, random_state=seed + 1)
        val_sub_idx, test_sub_idx = next(msss_val.split(temp_idx, labels[temp_idx]))
        val_idx = temp_idx[val_sub_idx]
        test_idx = temp_idx[test_sub_idx]
    return train_idx, val_idx, test_idx


def denorm(x: torch.Tensor):
    mean = torch.tensor(IMAGENET_MEAN, device=x.device).view(1, 3, 1, 1)
    std = torch.tensor(IMAGENET_STD, device=x.device).view(1, 3, 1, 1)
    x = x * std + mean
    x = x.clamp(0, 1)
    return x[0].permute(1, 2, 0).cpu().numpy()


def normalize_map(map_array: np.ndarray) -> np.ndarray:
    m = map_array.astype(np.float32)
    m = m - m.min()
    max_val = m.max()
    if max_val <= 1e-6:
        return np.zeros_like(m)
    return m / max_val


def threshold_heatmap(heatmap: np.ndarray, thr: float):
    hm = heatmap.astype(np.float32)
    active = hm >= thr
    scaled = np.zeros_like(hm)
    if np.any(active):
        active_vals = hm[active]
        min_active = active_vals.min()
        max_active = active_vals.max()
        if max_active - min_active <= 1e-6:
            scaled[active] = 1.0
        else:
            scaled[active] = (active_vals - min_active) / (max_active - min_active)
    return scaled, active.astype(np.uint8)


def overlay_heatmap(img, amap, mask=None, alpha=0.5):
    heat = cv2.applyColorMap((np.clip(amap, 0, 1) * 255).astype(np.uint8), cv2.COLORMAP_JET)
    blended = cv2.addWeighted(img, 1 - alpha, heat, alpha, 0)
    if mask is not None:
        mask_arr = mask.astype(np.float32)
        if mask_arr.ndim == 2:
            mask_arr = mask_arr[..., None]
        mask_arr = np.clip(mask_arr, 0.0, 1.0)
        out = img.astype(np.float32)
        blended = blended.astype(np.float32)
        out = out * (1.0 - mask_arr) + blended * mask_arr
        return out.astype(np.uint8)
    return blended


def build_retina_mask(bgr: np.ndarray, max_regions: int = 2) -> np.ndarray:
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (9, 9), 0)
    _, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, np.ones((11, 11), np.uint8))
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, np.ones((7, 7), np.uint8))
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    mask = np.zeros_like(gray, dtype=np.uint8)
    if contours:
        contours = sorted(contours, key=cv2.contourArea, reverse=True)[:max_regions]
        for cnt in contours:
            cv2.drawContours(mask, [cnt], -1, 1, thickness=cv2.FILLED)
    else:
        h, w = gray.shape
        radius = min(h, w) // 2
        cv2.circle(mask, (w // 2, h // 2), radius, 1, thickness=-1)
    return mask

def crop_to_retina(bgr: np.ndarray, margin_ratio: float = 0.05):
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    _, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return bgr, np.ones_like(gray, dtype=np.uint8)
    contour = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(contour)
    cx = x + w // 2
    cy = y + h // 2
    radius = int(max(w, h) * (0.5 + margin_ratio))
    h_img, w_img = gray.shape
    x0 = max(cx - radius, 0)
    y0 = max(cy - radius, 0)
    x1 = min(cx + radius, w_img)
    y1 = min(cy + radius, h_img)
    cropped = bgr[y0:y1, x0:x1]
    mask = np.zeros_like(gray)
    cv2.circle(mask, (cx, cy), radius, 1, thickness=-1)
    cropped_mask = mask[y0:y1, x0:x1]
    return cropped, cropped_mask
def prepare_eye_image(bgr: np.ndarray, target_shape, margin_ratio: float = 0.05):
    target_h, target_w = target_shape
    cropped_img, cropped_mask = crop_to_retina(bgr, margin_ratio=margin_ratio)
    resized_img = cv2.resize(cropped_img, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
    resized_mask = cv2.resize(cropped_mask.astype(np.uint8), (target_w, target_h), interpolation=cv2.INTER_NEAREST)
    resized_mask = (resized_mask > 0).astype(np.uint8)
    return resized_img, resized_mask


def prepare_eye_pair(left_img: np.ndarray, right_img: np.ndarray, target_h: int, target_w: int, margin_ratio: float = 0.05):
    left_w = target_w // 2
    right_w = target_w - left_w
    left_prepped, left_mask = prepare_eye_image(left_img, (target_h, left_w), margin_ratio=margin_ratio)
    right_prepped, right_mask = prepare_eye_image(right_img, (target_h, right_w), margin_ratio=margin_ratio)
    combined_img = np.concatenate([left_prepped, right_prepped], axis=1)
    combined_mask = np.concatenate([left_mask, right_mask], axis=1)
    return combined_img, combined_mask


class BackboneWrapper(torch.nn.Module):
    def __init__(self, backbone):
        super().__init__()
        self.backbone = backbone

    def forward(self, x):
        return self.backbone(x)


def load_ctrans_model(checkpoint: Path, num_labels: int, device: torch.device) -> CTranModel:
    try:
        backbone = models.densenet121(weights=models.DenseNet121_Weights.IMAGENET1K_V1).features
    except AttributeError:
        backbone = models.densenet121(pretrained=True).features
    backbone = backbone.to(device)
    model = CTranModel(num_labels=num_labels, use_lmt=True, device=device, backbone_model=backbone, pos_emb=False)
    state = torch.load(checkpoint, map_location=device)
    if isinstance(state, dict) and 'model' in state:
        state = state['model']
    elif isinstance(state, dict) and 'state_dict' in state:
        state = state['state_dict']
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing:
        print(f"Warning: missing keys when loading classifier: {missing}")
    if unexpected:
        print(f"Warning: unexpected keys when loading classifier: {unexpected}")
    model = model.to(device)
    model.eval()
    return model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='configs/classifier.yaml')
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--split', type=str, default='val', choices=['train', 'val', 'test'])
    parser.add_argument('--output', type=str, default='work/unsup_loc')
    parser.add_argument('--coreset', type=float, default=0.1)
    parser.add_argument('--downsample', type=int, default=2)
    parser.add_argument('--k', type=int, default=5)
    parser.add_argument('--prob_thr', type=float, default=0.4)
    parser.add_argument('--heat_thr', type=float, default=0.6)
    parser.add_argument('--overlay_alpha', type=float, default=0.5)
    parser.add_argument('--smooth_sigma', type=float, default=1.5)
    parser.add_argument('--max_labels', type=int, default=3)
    parser.add_argument('--crop_margin', type=float, default=0.05)
    args = parser.parse_args()

    with open(args.config, 'r', encoding='utf-8') as f:
        cfg = yaml.safe_load(f)
    data_cfg = cfg['data']
    df = pd.read_csv(data_cfg['csv_path'])
    id_to_files = {str(row['ID']): (row['Left-Fundus'], row['Right-Fundus']) for _, row in df.iterrows()}
    labels = df[df.columns[5:13]].values
    seed = data_cfg.get('seed', 42)
    set_seed(seed)
    splits = create_splits(labels, data_cfg['train_ratio'], data_cfg['val_ratio'], seed)
    if args.split == 'train':
        target_idx = splits[0]
    elif args.split == 'val':
        target_idx = splits[1]
    else:
        target_idx = splits[2] if splits[2] is not None else splits[1]
    base_kwargs = dict(img_dir=data_cfg['img_dir'], csv_path=data_cfg['csv_path'])
    dataset = ODIRDataset(stage='val', preprocess=True, cache_dir=data_cfg.get('cache_dir'), **base_kwargs)
    mask = np.isin(dataset.indices, target_idx)
    subset = Subset(dataset, np.arange(len(dataset))[mask])
    num_workers = cfg.get('logging', {}).get('num_workers', 4)
    loader = DataLoader(subset, batch_size=1, shuffle=False, num_workers=num_workers, pin_memory=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    classifier = load_ctrans_model(Path(args.checkpoint), labels.shape[1], device)
    cammer = GradCAM(classifier, classifier.get_last_conv_layer())

    classifier.backbone.eval()
    feature_model = BackboneWrapper(classifier.backbone).to(device)

    normal_idx = splits[0][labels[splits[0]].sum(axis=1) == 0]
    if len(normal_idx) == 0:
        sums = labels[splits[0]].sum(axis=1)
        order = np.argsort(sums)
        keep = splits[0][order[: max(10, len(order) // 4)]]
    else:
        keep = normal_idx
    bank_dataset = ODIRDataset(stage='val', preprocess=True, cache_dir=data_cfg.get('cache_dir'), **base_kwargs)
    bank_mask = np.isin(bank_dataset.indices, keep)
    bank_subset = Subset(bank_dataset, np.arange(len(bank_dataset))[bank_mask])
    bank_loader = DataLoader(bank_subset, batch_size=2, shuffle=False, num_workers=num_workers, pin_memory=True)
    bank, _ = build_feature_bank(feature_model, bank_loader, device, coreset_ratio=args.coreset, downsample=args.downsample)

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    for idx, (inputs, targets, sample_ids) in enumerate(tqdm(loader, total=len(loader))):
        inputs = inputs.to(device)
        with torch.no_grad():
            logits = classifier(inputs, apply_health_constraint=False)
        probs = torch.sigmoid(logits).cpu().numpy()[0]

        amap = anomaly_map_from_bank(feature_model, inputs, bank, k=args.k, downsample=args.downsample)
        amap = (amap - amap.min()) / (amap.max() - amap.min() + 1e-6)
        rgb = denorm(inputs)
        left_fname, right_fname = id_to_files.get(str(sample_ids[0]), (None, None))
        raw_combined = None
        retina_mask = None
        if left_fname is not None and right_fname is not None:
            left_path = Path(data_cfg['img_dir']) / left_fname
            right_path = Path(data_cfg['img_dir']) / right_fname
            left_img = cv2.imread(str(left_path))
            right_img = cv2.imread(str(right_path))
            if left_img is not None and right_img is not None:
                target_h = int(cfg['model'].get('img_height', 512))
                target_w = int(cfg['model'].get('img_width', 1024))
                raw_combined, retina_mask = prepare_eye_pair(left_img, right_img, target_h, target_w, margin_ratio=args.crop_margin)
        if raw_combined is None:
            raw_combined = (rgb * 255).astype(np.uint8)[..., ::-1]
        if retina_mask is None or retina_mask.shape[:2] != raw_combined.shape[:2]:
            retina_mask = build_retina_mask(raw_combined)
        retina_mask = (retina_mask > 0).astype(np.uint8)
        bgr = raw_combined.copy()

        class_cams = {}
        heat_map = np.zeros_like(amap, dtype=np.float32)
        activated_classes = []
        for c, name in enumerate(CLASSES):
            if name == 'N':
                continue
            prob = float(probs[c])
            if prob < args.prob_thr:
                continue
            if c not in class_cams:
                cam_map = cammer.generate_cam(inputs.detach().clone(), target_class=c)
                cam_map = normalize_map(cam_map)
                class_cams[c] = cam_map
            else:
                cam_map = class_cams[c]
            fused = 0.4 * amap + 0.6 * cam_map
            heat_map = np.maximum(heat_map, fused * prob)
            activated_classes.append((name, prob))

        if heat_map.max() <= 1e-6:
            heat_map = amap.copy()

        if args.smooth_sigma > 0:
            heat_map = cv2.GaussianBlur(heat_map, (0, 0), args.smooth_sigma)
        heat_map = normalize_map(heat_map)

        retina_norm = retina_mask.astype(np.float32)
        if retina_norm.max() > 0:
            retina_norm = retina_norm / retina_norm.max()
        heat_map *= retina_norm

        activated_map, binary_mask = threshold_heatmap(heat_map, args.heat_thr)
        soft_mask = binary_mask.astype(np.float32)
        if args.smooth_sigma > 0:
            soft_mask = cv2.GaussianBlur(soft_mask, (0, 0), args.smooth_sigma)
        if soft_mask.max() > 0:
            soft_mask = soft_mask / soft_mask.max()
        overlay_map = np.clip(heat_map * soft_mask, 0.0, 1.0)
        overlay = overlay_heatmap(bgr, overlay_map, mask=soft_mask, alpha=args.overlay_alpha)
        ranked = sorted(activated_classes, key=lambda x: x[1], reverse=True)
        if not ranked:
            ranked = sorted([(name, float(probs[i])) for i, name in enumerate(CLASSES) if name != 'N'], key=lambda x: x[1], reverse=True)[:1]
        overlay = annotate_overlay_with_labels(overlay, ranked[:args.max_labels])
        overlay_path = output_dir / f"{idx:05d}_heat_overlay.png"
        mask_path = output_dir / f"{idx:05d}_heat_mask.png"
        cv2.imwrite(str(overlay_path), overlay)
        cv2.imwrite(str(mask_path), binary_mask.astype(np.uint8) * 255)
        np.save(output_dir / f"{idx:05d}_heatmap.npy", heat_map)
        np.save(output_dir / f"{idx:05d}_activated.npy", activated_map)
        
        np.save(output_dir / f"{idx:05d}_amap.npy", amap)

if __name__ == '__main__':
    main()
