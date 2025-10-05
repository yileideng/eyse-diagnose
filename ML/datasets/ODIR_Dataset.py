import os
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Dataset
import pandas as pd
from PIL import Image
from .preprocess import preprocess_fundus
from .transforms import build_train_transforms, build_eval_transforms

class ODIRDataset(Dataset):
    def __init__(self, img_dir, csv_path, stage="train", transform=None, preprocess=True, cache_dir=None, resample_prob=0.0, seed=42):
        self.img_dir = Path(img_dir)
        self.csv_path = Path(csv_path)
        self.stage = stage
        self.preprocess = preprocess
        self.df = pd.read_csv(self.csv_path)
        self.label_cols = self.df.columns[5:13].tolist()
        self.labels = torch.tensor(self.df[self.label_cols].values, dtype=torch.float32)
        self.ids = self.df["ID"].astype(str).tolist()
        self.rng = np.random.default_rng(seed)
        self.cache_dir = Path(cache_dir) if cache_dir else None
        if transform is None:
            if stage == "train":
                self.transform = build_train_transforms()
            else:
                self.transform = build_eval_transforms()
        else:
            self.transform = transform
        if self.cache_dir:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.indices = np.arange(len(self.df))
        if stage == "train" and resample_prob > 0:
            freq = self.labels.sum(dim=0).numpy() + 1
            inv = freq.max() / freq
            weights = (self.labels.numpy() * inv).sum(axis=1)
            weights = weights / weights.sum()
            extra = int(len(self.df) * resample_prob)
            sampled = self.rng.choice(len(self.df), size=extra, p=weights)
            self.indices = np.concatenate([self.indices, sampled])
        self.length = len(self.indices)

    def __len__(self):
        return self.length

    def _load_image(self, name):
        path = self.img_dir / name
        with Image.open(path) as im:
            return im.convert("RGB")

    def _maybe_preprocess(self, im, cache_key):
        if not self.preprocess:
            return im
        if self.cache_dir:
            cache_path = self.cache_dir / f"{cache_key}.png"
            if cache_path.exists():
                with Image.open(cache_path) as cached:
                    return cached.convert("RGB")
            processed = preprocess_fundus(im)
            processed.save(cache_path)
            return processed
        return preprocess_fundus(im)

    def __getitem__(self, idx):
        real_idx = self.indices[idx]
        row = self.df.iloc[real_idx]
        left = self._load_image(row["Left-Fundus"])
        right = self._load_image(row["Right-Fundus"])
        left = self._maybe_preprocess(left, f"{row['ID']}_L")
        right = self._maybe_preprocess(right, f"{row['ID']}_R")
        combined = Image.new("RGB", (left.width + right.width, left.height))
        combined.paste(left, (0, 0))
        combined.paste(right, (left.width, 0))
        arr = np.array(combined)
        data = self.transform(image=arr)
        tensor = data["image"].float()
        target = self.labels[real_idx]
        return tensor, target, self.ids[real_idx]
