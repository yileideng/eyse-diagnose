import albumentations as A
from albumentations.pytorch import ToTensorV2

def build_train_transforms(height=256, width=512):
    return A.Compose([
        A.Resize(height, width, interpolation=3),
        A.OneOf([
            A.Affine(scale=(0.95, 1.05), translate_percent=(0.0, 0.05), rotate=(-15, 15), shear=(-5, 5), fit_output=False, p=0.7),
            A.Perspective(scale=(0.02, 0.05), keep_size=True, p=0.3)
        ], p=0.8),
        A.RandomResizedCrop(size=(height, width), scale=(0.85, 1.0), ratio=(1.8, 2.2), interpolation=3, p=0.5),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.1),
        A.OneOf([
            A.CLAHE(clip_limit=2.0, tile_grid_size=(8, 8), p=0.5),
            A.Emboss(alpha=(0.2, 0.4), strength=(0.2, 0.4), p=0.5),
            A.MotionBlur(blur_limit=5, p=0.5)
        ], p=0.5),
        A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=0.7),
        A.RandomGamma(gamma_limit=(80, 120), p=0.3),
        A.ISONoise(color_shift=(0.01, 0.05), intensity=(0.05, 0.4), p=0.3),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2()
    ])

def build_eval_transforms(height=256, width=512):
    return A.Compose([
        A.Resize(height, width, interpolation=3),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2()
    ])
