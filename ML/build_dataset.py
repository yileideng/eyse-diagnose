from torch.utils.data import DataLoader, Dataset
import torch
from torchvision import transforms, datasets
import pandas as pd
from PIL import Image
import os
from torchvision.datasets.folder import default_loader
import matplotlib.pyplot as plt
import numpy as np 
import sys
sys.stdin.reconfigure(encoding='utf-8')
sys.stdout.reconfigure(encoding='utf-8')


data_dir = 'D:/Fc25_07/FcccccUestc/Training_data'
label_dir = 'D:/Fc25_07/FcccccUestc/total_data.csv'
# Excel 文件路径


# 将 DataFrame 保存为 CSV 文件
df = pd.read_csv(label_dir, header=0, encoding='utf-8')

class CustomDataset(Dataset):
    def __init__(self, data_dir, label_dir, transform=None):
        self.data_dir = data_dir
        self.transform = transform
        
        # 读取 CSV 文件
        self.df = df
        
        self.left_image_names = self.df['Left-Fundus'].values
        self.right_image_names = self.df['Right-Fundus'].values

        self.labels = self.df.iloc[:, 5:13].values.astype(int)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        # 获取左右眼图片的文件名
        left_img_name = self.left_image_names[idx]
        right_img_name = self.right_image_names[idx]
        
        # 构造图片路径
        left_img_path = os.path.join(self.data_dir, left_img_name)
        right_img_path = os.path.join(self.data_dir, right_img_name)
        
        # 加载图片
        left_image = default_loader(left_img_path)
        right_image = default_loader(right_img_path)
        
        # 拼接左右眼图片
        image = Image.new('RGB', (left_image.width + right_image.width, left_image.height))
        image.paste(left_image, (0, 0))
        image.paste(right_image, (left_image.width, 0))
        
        # 应用变换
        if self.transform:
            image = self.transform(image)
        
        # 获取标签
        label = torch.tensor(self.labels[idx], dtype=torch.float32)
        
        return image, label

batch_size = 32

# 图像变换
train_transforms = transforms.Compose([
    transforms.RandomResizedCrop(224),
    # transforms.ColorJitter(brightness=1.1, contrast=1.5, saturation=0.8),       # 可根据训练结果调节
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomVerticalFlip(p=0.5),
    # transforms.RandomRotation(degrees=35),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

valid_transforms = transforms.Compose([
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])
# -------------------------------------------------------------------------------------------------

# 数据集划分（80%训练，20%验证）

dataset = CustomDataset(data_dir=data_dir, label_dir=label_dir, transform=None)
data_len = len(dataset)


train_size = int(0.8 * data_len)
val_size = data_len - train_size
train_indices, val_indices = torch.utils.data.random_split(
    dataset=dataset, lengths=[train_size, val_size]
)


class TransformSubset(torch.utils.data.Dataset):
    def __init__(self, subset, transform=None):
        self.subset = subset
        self.transform = transform

    def __getitem__(self, index):
        x, y = self.subset[index]
        if self.transform:
            x = self.transform(x)
        return x, y

    def __len__(self):
        return len(self.subset)


train_dataset = TransformSubset(train_indices, transform=train_transforms)
val_dataset = TransformSubset(val_indices, transform=valid_transforms)

train_dataloader = DataLoader(dataset=train_dataset, shuffle=True, batch_size=batch_size)
valid_dataloader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True)


def denormalize(tensor, mean, std):
    """反归一化张量"""
    for t, m, s in zip(tensor, mean, std):
        t.mul_(s).add_(m)
    return tensor

# 可视化函数
def show_images_and_labels(dataloader, num_images=5):
    # 获取一批数据
    images, labels = next(iter(dataloader))
    
    # 反归一化
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    images = denormalize(images, mean, std)
    
    # 创建画布
    fig, axes = plt.subplots(nrows=1, ncols=num_images, figsize=(15, 5))
    
    for i in range(num_images):
        # 转换为 PIL 图像
        img = images[i].permute(1, 2, 0)  # 从 (C, H, W) 转为 (H, W, C)
        img = np.clip(img.numpy(), 0, 1)  # 确保像素值在 0-1 范围内
        
        # 显示图像
        axes[i].imshow(img)
        axes[i].set_title(f"Label: {labels[i].numpy()}")
        axes[i].axis('off')
    
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":

# 查看前五个患者的拼接后图片和标签
    show_images_and_labels(train_dataloader, num_images=5)