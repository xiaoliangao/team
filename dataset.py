"""
WikiArt数据集加载和处理
支持从parquet文件加载图像数据
"""

import os
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import io
import numpy as np
from torchvision import transforms


class WikiArtDataset(Dataset):
    """WikiArt数据集类"""
    
    def __init__(self, parquet_file, transform=None, max_samples=None):
        """
        初始化WikiArt数据集
        
        Args:
            parquet_file: parquet文件路径
            transform: 图像变换
            max_samples: 最大样本数（用于测试）
        """
        self.df = pd.read_parquet(parquet_file)
        
        if max_samples:
            self.df = self.df.head(max_samples)
        
        self.transform = transform
        
        print(f"加载了 {len(self.df)} 张图片")
        
        # 打印数据集信息
        if 'style' in self.df.columns:
            print(f"包含风格: {self.df['style'].unique()[:10]}")
        if 'artist' in self.df.columns:
            print(f"包含艺术家: {len(self.df['artist'].unique())} 位")
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        """获取一个样本"""
        row = self.df.iloc[idx]
        
        # 从bytes加载图像
        if 'image' in row:
            image_bytes = row['image']['bytes']
            image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
        else:
            # 如果没有image字段，尝试其他可能的字段
            raise ValueError("parquet文件中找不到图像数据")
        
        # 应用变换
        if self.transform:
            image = self.transform(image)
        
        # 获取元数据
        metadata = {}
        for col in ['style', 'artist', 'genre', 'title']:
            if col in row:
                metadata[col] = row[col]
        
        return image, metadata


class StyleImageLoader:
    """风格图像加载器，用于选择不同风格的图片"""
    
    def __init__(self, parquet_file, image_size=512):
        """
        初始化
        
        Args:
            parquet_file: parquet文件路径
            image_size: 目标图像大小
        """
        self.df = pd.read_parquet(parquet_file)
        self.image_size = image_size
        
        self.transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor()
        ])
        
        print(f"风格图像库包含 {len(self.df)} 张图片")
        
        # 如果有风格信息，统计各个风格
        if 'style' in self.df.columns:
            style_counts = self.df['style'].value_counts()
            print(f"\n风格分布：")
            for style, count in style_counts.head(10).items():
                print(f"  {style}: {count}")
    
    def get_random_image(self, style=None, artist=None):
        """
        随机获取一张风格图像
        
        Args:
            style: 指定风格（可选）
            artist: 指定艺术家（可选）
        Returns:
            image_tensor, metadata
        """
        df_filtered = self.df.copy()
        
        # 按条件筛选
        if style and 'style' in df_filtered.columns:
            df_filtered = df_filtered[df_filtered['style'] == style]
        
        if artist and 'artist' in df_filtered.columns:
            df_filtered = df_filtered[df_filtered['artist'] == artist]
        
        if len(df_filtered) == 0:
            print(f"警告：没有找到匹配的图片，使用随机图片")
            df_filtered = self.df
        
        # 随机选择一张
        idx = np.random.randint(0, len(df_filtered))
        row = df_filtered.iloc[idx]
        
        # 加载图像
        image_bytes = row['image']['bytes']
        image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
        
        # 应用变换
        image_tensor = self.transform(image)
        
        # 元数据
        metadata = {}
        for col in ['style', 'artist', 'genre', 'title']:
            if col in row:
                metadata[col] = row[col]
        
        return image_tensor, metadata
    
    def get_images_by_style(self, style, num_images=5):
        """
        获取指定风格的多张图像
        
        Args:
            style: 风格名称
            num_images: 图像数量
        Returns:
            List of (image_tensor, metadata)
        """
        if 'style' not in self.df.columns:
            raise ValueError("数据集中没有风格信息")
        
        df_style = self.df[self.df['style'] == style]
        
        if len(df_style) == 0:
            raise ValueError(f"没有找到风格 '{style}' 的图片")
        
        # 随机选择
        if len(df_style) < num_images:
            print(f"警告：只有 {len(df_style)} 张 {style} 风格的图片")
            num_images = len(df_style)
        
        indices = np.random.choice(len(df_style), num_images, replace=False)
        
        results = []
        for idx in indices:
            row = df_style.iloc[idx]
            
            # 加载图像
            image_bytes = row['image']['bytes']
            image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
            image_tensor = self.transform(image)
            
            # 元数据
            metadata = {}
            for col in ['style', 'artist', 'genre', 'title']:
                if col in row:
                    metadata[col] = row[col]
            
            results.append((image_tensor, metadata))
        
        return results
    
    def list_styles(self):
        """列出所有可用的风格"""
        if 'style' in self.df.columns:
            return sorted(self.df['style'].unique().tolist())
        return []
    
    def list_artists(self, top_n=20):
        """列出最多作品的艺术家"""
        if 'artist' in self.df.columns:
            artist_counts = self.df['artist'].value_counts().head(top_n)
            return artist_counts.to_dict()
        return {}


def create_dataloader(parquet_file, batch_size=4, shuffle=True, 
                      image_size=512, max_samples=None):
    """
    创建数据加载器
    
    Args:
        parquet_file: parquet文件路径
        batch_size: 批大小
        shuffle: 是否打乱
        image_size: 图像大小
        max_samples: 最大样本数
    Returns:
        DataLoader
    """
    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
    ])
    
    dataset = WikiArtDataset(parquet_file, transform=transform, 
                            max_samples=max_samples)
    
    dataloader = DataLoader(dataset, batch_size=batch_size, 
                           shuffle=shuffle, num_workers=2)
    
    return dataloader


# ==================== 测试代码 ====================
if __name__ == "__main__":
    # 测试数据集加载
    parquet_file = "wikiart/data/train-00000-of-00072.parquet"
    
    if os.path.exists(parquet_file):
        print("=" * 60)
        print("测试 WikiArt 数据集加载")
        print("=" * 60)
        
        # 创建风格图像加载器
        loader = StyleImageLoader(parquet_file, image_size=512)
        
        print("\n可用风格：")
        styles = loader.list_styles()
        for i, style in enumerate(styles[:20], 1):
            print(f"{i}. {style}")
        
        print("\n\n顶级艺术家：")
        artists = loader.list_artists(10)
        for artist, count in artists.items():
            print(f"{artist}: {count} 作品")
        
        # 随机获取一张图片
        print("\n\n随机获取一张图片：")
        image, metadata = loader.get_random_image()
        print(f"图像shape: {image.shape}")
        print(f"元数据: {metadata}")
        
        # 获取特定风格的图片
        if len(styles) > 0:
            test_style = styles[0]
            print(f"\n\n获取 '{test_style}' 风格的图片：")
            try:
                images = loader.get_images_by_style(test_style, num_images=3)
                print(f"成功获取 {len(images)} 张图片")
                for i, (img, meta) in enumerate(images, 1):
                    print(f"  {i}. {meta.get('title', 'Unknown')} - {meta.get('artist', 'Unknown')}")
            except Exception as e:
                print(f"错误: {e}")
        
        print("\n" + "=" * 60)
        print("测试完成！")
        print("=" * 60)
    else:
        print(f"错误：找不到文件 {parquet_file}")
        print("请确保 WikiArt 数据集已下载到 wikiart/data/ 目录")
