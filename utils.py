"""
工具函数 - 图像处理、可视化等辅助功能
"""

import os
import torch
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFont
import numpy as np
from torchvision import transforms


def save_comparison_grid(images, titles, output_path, rows=1, cols=None, figsize=None):
    """
    保存多张图片的对比网格
    
    Args:
        images: 图片列表（PIL Image或tensor）
        titles: 标题列表
        output_path: 保存路径
        rows: 行数
        cols: 列数（如果None则自动计算）
        figsize: 图片大小
    """
    n = len(images)
    
    if cols is None:
        cols = (n + rows - 1) // rows
    
    if figsize is None:
        figsize = (5 * cols, 5 * rows)
    
    fig, axes = plt.subplots(rows, cols, figsize=figsize)
    
    if rows == 1 and cols == 1:
        axes = [[axes]]
    elif rows == 1 or cols == 1:
        axes = axes.reshape(rows, cols)
    
    for i, (img, title) in enumerate(zip(images, titles)):
        row = i // cols
        col = i % cols
        
        # 转换tensor到PIL Image
        if torch.is_tensor(img):
            img = transforms.ToPILImage()(img.cpu().squeeze(0))
        
        axes[row, col].imshow(img)
        axes[row, col].set_title(title, fontsize=12)
        axes[row, col].axis('off')
    
    # 隐藏多余的子图
    for i in range(n, rows * cols):
        row = i // cols
        col = i % cols
        axes[row, col].axis('off')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"对比图已保存: {output_path}")


def create_labeled_image(image, label, font_size=40):
    """
    在图片上添加文字标签
    
    Args:
        image: PIL Image
        label: 标签文字
        font_size: 字体大小
    Returns:
        带标签的PIL Image
    """
    # 创建副本
    img_copy = image.copy()
    draw = ImageDraw.Draw(img_copy)
    
    # 尝试加载字体
    try:
        font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", font_size)
    except:
        font = ImageFont.load_default()
    
    # 计算文字位置（左上角）
    bbox = draw.textbbox((0, 0), label, font=font)
    text_width = bbox[2] - bbox[0]
    text_height = bbox[3] - bbox[1]
    
    # 绘制背景
    padding = 10
    draw.rectangle(
        [(0, 0), (text_width + 2*padding, text_height + 2*padding)],
        fill='black'
    )
    
    # 绘制文字
    draw.text((padding, padding), label, fill='white', font=font)
    
    return img_copy


def tensor_to_pil(tensor):
    """
    将tensor转换为PIL Image
    
    Args:
        tensor: 图像tensor (C, H, W) 或 (1, C, H, W)
    Returns:
        PIL Image
    """
    if tensor.dim() == 4:
        tensor = tensor.squeeze(0)
    
    tensor = tensor.cpu().clone()
    tensor = torch.clamp(tensor, 0, 1)
    
    return transforms.ToPILImage()(tensor)


def pil_to_tensor(image, device='cpu'):
    """
    将PIL Image转换为tensor
    
    Args:
        image: PIL Image
        device: 目标设备
    Returns:
        图像tensor (1, C, H, W)
    """
    tensor = transforms.ToTensor()(image)
    return tensor.unsqueeze(0).to(device)


def resize_to_match(img1, img2, mode='smaller'):
    """
    调整图片大小使两张图片尺寸匹配
    
    Args:
        img1, img2: PIL Images
        mode: 'smaller' - 调整到较小尺寸, 'larger' - 调整到较大尺寸
    Returns:
        调整后的(img1, img2)
    """
    w1, h1 = img1.size
    w2, h2 = img2.size
    
    if mode == 'smaller':
        target_w = min(w1, w2)
        target_h = min(h1, h2)
    else:
        target_w = max(w1, w2)
        target_h = max(h1, h2)
    
    if (w1, h1) != (target_w, target_h):
        img1 = img1.resize((target_w, target_h), Image.LANCZOS)
    
    if (w2, h2) != (target_w, target_h):
        img2 = img2.resize((target_w, target_h), Image.LANCZOS)
    
    return img1, img2


def create_side_by_side(images, labels=None, output_path=None):
    """
    创建横向拼接的图片
    
    Args:
        images: PIL Images列表
        labels: 标签列表（可选）
        output_path: 保存路径（可选）
    Returns:
        拼接后的PIL Image
    """
    # 添加标签
    if labels:
        images = [create_labeled_image(img, label) for img, label in zip(images, labels)]
    
    # 统一高度
    heights = [img.size[1] for img in images]
    target_height = max(heights)
    
    resized_images = []
    for img in images:
        if img.size[1] != target_height:
            ratio = target_height / img.size[1]
            new_width = int(img.size[0] * ratio)
            img = img.resize((new_width, target_height), Image.LANCZOS)
        resized_images.append(img)
    
    # 拼接
    total_width = sum(img.size[0] for img in resized_images)
    result = Image.new('RGB', (total_width, target_height))
    
    x_offset = 0
    for img in resized_images:
        result.paste(img, (x_offset, 0))
        x_offset += img.size[0]
    
    # 保存
    if output_path:
        result.save(output_path)
        print(f"拼接图片已保存: {output_path}")
    
    return result


def create_gif_from_frames(frame_paths, output_path, duration=200):
    """
    从图片序列创建GIF动画
    
    Args:
        frame_paths: 图片路径列表
        output_path: 输出GIF路径
        duration: 每帧持续时间（毫秒）
    """
    frames = []
    
    for path in frame_paths:
        if os.path.exists(path):
            frames.append(Image.open(path))
    
    if len(frames) == 0:
        print("错误：没有找到有效的图片")
        return
    
    # 保存为GIF
    frames[0].save(
        output_path,
        save_all=True,
        append_images=frames[1:],
        duration=duration,
        loop=0
    )
    
    print(f"GIF动画已保存: {output_path} ({len(frames)} 帧)")


def calculate_image_stats(tensor):
    """
    计算图像的统计信息
    
    Args:
        tensor: 图像tensor
    Returns:
        字典包含mean, std, min, max等信息
    """
    return {
        'mean': tensor.mean().item(),
        'std': tensor.std().item(),
        'min': tensor.min().item(),
        'max': tensor.max().item(),
        'shape': tuple(tensor.shape)
    }


def print_image_info(tensor, name="Image"):
    """打印图像信息"""
    stats = calculate_image_stats(tensor)
    print(f"\n{name} 信息:")
    print(f"  形状: {stats['shape']}")
    print(f"  均值: {stats['mean']:.4f}")
    print(f"  标准差: {stats['std']:.4f}")
    print(f"  最小值: {stats['min']:.4f}")
    print(f"  最大值: {stats['max']:.4f}")


def save_tensor_as_image(tensor, output_path, normalize=True):
    """
    保存tensor为图片文件
    
    Args:
        tensor: 图像tensor
        output_path: 保存路径
        normalize: 是否归一化到[0,1]
    """
    if normalize:
        tensor = torch.clamp(tensor, 0, 1)
    
    img = tensor_to_pil(tensor)
    img.save(output_path)
    print(f"图片已保存: {output_path}")


def load_image_as_tensor(image_path, size=None, device='cpu'):
    """
    加载图片为tensor
    
    Args:
        image_path: 图片路径
        size: 目标尺寸 (width, height)
        device: 目标设备
    Returns:
        图像tensor (1, C, H, W)
    """
    image = Image.open(image_path).convert('RGB')
    
    if size:
        image = image.resize(size, Image.LANCZOS)
    
    tensor = transforms.ToTensor()(image)
    return tensor.unsqueeze(0).to(device)


def create_video_from_frames(frame_dir, output_path, fps=10, pattern='*.jpg'):
    """
    从图片序列创建视频（需要安装imageio或opencv）
    
    Args:
        frame_dir: 图片文件夹
        output_path: 输出视频路径
        fps: 帧率
        pattern: 文件匹配模式
    """
    try:
        import imageio
        from pathlib import Path
        
        # 获取所有图片
        frames = sorted(Path(frame_dir).glob(pattern))
        
        if len(frames) == 0:
            print(f"错误：在 {frame_dir} 中没有找到匹配 {pattern} 的图片")
            return
        
        # 读取图片并写入视频
        writer = imageio.get_writer(output_path, fps=fps)
        
        for frame_path in frames:
            image = imageio.imread(str(frame_path))
            writer.append_data(image)
        
        writer.close()
        
        print(f"视频已保存: {output_path} ({len(frames)} 帧, {fps} fps)")
        
    except ImportError:
        print("错误：需要安装 imageio 库")
        print("运行: pip install imageio imageio-ffmpeg")


# ==================== 测试代码 ====================
if __name__ == "__main__":
    print("工具函数模块")
    print("包含以下功能：")
    print("  - save_comparison_grid: 保存对比网格")
    print("  - create_labeled_image: 添加标签")
    print("  - tensor_to_pil / pil_to_tensor: 格式转换")
    print("  - create_side_by_side: 横向拼接")
    print("  - create_gif_from_frames: 创建GIF")
    print("  - create_video_from_frames: 创建视频")
