"""
训练脚本 - 支持命令行参数进行风格迁移实验
"""

import os
import argparse
import time
from PIL import Image
import torch
import matplotlib.pyplot as plt

from neural_style_transfer import NeuralStyleTransfer
from dataset import StyleImageLoader
from torchvision import transforms as T


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='神经风格迁移训练')
    
    # 输入输出
    parser.add_argument('--content', type=str, required=True,
                       help='内容图片路径')
    parser.add_argument('--style', type=str, default=None,
                       help='风格图片路径（如果不指定，从数据集随机选择）')
    parser.add_argument('--output', type=str, default='output',
                       help='输出目录')
    
    # 数据集相关
    parser.add_argument('--wikiart', type=str, 
                       default='wikiart/data/train-00000-of-00072.parquet',
                       help='WikiArt数据集路径')
    parser.add_argument('--style-name', type=str, default=None,
                       help='从数据集中选择特定风格（如：Impressionism）')
    parser.add_argument('--artist', type=str, default=None,
                       help='从数据集中选择特定艺术家的作品')
    
    # 训练参数
    parser.add_argument('--steps', type=int, default=300,
                       help='优化步数')
    parser.add_argument('--style-weight', type=float, default=1e6,
                       help='风格权重')
    parser.add_argument('--content-weight', type=float, default=1,
                       help='内容权重')
    parser.add_argument('--size', type=int, default=512,
                       help='图像尺寸')
    
    # 其他选项
    parser.add_argument('--device', type=str, default='auto',
                       choices=['auto', 'cuda', 'cpu'],
                       help='设备选择')
    parser.add_argument('--save-every', type=int, default=50,
                       help='每N步保存一次中间结果')
    parser.add_argument('--no-display', action='store_true',
                       help='不显示图像')
    
    return parser.parse_args()


def load_content_image(path, nst):
    """加载内容图片"""
    print(f"加载内容图片: {path}")
    if not os.path.exists(path):
        raise FileNotFoundError(f"找不到内容图片: {path}")
    
    content_img = nst.load_image(path)
    return content_img


def load_style_image(args, nst):
    """加载风格图片"""
    # 如果指定了风格图片路径
    if args.style:
        print(f"加载风格图片: {args.style}")
        if not os.path.exists(args.style):
            raise FileNotFoundError(f"找不到风格图片: {args.style}")
        style_img = nst.load_image(args.style)
        style_info = {"source": "file", "path": args.style}
    
    # 否则从WikiArt数据集加载
    else:
        if not os.path.exists(args.wikiart):
            raise FileNotFoundError(
                f"找不到WikiArt数据集: {args.wikiart}\n"
                f"请指定 --style 参数或下载WikiArt数据集"
            )
        
        print(f"从WikiArt数据集加载风格图片...")
        loader = StyleImageLoader(args.wikiart, image_size=args.size)
        
        # 根据条件获取
        style_tensor, metadata = loader.get_random_image(
            style=args.style_name,
            artist=args.artist
        )
        
        style_img = style_tensor.unsqueeze(0).to(nst.device)
        style_info = {
            "source": "wikiart",
            "metadata": metadata
        }
        
        print(f"选择了风格图片：")
        for key, value in metadata.items():
            print(f"  {key}: {value}")
    
    return style_img, style_info


def save_intermediate_result(nst, input_img, step, output_dir):
    """保存中间结果"""
    intermediate_dir = os.path.join(output_dir, "intermediate")
    os.makedirs(intermediate_dir, exist_ok=True)
    
    result_img = nst.show_image(input_img, title=f"Step {step}")
    save_path = os.path.join(intermediate_dir, f"step_{step:04d}.jpg")
    result_img.save(save_path)
    plt.close()
    
    return save_path


def main():
    """主函数"""
    args = parse_args()
    
    # 创建输出目录
    os.makedirs(args.output, exist_ok=True)
    
    # 设置设备
    if args.device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    else:
        device = args.device
    
    print("=" * 70)
    print("神经风格迁移训练")
    print("=" * 70)
    print(f"设备: {device}")
    print(f"图像尺寸: {args.size}x{args.size}")
    print(f"优化步数: {args.steps}")
    print(f"风格权重: {args.style_weight}")
    print(f"内容权重: {args.content_weight}")
    print("=" * 70)
    
    # 初始化模型
    nst = NeuralStyleTransfer(device=device)
    # 设置期望的图像尺寸并更新 loader（需要处理 PIL -> Tensor 的转换）
    nst.imsize = args.size
    nst.loader = T.Compose([
        T.Resize((args.size, args.size)),
        T.ToTensor()
    ])
    
    # 加载图片
    content_img = load_content_image(args.content, nst)
    style_img, style_info = load_style_image(args, nst)
    
    # 显示输入图片
    if not args.no_display:
        plt.figure(figsize=(12, 4))
        
        plt.subplot(1, 2, 1)
        plt.imshow(nst.show_image(content_img))
        plt.title('Content Image')
        plt.axis('off')
        
        plt.subplot(1, 2, 2)
        plt.imshow(nst.show_image(style_img))
        plt.title('Style Image')
        plt.axis('off')
        
        plt.tight_layout()
        input_path = os.path.join(args.output, "input_images.jpg")
        plt.savefig(input_path, bbox_inches='tight', dpi=150)
        print(f"输入图片已保存: {input_path}")
        plt.close()
    
    # 执行风格迁移
    print("\n开始风格迁移...")
    start_time = time.time()
    
    # 修改run_style_transfer以支持中间结果保存
    output_img = nst.run_style_transfer(
        content_img,
        style_img,
        num_steps=args.steps,
        style_weight=args.style_weight,
        content_weight=args.content_weight
    )
    
    end_time = time.time()
    elapsed_time = end_time - start_time
    
    print(f"\n风格迁移完成！耗时: {elapsed_time:.2f}秒")
    
    # 保存最终结果
    result_img = nst.show_image(output_img, title='Result')
    
    # 生成输出文件名
    content_name = os.path.splitext(os.path.basename(args.content))[0]
    if args.style:
        style_name = os.path.splitext(os.path.basename(args.style))[0]
    else:
        style_name = style_info['metadata'].get('style', 'unknown')
        style_name = style_name.replace(' ', '_')
    
    output_filename = f"{content_name}_{style_name}.jpg"
    output_path = os.path.join(args.output, output_filename)
    result_img.save(output_path)
    
    print(f"结果已保存: {output_path}")
    
    # 保存对比图
    if not args.no_display:
        plt.figure(figsize=(15, 5))
        
        plt.subplot(1, 3, 1)
        plt.imshow(nst.show_image(content_img))
        plt.title('Content')
        plt.axis('off')
        
        plt.subplot(1, 3, 2)
        plt.imshow(nst.show_image(style_img))
        plt.title('Style')
        plt.axis('off')
        
        plt.subplot(1, 3, 3)
        plt.imshow(result_img)
        plt.title('Result')
        plt.axis('off')
        
        plt.tight_layout()
        comparison_path = os.path.join(args.output, f"comparison_{content_name}_{style_name}.jpg")
        plt.savefig(comparison_path, bbox_inches='tight', dpi=150)
        print(f"对比图已保存: {comparison_path}")
        plt.close()
    
    # 保存训练信息
    info_path = os.path.join(args.output, f"info_{content_name}_{style_name}.txt")
    with open(info_path, 'w', encoding='utf-8') as f:
        f.write("风格迁移训练信息\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"内容图片: {args.content}\n")
        f.write(f"风格来源: {style_info['source']}\n")
        if style_info['source'] == 'wikiart':
            f.write("\n风格图片信息:\n")
            for key, value in style_info['metadata'].items():
                f.write(f"  {key}: {value}\n")
        else:
            f.write(f"风格图片: {args.style}\n")
        f.write(f"\n训练参数:\n")
        f.write(f"  图像尺寸: {args.size}x{args.size}\n")
        f.write(f"  优化步数: {args.steps}\n")
        f.write(f"  风格权重: {args.style_weight}\n")
        f.write(f"  内容权重: {args.content_weight}\n")
        f.write(f"  设备: {device}\n")
        f.write(f"\n训练时间: {elapsed_time:.2f}秒\n")
    
    print(f"训练信息已保存: {info_path}")
    print("\n" + "=" * 70)
    print("全部完成！")
    print("=" * 70)


if __name__ == "__main__":
    main()
