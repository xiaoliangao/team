"""
批量处理脚本 - 批量生成不同内容和风格的组合
"""

import os
import argparse
import time
from pathlib import Path
from PIL import Image
import torch
import matplotlib.pyplot as plt
import pandas as pd

from neural_style_transfer import NeuralStyleTransfer
from dataset import StyleImageLoader


def batch_process_folder(content_dir, style_dir, output_dir, args):
    """
    批量处理：将内容文件夹中的所有图片应用到风格文件夹中的所有风格
    
    Args:
        content_dir: 内容图片文件夹
        style_dir: 风格图片文件夹
        output_dir: 输出文件夹
        args: 其他参数
    """
    print("=" * 70)
    print("批量处理模式：文件夹 × 文件夹")
    print("=" * 70)
    
    # 获取所有图片文件
    content_files = []
    style_files = []
    
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp'}
    
    for ext in image_extensions:
        content_files.extend(Path(content_dir).glob(f'*{ext}'))
        content_files.extend(Path(content_dir).glob(f'*{ext.upper()}'))
        style_files.extend(Path(style_dir).glob(f'*{ext}'))
        style_files.extend(Path(style_dir).glob(f'*{ext.upper()}'))
    
    content_files = sorted(list(set(content_files)))
    style_files = sorted(list(set(style_files)))
    
    print(f"\n找到 {len(content_files)} 个内容图片")
    print(f"找到 {len(style_files)} 个风格图片")
    print(f"总共需要生成 {len(content_files) * len(style_files)} 张图片")
    
    if len(content_files) == 0 or len(style_files) == 0:
        print("错误：没有找到图片文件")
        return
    
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 初始化模型
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    nst = NeuralStyleTransfer(device=device)
    nst.imsize = args.size
    
    # 批量处理
    total = len(content_files) * len(style_files)
    count = 0
    results = []
    
    start_time = time.time()
    
    for content_file in content_files:
        content_name = content_file.stem
        
        # 加载内容图片
        content_img = nst.load_image(str(content_file))
        
        for style_file in style_files:
            style_name = style_file.stem
            count += 1
            
            print(f"\n[{count}/{total}] 处理: {content_name} + {style_name}")
            
            # 加载风格图片
            style_img = nst.load_image(str(style_file))
            
            # 执行风格迁移
            try:
                output = nst.run_style_transfer(
                    content_img,
                    style_img,
                    num_steps=args.steps,
                    style_weight=args.style_weight,
                    content_weight=args.content_weight
                )
                
                # 保存结果
                result_img = nst.show_image(output)
                output_filename = f"{content_name}_{style_name}.jpg"
                output_path = os.path.join(output_dir, output_filename)
                result_img.save(output_path)
                
                print(f"✓ 已保存: {output_filename}")
                
                results.append({
                    'content': content_name,
                    'style': style_name,
                    'output': output_filename,
                    'status': 'success'
                })
                
            except Exception as e:
                print(f"✗ 错误: {e}")
                results.append({
                    'content': content_name,
                    'style': style_name,
                    'output': None,
                    'status': f'error: {e}'
                })
    
    elapsed_time = time.time() - start_time
    
    # 保存处理结果记录
    df = pd.DataFrame(results)
    df.to_csv(os.path.join(output_dir, 'batch_results.csv'), index=False)
    
    print("\n" + "=" * 70)
    print(f"批量处理完成！")
    print(f"总耗时: {elapsed_time:.2f}秒")
    print(f"成功: {len([r for r in results if r['status'] == 'success'])} / {total}")
    print(f"结果保存在: {output_dir}")
    print("=" * 70)


def batch_process_wikiart(content_dir, output_dir, args):
    """
    批量处理：使用WikiArt数据集作为风格源
    
    Args:
        content_dir: 内容图片文件夹
        output_dir: 输出文件夹
        args: 其他参数
    """
    print("=" * 70)
    print("批量处理模式：内容文件夹 × WikiArt数据集")
    print("=" * 70)
    
    # 检查WikiArt数据集
    if not os.path.exists(args.wikiart):
        print(f"错误：找不到WikiArt数据集: {args.wikiart}")
        return
    
    # 获取内容图片
    content_files = []
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp'}
    
    for ext in image_extensions:
        content_files.extend(Path(content_dir).glob(f'*{ext}'))
        content_files.extend(Path(content_dir).glob(f'*{ext.upper()}'))
    
    content_files = sorted(list(set(content_files)))
    
    print(f"\n找到 {len(content_files)} 个内容图片")
    print(f"将为每个内容图片生成 {args.num_styles} 种风格")
    
    if len(content_files) == 0:
        print("错误：没有找到内容图片")
        return
    
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 初始化
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    nst = NeuralStyleTransfer(device=device)
    nst.imsize = args.size
    
    style_loader = StyleImageLoader(args.wikiart, image_size=args.size)
    
    # 获取要使用的风格列表
    if args.styles:
        styles_to_use = args.styles.split(',')
    else:
        all_styles = style_loader.list_styles()
        # 随机选择几个风格
        import random
        styles_to_use = random.sample(all_styles, min(args.num_styles, len(all_styles)))
    
    print(f"\n将使用以下风格: {styles_to_use}")
    
    # 批量处理
    results = []
    start_time = time.time()
    total = len(content_files) * len(styles_to_use)
    count = 0
    
    for content_file in content_files:
        content_name = content_file.stem
        
        # 加载内容图片
        content_img = nst.load_image(str(content_file))
        
        for style_name in styles_to_use:
            count += 1
            print(f"\n[{count}/{total}] 处理: {content_name} + {style_name}")
            
            try:
                # 从WikiArt获取风格图片
                style_tensor, metadata = style_loader.get_random_image(style=style_name)
                style_img = style_tensor.unsqueeze(0).to(nst.device)
                
                print(f"  风格图片: {metadata.get('title', 'Unknown')} - {metadata.get('artist', 'Unknown')}")
                
                # 执行风格迁移
                output = nst.run_style_transfer(
                    content_img,
                    style_img,
                    num_steps=args.steps,
                    style_weight=args.style_weight,
                    content_weight=args.content_weight
                )
                
                # 保存结果
                result_img = nst.show_image(output)
                style_name_clean = style_name.replace(' ', '_')
                output_filename = f"{content_name}_{style_name_clean}.jpg"
                output_path = os.path.join(output_dir, output_filename)
                result_img.save(output_path)
                
                print(f"✓ 已保存: {output_filename}")
                
                results.append({
                    'content': content_name,
                    'style': style_name,
                    'artist': metadata.get('artist', 'Unknown'),
                    'title': metadata.get('title', 'Unknown'),
                    'output': output_filename,
                    'status': 'success'
                })
                
            except Exception as e:
                print(f"✗ 错误: {e}")
                results.append({
                    'content': content_name,
                    'style': style_name,
                    'artist': None,
                    'title': None,
                    'output': None,
                    'status': f'error: {e}'
                })
    
    elapsed_time = time.time() - start_time
    
    # 保存结果记录
    df = pd.DataFrame(results)
    df.to_csv(os.path.join(output_dir, 'batch_wikiart_results.csv'), index=False)
    
    print("\n" + "=" * 70)
    print(f"批量处理完成！")
    print(f"总耗时: {elapsed_time:.2f}秒 (平均每张: {elapsed_time/total:.2f}秒)")
    print(f"成功: {len([r for r in results if r['status'] == 'success'])} / {total}")
    print(f"结果保存在: {output_dir}")
    print("=" * 70)


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='批量风格迁移处理')
    
    # 模式选择
    parser.add_argument('--mode', type=str, default='folder',
                       choices=['folder', 'wikiart'],
                       help='批量处理模式')
    
    # 输入输出
    parser.add_argument('--content-dir', type=str, required=True,
                       help='内容图片文件夹')
    parser.add_argument('--style-dir', type=str, default=None,
                       help='风格图片文件夹（folder模式需要）')
    parser.add_argument('--output', type=str, default='batch_output',
                       help='输出目录')
    
    # WikiArt相关
    parser.add_argument('--wikiart', type=str,
                       default='wikiart/data/train-00000-of-00072.parquet',
                       help='WikiArt数据集路径')
    parser.add_argument('--styles', type=str, default=None,
                       help='指定风格列表，逗号分隔（如：Impressionism,Cubism）')
    parser.add_argument('--num-styles', type=int, default=5,
                       help='每个内容图片应用的风格数量（wikiart模式）')
    
    # 训练参数
    parser.add_argument('--steps', type=int, default=200,
                       help='优化步数（批量处理建议减少）')
    parser.add_argument('--style-weight', type=float, default=1e6,
                       help='风格权重')
    parser.add_argument('--content-weight', type=float, default=1,
                       help='内容权重')
    parser.add_argument('--size', type=int, default=512,
                       help='图像尺寸')
    
    return parser.parse_args()


def main():
    """主函数"""
    args = parse_args()
    
    # 检查内容目录
    if not os.path.exists(args.content_dir):
        print(f"错误：找不到内容目录: {args.content_dir}")
        return
    
    # 根据模式执行
    if args.mode == 'folder':
        if not args.style_dir:
            print("错误：folder模式需要指定 --style-dir")
            return
        if not os.path.exists(args.style_dir):
            print(f"错误：找不到风格目录: {args.style_dir}")
            return
        
        batch_process_folder(args.content_dir, args.style_dir, args.output, args)
    
    elif args.mode == 'wikiart':
        batch_process_wikiart(args.content_dir, args.output, args)


if __name__ == "__main__":
    main()
