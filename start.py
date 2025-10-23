"""
快速启动脚本 - 神经风格迁移项目
"""

import os
import sys

def check_dependencies():
    """检查依赖是否安装"""
    print("检查依赖...")
    
    required = ['torch', 'torchvision', 'gradio', 'PIL', 'pandas', 'matplotlib']
    missing = []
    
    for package in required:
        try:
            if package == 'PIL':
                __import__('PIL')
            else:
                __import__(package)
            print(f"✓ {package}")
        except ImportError:
            print(f"✗ {package} - 未安装")
            missing.append(package)
    
    if missing:
        print(f"\n缺少依赖: {', '.join(missing)}")
        print("\n请运行: pip install -r requirements.txt")
        return False
    
    print("\n所有依赖已安装 ✓")
    return True


def check_structure():
    """检查目录结构"""
    print("\n检查目录结构...")
    
    dirs = [
        'data/outputs',
        'wikiart/data'
    ]
    
    for d in dirs:
        if os.path.exists(d):
            print(f"✓ {d}")
        else:
            print(f"✗ {d} - 不存在，正在创建...")
            os.makedirs(d, exist_ok=True)
    
    print("\n目录结构正常 ✓")


def check_wikiart():
    """检查WikiArt数据集"""
    print("\n检查WikiArt数据集...")
    
    wikiart_dir = 'wikiart/data'
    if not os.path.exists(wikiart_dir):
        print("✗ WikiArt目录不存在")
        return False
    
    parquet_files = [f for f in os.listdir(wikiart_dir) if f.endswith('.parquet')]
    
    if parquet_files:
        print(f"✓ 找到 {len(parquet_files)} 个parquet文件")
        return True
    else:
        print("⚠ 未找到WikiArt数据集文件")
        print("  可以手动上传风格图片，或下载WikiArt数据集")
        return False


def main():
    """主函数"""
    print("=" * 60)
    print("神经风格迁移项目 - 快速启动")
    print("=" * 60)
    print()
    
    # 检查依赖
    if not check_dependencies():
        sys.exit(1)
    
    # 检查目录结构
    check_structure()
    
    # 检查WikiArt
    has_wikiart = check_wikiart()
    
    print("\n" + "=" * 60)
    print("准备完成！")
    print("=" * 60)
    print()
    print("启动方式:")
    print()
    print("1. 启动Web GUI (推荐):")
    print("   python app.py")
    print()
    print("2. 命令行使用:")
    print("   python train.py --content photo.jpg --style art.jpg")
    print()
    
    if has_wikiart:
        print("3. 使用WikiArt数据集:")
        print("   python train.py --content photo.jpg --style-name Impressionism")
        print()
    
    print("详细使用说明请查看 USAGE.md")
    print()
    
    # 询问是否启动GUI
    try:
        response = input("是否现在启动Web GUI? (y/n): ").strip().lower()
        if response == 'y':
            print("\n正在启动Web GUI...")
            os.system('python app.py')
    except KeyboardInterrupt:
        print("\n\n再见！")


if __name__ == "__main__":
    main()
