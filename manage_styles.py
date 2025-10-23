"""
风格图片管理工具
用于管理 data/style/ 文件夹中的风格图片
"""

import os
import sys
import shutil
import argparse
from PIL import Image
import matplotlib.pyplot as plt


class StyleManager:
    """风格图片管理器"""
    
    def __init__(self):
        self.base_dir = os.path.dirname(os.path.abspath(__file__))
        self.style_dir = os.path.join(self.base_dir, 'data', 'style')
        self.content_dir = os.path.join(self.base_dir, 'data', 'content')
        
        # 确保目录存在
        os.makedirs(self.style_dir, exist_ok=True)
        os.makedirs(self.content_dir, exist_ok=True)
        
        # 支持的图片格式
        self.supported_formats = ('.jpg', '.jpeg', '.png', '.bmp', '.webp')
    
    def list_styles(self):
        """列出所有风格图片"""
        print("=" * 70)
        print("📚 本地风格图片库")
        print("=" * 70)
        print(f"路径: {self.style_dir}\n")
        
        files = self._get_image_files(self.style_dir)
        
        if not files:
            print("❌ 风格库为空，还没有添加任何风格图片")
            print("\n💡 使用以下命令添加风格图片：")
            print("   python manage_styles.py add <图片路径>")
            return
        
        print(f"共找到 {len(files)} 张风格图片：\n")
        
        for i, file in enumerate(files, 1):
            file_path = os.path.join(self.style_dir, file)
            file_size = os.path.getsize(file_path) / 1024  # KB
            
            try:
                with Image.open(file_path) as img:
                    width, height = img.size
                    print(f"{i:2d}. {file}")
                    print(f"     尺寸: {width}x{height} | 大小: {file_size:.1f} KB")
            except Exception as e:
                print(f"{i:2d}. {file} (读取失败: {e})")
        
        print("\n" + "=" * 70)
    
    def list_contents(self):
        """列出所有内容图片"""
        print("=" * 70)
        print("📷 本地内容图片库")
        print("=" * 70)
        print(f"路径: {self.content_dir}\n")
        
        files = self._get_image_files(self.content_dir)
        
        if not files:
            print("❌ 内容库为空，还没有添加任何内容图片")
            print("\n💡 使用以下命令添加内容图片：")
            print("   python manage_styles.py add-content <图片路径>")
            return
        
        print(f"共找到 {len(files)} 张内容图片：\n")
        
        for i, file in enumerate(files, 1):
            file_path = os.path.join(self.content_dir, file)
            file_size = os.path.getsize(file_path) / 1024  # KB
            
            try:
                with Image.open(file_path) as img:
                    width, height = img.size
                    print(f"{i:2d}. {file}")
                    print(f"     尺寸: {width}x{height} | 大小: {file_size:.1f} KB")
            except Exception as e:
                print(f"{i:2d}. {file} (读取失败: {e})")
        
        print("\n" + "=" * 70)
    
    def add_style(self, source_path, rename=None):
        """添加风格图片"""
        if not os.path.exists(source_path):
            print(f"❌ 错误：文件不存在 '{source_path}'")
            return False
        
        # 检查是否为图片
        if not source_path.lower().endswith(self.supported_formats):
            print(f"❌ 错误：不支持的文件格式")
            print(f"   支持的格式: {', '.join(self.supported_formats)}")
            return False
        
        # 确定目标文件名
        if rename:
            # 确保保留扩展名
            _, ext = os.path.splitext(source_path)
            if not rename.lower().endswith(self.supported_formats):
                dest_filename = rename + ext
            else:
                dest_filename = rename
        else:
            dest_filename = os.path.basename(source_path)
        
        dest_path = os.path.join(self.style_dir, dest_filename)
        
        # 检查是否已存在
        if os.path.exists(dest_path):
            response = input(f"⚠️  文件 '{dest_filename}' 已存在，是否覆盖？(y/n): ")
            if response.lower() != 'y':
                print("❌ 取消添加")
                return False
        
        try:
            # 复制文件
            shutil.copy2(source_path, dest_path)
            
            # 显示信息
            with Image.open(dest_path) as img:
                width, height = img.size
                file_size = os.path.getsize(dest_path) / 1024
            
            print("✅ 风格图片已添加到本地库")
            print(f"   文件名: {dest_filename}")
            print(f"   尺寸: {width}x{height}")
            print(f"   大小: {file_size:.1f} KB")
            print(f"   路径: {dest_path}")
            
            return True
            
        except Exception as e:
            print(f"❌ 添加失败: {e}")
            return False
    
    def add_content(self, source_path, rename=None):
        """添加内容图片"""
        if not os.path.exists(source_path):
            print(f"❌ 错误：文件不存在 '{source_path}'")
            return False
        
        # 检查是否为图片
        if not source_path.lower().endswith(self.supported_formats):
            print(f"❌ 错误：不支持的文件格式")
            print(f"   支持的格式: {', '.join(self.supported_formats)}")
            return False
        
        # 确定目标文件名
        if rename:
            _, ext = os.path.splitext(source_path)
            if not rename.lower().endswith(self.supported_formats):
                dest_filename = rename + ext
            else:
                dest_filename = rename
        else:
            dest_filename = os.path.basename(source_path)
        
        dest_path = os.path.join(self.content_dir, dest_filename)
        
        # 检查是否已存在
        if os.path.exists(dest_path):
            response = input(f"⚠️  文件 '{dest_filename}' 已存在，是否覆盖？(y/n): ")
            if response.lower() != 'y':
                print("❌ 取消添加")
                return False
        
        try:
            # 复制文件
            shutil.copy2(source_path, dest_path)
            
            # 显示信息
            with Image.open(dest_path) as img:
                width, height = img.size
                file_size = os.path.getsize(dest_path) / 1024
            
            print("✅ 内容图片已添加到本地库")
            print(f"   文件名: {dest_filename}")
            print(f"   尺寸: {width}x{height}")
            print(f"   大小: {file_size:.1f} KB")
            print(f"   路径: {dest_path}")
            
            return True
            
        except Exception as e:
            print(f"❌ 添加失败: {e}")
            return False
    
    def remove_style(self, filename):
        """删除风格图片"""
        file_path = os.path.join(self.style_dir, filename)
        
        if not os.path.exists(file_path):
            print(f"❌ 错误：文件不存在 '{filename}'")
            return False
        
        response = input(f"⚠️  确定要删除 '{filename}' 吗？(y/n): ")
        if response.lower() != 'y':
            print("❌ 取消删除")
            return False
        
        try:
            os.remove(file_path)
            print(f"✅ 已删除: {filename}")
            return True
        except Exception as e:
            print(f"❌ 删除失败: {e}")
            return False
    
    def remove_content(self, filename):
        """删除内容图片"""
        file_path = os.path.join(self.content_dir, filename)
        
        if not os.path.exists(file_path):
            print(f"❌ 错误：文件不存在 '{filename}'")
            return False
        
        response = input(f"⚠️  确定要删除 '{filename}' 吗？(y/n): ")
        if response.lower() != 'y':
            print("❌ 取消删除")
            return False
        
        try:
            os.remove(file_path)
            print(f"✅ 已删除: {filename}")
            return True
        except Exception as e:
            print(f"❌ 删除失败: {e}")
            return False
    
    def preview_style(self, filename):
        """预览风格图片"""
        file_path = os.path.join(self.style_dir, filename)
        
        if not os.path.exists(file_path):
            print(f"❌ 错误：文件不存在 '{filename}'")
            return False
        
        try:
            img = Image.open(file_path)
            
            plt.figure(figsize=(10, 8))
            plt.imshow(img)
            plt.title(f"风格图片: {filename}\n尺寸: {img.size[0]}x{img.size[1]}")
            plt.axis('off')
            plt.tight_layout()
            plt.show()
            
            return True
        except Exception as e:
            print(f"❌ 预览失败: {e}")
            return False
    
    def preview_content(self, filename):
        """预览内容图片"""
        file_path = os.path.join(self.content_dir, filename)
        
        if not os.path.exists(file_path):
            print(f"❌ 错误：文件不存在 '{filename}'")
            return False
        
        try:
            img = Image.open(file_path)
            
            plt.figure(figsize=(10, 8))
            plt.imshow(img)
            plt.title(f"内容图片: {filename}\n尺寸: {img.size[0]}x{img.size[1]}")
            plt.axis('off')
            plt.tight_layout()
            plt.show()
            
            return True
        except Exception as e:
            print(f"❌ 预览失败: {e}")
            return False
    
    def _get_image_files(self, directory):
        """获取目录中的所有图片文件"""
        if not os.path.exists(directory):
            return []
        
        files = []
        for file in os.listdir(directory):
            if file.lower().endswith(self.supported_formats):
                files.append(file)
        
        return sorted(files)


def main():
    """主函数"""
    parser = argparse.ArgumentParser(
        description='风格图片管理工具',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例用法:
  # 列出所有风格图片
  python manage_styles.py list
  
  # 列出所有内容图片
  python manage_styles.py list-content
  
  # 添加风格图片
  python manage_styles.py add path/to/artwork.jpg
  
  # 添加并重命名
  python manage_styles.py add path/to/artwork.jpg --rename monet_water_lilies.jpg
  
  # 添加内容图片
  python manage_styles.py add-content path/to/photo.jpg
  
  # 预览风格图片
  python manage_styles.py preview monet_water_lilies.jpg
  
  # 删除风格图片
  python manage_styles.py remove old_style.jpg
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='命令')
    
    # list 命令
    subparsers.add_parser('list', help='列出所有风格图片')
    
    # list-content 命令
    subparsers.add_parser('list-content', help='列出所有内容图片')
    
    # add 命令
    add_parser = subparsers.add_parser('add', help='添加风格图片')
    add_parser.add_argument('path', help='图片文件路径')
    add_parser.add_argument('--rename', help='重命名文件', default=None)
    
    # add-content 命令
    add_content_parser = subparsers.add_parser('add-content', help='添加内容图片')
    add_content_parser.add_argument('path', help='图片文件路径')
    add_content_parser.add_argument('--rename', help='重命名文件', default=None)
    
    # remove 命令
    remove_parser = subparsers.add_parser('remove', help='删除风格图片')
    remove_parser.add_argument('filename', help='要删除的文件名')
    
    # remove-content 命令
    remove_content_parser = subparsers.add_parser('remove-content', help='删除内容图片')
    remove_content_parser.add_argument('filename', help='要删除的文件名')
    
    # preview 命令
    preview_parser = subparsers.add_parser('preview', help='预览风格图片')
    preview_parser.add_argument('filename', help='要预览的文件名')
    
    # preview-content 命令
    preview_content_parser = subparsers.add_parser('preview-content', help='预览内容图片')
    preview_content_parser.add_argument('filename', help='要预览的文件名')
    
    args = parser.parse_args()
    
    # 创建管理器
    manager = StyleManager()
    
    # 执行命令
    if args.command == 'list':
        manager.list_styles()
    
    elif args.command == 'list-content':
        manager.list_contents()
    
    elif args.command == 'add':
        manager.add_style(args.path, args.rename)
    
    elif args.command == 'add-content':
        manager.add_content(args.path, args.rename)
    
    elif args.command == 'remove':
        manager.remove_style(args.filename)
    
    elif args.command == 'remove-content':
        manager.remove_content(args.filename)
    
    elif args.command == 'preview':
        manager.preview_style(args.filename)
    
    elif args.command == 'preview-content':
        manager.preview_content(args.filename)
    
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
