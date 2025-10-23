````markdown
# 神经风格迁移 (Neural Style Transfer)

基于PyTorch的神经风格迁移工具，将艺术作品的风格应用到图片和视频。

## ✨ 特性

- 🎨 **风格迁移**: 将艺术风格应用到图片
- 🎯 **风格推荐**: 智能分析并推荐最适合的艺术风格
- 🎬 **视频处理**: 支持视频风格迁移和帧间一致性优化
- 🖥️ **Web界面**: 友好的Gradio GUI
-  **WikiArt数据集**: 支持72个parquet文件，包含27+种艺术风格
- ⚡ **GPU加速**: 支持CUDA和CPU

## 🚀 快速开始

### 安装依赖

```bash
pip install -r requirements.txt
```

### 启动Web界面（推荐）

```bash
python app.py
```

浏览器将自动打开界面，提供三个功能标签：
- 📤 **风格迁移**: 上传或选择图片进行风格迁移
- 🎯 **风格推荐**: 分析图片并获取风格推荐
- 🎬 **视频处理**: 为视频添加艺术风格

### 命令行使用

#### 图像风格迁移
```bash
# 基本使用
python train.py --content photo.jpg --style art.jpg

# 使用WikiArt数据集
python train.py --content photo.jpg --style-name Impressionism

# 高质量输出
python train.py --content photo.jpg --style art.jpg --steps 500 --size 1024
```

#### 视频风格迁移
```bash
# 基本使用（文件在 data/ 目录中）
python video_style_transfer.py my_video.mp4 starry_night.jpg

# 快速测试（只处理前50帧）
python video_style_transfer.py video.mp4 style.jpg -f 50

# 高质量处理
python video_style_transfer.py video.mp4 style.jpg -s 300 --size 1024

# 查看所有选项
python video_style_transfer.py -h
```

#### 图片库管理
```bash
python manage_styles.py list              # 查看风格图片
python manage_styles.py add art.jpg       # 添加风格图片
python manage_styles.py list-content      # 查看内容图片
```

## 📁 项目结构

```
team/
├── app.py                      # Web界面主程序（推荐）
├── train.py                    # 命令行工具
├── manage_styles.py            # 图片库管理
├── batch_process.py            # 批量处理
├── neural_style_transfer.py    # 风格迁移核心
├── style_recommendation.py     # 风格推荐系统
├── video_style_transfer.py     # 视频处理模块
├── dataset.py                  # WikiArt数据集加载
├── config.py                   # 配置管理
├── utils.py                    # 工具函数
├── start.py                    # 环境检查
├── requirements.txt            # 依赖列表
├── README.md                   # 本文件
├── USAGE.md                    # 使用指南
├── PROJECT_INFO.md             # 项目详细信息
│
├── data/
│   ├── content/                # 内容图片库
│   ├── style/                  # 风格图片库
│   └── outputs/                # 输出结果
│
└── wikiart/data/               # WikiArt数据集
```

## 🎯 主要功能

### 1. 图片风格迁移
将艺术风格应用到图片，支持：
- 上传自定义图片
- 使用本地图片库
- 使用WikiArt数据集（27+种艺术风格）

### 2. 风格推荐系统
- 自动分析图片色彩特征（色调、饱和度、亮度）
- 智能推荐Top 5-10个最适合的风格
- 显示匹配度和推荐理由

### 3. 视频风格迁移
- 逐帧处理视频
- 帧间一致性优化（减少闪烁）
- 断点续传支持（处理中断后可继续）
- 实时进度显示和时间预估
- 灵活的命令行参数（分辨率、步数、帧数限制等）

**注意**: 视频处理较耗时，建议：
- 先用 `-f 30` 测试前30帧
- 使用 `--size 256` 降低分辨率加快速度
- GPU处理速度约为CPU的5-10倍

**处理时间参考**（30秒视频，30fps）：
- 快速模式（256px, 100步）: 约1-1.5小时
- 标准模式（512px, 150步）: 约2.5-3.5小时
- 高质量（1024px, 300步）: 约8-12小时

## 📊 WikiArt数据集

支持27+种艺术风格，包括：
- 印象派 (Impressionism)
- 后印象派 (Post-Impressionism)
- 立体派 (Cubism)
- 表现主义 (Expressionism)
- 浮世绘 (Ukiyo-e)
- 等等...

数据集下载：
```bash
pip install -U huggingface_hub
export HF_ENDPOINT=https://hf-mirror.com
huggingface-cli download huggan/wikiart --repo-type dataset --local-dir ./wikiart
```

## 🔧 系统要求

- Python 3.8+
- PyTorch 2.0+
- 推荐：NVIDIA GPU（快5-10倍）
- 最低：8GB RAM

## ⚡ 处理速度参考

### 图像处理

| 配置 | 图像大小 | 步数 | GPU耗时 | CPU耗时 |
|------|---------|------|---------|---------|
| 快速 | 256×256 | 100 | ~30秒 | ~2分钟 |
| 标准 | 512×512 | 200 | ~1分钟 | ~5分钟 |
| 高质量 | 512×512 | 300 | ~2分钟 | ~10分钟 |

### 视频处理

**每帧处理时间**（GPU）：
- 256px, 100步: 约3-5秒/帧
- 512px, 150步: 约8-12秒/帧
- 1024px, 300步: 约30-50秒/帧

**完整视频示例**（30秒，30fps = 900帧）：
- 快速模式: 约1-1.5小时
- 标准模式: 约2.5-3.5小时
- 高质量: 约8-12小时

💡 **建议**：先用 `-f 30` 处理30帧测试，根据耗时估算完整视频所需时间

## ❓ 常见问题

**Q: 需要GPU吗？**  
A: 不是必须。GPU快很多，但CPU也能用。视频处理强烈推荐GPU。

**Q: 如何加快速度？**  
A: 减小图像尺寸（256）、减少步数（100）、使用GPU。

**Q: 第一次运行会下载什么？**  
A: PyTorch自动下载VGG19模型（~500MB），仅首次需要。

**Q: 视频处理中断了怎么办？**  
A: 支持断点续传。重新运行相同命令，会自动从中断处继续。

## 📚 文档

- **使用指南**: [USAGE.md](USAGE.md) - 详细使用说明和参数调优
- **图片库管理**: [data/ORGANIZATION_GUIDE.md](data/ORGANIZATION_GUIDE.md) - 图片库组织建议

---

**享受艺术创作！** 🎨
