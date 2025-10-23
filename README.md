# 神经风格迁移 (Neural Style Transfer)

基于PyTorch的神经风格迁移工具，将艺术作品的风格应用到普通照片上。

## ✨ 特性

- 🎨 **风格迁移**: 将艺术作品风格应用到任意图片
- 🖥️ **Web界面**: 友好的Gradio GUI，支持多种选择方式
- 📁 **本地图片库**: 管理你的照片和艺术作品收藏
- 📊 **WikiArt数据集**: 支持72个parquet文件，包含丰富的艺术作品
- ⚡ **GPU加速**: 支持CUDA，也可使用CPU
- 🔧 **灵活配置**: 命令行和GUI多种使用方式

## 🚀 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 启动Web GUI（推荐）

```bash
python app.py
```

浏览器自动打开 → 选择图片（上传或从本地库选择）→ 开始处理！

### 3. 使用本地图片库

```bash
# 添加艺术作品到风格库
python manage_styles.py add path/to/artwork.jpg

# 添加照片到内容库
python manage_styles.py add-content path/to/photo.jpg

# 查看图片库
python manage_styles.py list
python manage_styles.py list-content
```

### 4. 命令行使用

```bash
# 使用本地风格图片
python train.py --content data/content/photo.jpg --style data/style/art.jpg

# 使用WikiArt数据集
python train.py --content photo.jpg --style-name Impressionism
```

## 📁 项目结构

```
team/
├── start.py                    # ⚡ 快速启动脚本（检查环境）
├── app.py                      # ⭐ Web GUI应用（推荐入口）
├── neural_style_transfer.py    # 核心算法
├── train.py                    # 命令行处理脚本
├── batch_process.py            # 批量处理
├── manage_styles.py            # 📁 图片库管理工具（新增）
├── dataset.py                  # WikiArt数据集加载
├── config.py                   # 配置
├── utils.py                    # 工具函数
├── requirements.txt            # 依赖列表
│
├── README.md                   # 项目介绍（本文件）
├── USAGE.md                    # 详细使用指南
├── QUICKSTART.md               # 🚀 快速开始指南（新增）
│
├── data/
│   ├── content/                # 📷 内容图片库（新增）
│   ├── style/                  # 🎨 风格图片库（新增）
│   ├── outputs/                # 输出结果
│   └── ORGANIZATION_GUIDE.md   # 📁 图片库组织指南（新增）
│
└── wikiart/data/               # WikiArt数据集（72个parquet文件）
```

## 🎯 核心功能

### 三种使用方式

#### 1. 直接上传图片（最简单）
- Web界面上传内容图和风格图
- 适合快速体验和一次性使用

#### 2. 本地图片库（最方便）
- 📁 管理你的照片集和艺术作品收藏
- 🔄 快速复用常用图片
- 📊 批量处理多种组合
- 使用 `manage_styles.py` 工具管理

#### 3. WikiArt数据集（最丰富）
- 📚 27+种艺术风格（印象派、立体派等）
- 🎨 数千件艺术作品可选
- 🎲 随机选择或指定风格/艺术家

### Web GUI模式（推荐）

启动 `app.py` 后提供：

- 📤 上传内容图和风格图
- 📂 从本地图片库选择（`data/content/` 和 `data/style/`）
- 📚 从WikiArt数据集选择艺术风格
- ⚙️ 调节参数：风格强度、迭代步数、图像大小
- 💾 自动保存结果

### 命令行模式

- 单图处理：`train.py`
- 批量处理：`batch_process.py`
- 图片管理：`manage_styles.py`

## 🛠️ 技术原理

1. **预训练VGG19** - 使用ImageNet训练好的模型提取特征
2. **固定网络** - VGG19权重不变，无需更新
3. **优化图像** - 调整图像像素（不是网络参数）
4. **L-BFGS优化器** - 迭代优化图像使其同时保留内容和风格

**工作流程**：
```
内容图 + 风格图 → VGG19特征提取 → 计算损失 → 优化图像像素 → 生成结果
```

**核心组件**：
- VGG19网络：提取特征（预训练，固定权重）
- 内容损失：MSE(内容特征, 生成图特征)
- 风格损失：MSE(Gram(风格), Gram(生成图))
- L-BFGS：优化算法（优化图像，不是网络）

## 📖 WikiArt数据集

支持使用WikiArt数据集（72个parquet文件），包含：

- 印象派 (Impressionism)
- 后印象派 (Post-Impressionism)
- 立体派 (Cubism)
- 表现主义 (Expressionism)
- 浮世绘 (Ukiyo-e)
- 等多种艺术风格...

数据集路径：`wikiart/data/train-*-of-00072.parquet`
数据集下载方式：
```bash
pip install -U huggingface_hub

export HF_ENDPOINT=https://hf-mirror.com

huggingface-cli download huggan/wikiart --repo-type dataset --local-dir ./wikiart --local-dir-use-symlinks False
```

## 🔧 系统要求

- Python 3.8+
- PyTorch 2.0+
- 推荐：NVIDIA GPU（快5-10倍）
- 最低：8GB RAM

## ⚡ 处理速度参考

| 配置 | 图像大小 | 步数 | GPU耗时 | CPU耗时 |
|------|---------|------|---------|---------|
| 快速 | 256×256 | 100 | ~30秒 | ~2分钟 |
| 标准 | 512×512 | 200 | ~1分钟 | ~5分钟 |
| 高质量 | 512×512 | 300 | ~2分钟 | ~10分钟 |

## 📚 使用文档

- **快速开始**: [QUICKSTART.md](QUICKSTART.md) - 5分钟快速上手
- **详细指南**: [USAGE.md](USAGE.md) - 完整使用流程
- **图片库管理**: [data/ORGANIZATION_GUIDE.md](data/ORGANIZATION_GUIDE.md) - 图片库组织建议

## ❓ 常见问题

**Q: 第一次运行会下载什么？**  
A: PyTorch会自动下载预训练的VGG19模型（~500MB），仅首次需要。

**Q: 需要GPU吗？**  
A: 不是必须的。GPU快很多（1分钟 vs 5分钟），但CPU也能用。

**Q: 如何加快速度？**  
A: 减小图像尺寸（256）、减少步数（100）、使用GPU。

**详细使用说明请查看 [USAGE.md](USAGE.md)**
