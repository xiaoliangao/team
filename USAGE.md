## 目录

1. [环境配置](#1-环境配置)
2. [Web GUI使用](#2-web-gui使用推荐)
3. [命令行使用](#3-命令行使用)
4. [本地风格图片库](#4-本地风格图片库)
5. [WikiArt数据集](#5-wikiart数据集)
6. [参数调优](#6-参数调优)
7. [批量处理](#7-批量处理)
8. [技术原理](#8-技术原理-为什么不需要训练)
9. [常见问题](#9-常见问题)

---

## 1. 环境配置

### 安装步骤

```bash
# 1. 进入项目目录
cd team

# 2. 安装依赖
pip install -r requirements.txt

# 3. (可选) 如果使用GPU，确认CUDA可用
python -c "import torch; print(f'CUDA可用: {torch.cuda.is_available()}')"
```

### 依赖说明

- **torch**: PyTorch深度学习框架
- **torchvision**: 预训练模型和图像处理
- **gradio**: Web GUI界面
- **pillow**: 图像处理
- **pandas/pyarrow**: WikiArt数据集加载

---

## 2. Web GUI使用（推荐）

### 启动界面

```bash
python app.py
```

启动后会自动打开浏览器，显示Web界面。

### 界面操作

#### 方式1: 直接上传图片

1. **上传内容图片**: 点击"上传内容图片"，选择你的照片
2. **上传风格图片**: 点击"上传风格图片"，选择艺术作品
3. **调整参数** (可选):
   - 风格强度: 1-10 (默认5)
   - 迭代步数: 100-500 (默认200)
   - 图像大小: 256/512/1024 (默认512)
4. **点击"开始风格迁移"**
5. **等待处理完成**，结果会显示在右侧
6. **下载结果**: 点击下载按钮保存图片

#### 方式2: 使用本地图片库

1. **从本地内容库选择**: 点击"从本地内容库选择"，从下拉菜单选择你的照片
2. **从本地风格库选择**: 点击"从本地风格库选择"，从下拉菜单选择风格图片
3. **调整参数并开始处理**

> 💡 **提示**: 将照片放入 `data/content/` 文件夹，将艺术作品放入 `data/style/` 文件夹，即可快速选择使用。

#### 方式3: 使用WikiArt数据集

1. **上传内容图片**
2. **从WikiArt数据集选择**: 点击"从 WikiArt 数据集选择"
3. **选择艺术风格**: 如"印象派 (Impressionism)"、"立体主义 (Cubism)"等
4. **自动加载风格图片**: 系统会从WikiArt数据集中随机选择该风格的艺术作品
5. **调整参数并开始处理**

### GUI参数说明

| 参数 | 范围 | 推荐值 | 说明 |
|------|------|--------|------|
| 风格强度 | 1-10 | 5 | 值越大风格越强，内容越弱 |
| 迭代步数 | 100-500 | 200 | 步数越多质量越好但耗时越长 |
| 图像大小 | 256/512/1024 | 512 | 尺寸越大质量越好但需要更多内存 |

### 处理时间参考

| 配置 | 快速预览 | 标准质量 | 高质量 |
|------|---------|---------|--------|
| 图像大小 | 256 | 512 | 1024 |
| 迭代步数 | 100 | 200 | 300 |
| GPU耗时 | ~30秒 | ~1分钟 | ~3分钟 |
| CPU耗时 | ~2分钟 | ~5分钟 | ~15分钟 |

---

## 3. 命令行使用

### 基础用法

#### 使用自己的风格图片

```bash
python train.py \
    --content photo.jpg \
    --style your_art.jpg \
    --output results
```

#### 使用WikiArt数据集

```bash
# 随机选择风格
python train.py --content photo.jpg --output results

# 指定风格类型
python train.py \
    --content photo.jpg \
    --style-name Impressionism \
    --output results

# 指定艺术家
python train.py \
    --content photo.jpg \
    --artist "Vincent van Gogh" \
    --output results
```

### 命令行参数

| 参数 | 类型 | 必需 | 默认值 | 说明 |
|------|------|------|--------|------|
| `--content` | str | ✓ | - | 内容图片路径 |
| `--style` | str | | None | 风格图片路径 |
| `--output` | str | | output | 输出目录 |
| `--steps` | int | | 300 | 迭代步数 |
| `--style-weight` | float | | 1e6 | 风格权重 |
| `--content-weight` | float | | 1 | 内容权重 |
| `--size` | int | | 512 | 图像尺寸 |
| `--device` | str | | auto | cpu/cuda/auto |
| `--style-name` | str | | None | WikiArt风格名称 |
| `--artist` | str | | None | WikiArt艺术家名称 |

### 命令行示例

```bash
# 快速预览（低分辨率）
python train.py --content photo.jpg --style art.jpg \
    --steps 100 --size 256

# 标准质量
python train.py --content photo.jpg --style art.jpg \
    --steps 300 --size 512

# 高质量（需要GPU）
python train.py --content photo.jpg --style art.jpg \
    --steps 500 --size 1024 --style-weight 5e6

# 强调内容保留
python train.py --content photo.jpg --style art.jpg \
    --style-weight 5e5 --content-weight 10

# 强调风格效果
python train.py --content photo.jpg --style art.jpg \
    --style-weight 5e6 --content-weight 0.1
```

---

## 4. 本地风格图片库

### 目录结构

项目包含两个本地图片库：

```
data/
├── content/          # 内容图片库（你的照片）
│   ├── README.md
│   └── *.jpg
├── style/            # 风格图片库（艺术作品）
│   ├── README.md
│   └── *.jpg
└── outputs/          # 处理结果输出
```

### 管理风格图片

使用 `manage_styles.py` 工具管理本地图片库：

#### 查看图片列表

```bash
# 查看所有风格图片
python manage_styles.py list

# 查看所有内容图片
python manage_styles.py list-content
```

#### 添加图片

```bash
# 添加风格图片（艺术作品）
python manage_styles.py add path/to/artwork.jpg

# 添加并重命名
python manage_styles.py add path/to/artwork.jpg --rename monet_water_lilies.jpg

# 添加内容图片（照片）
python manage_styles.py add-content path/to/photo.jpg --rename my_photo.jpg
```

#### 预览图片

```bash
# 预览风格图片
python manage_styles.py preview monet_water_lilies.jpg

# 预览内容图片
python manage_styles.py preview-content my_photo.jpg
```

#### 删除图片

```bash
# 删除风格图片
python manage_styles.py remove old_style.jpg

# 删除内容图片
python manage_styles.py remove-content old_photo.jpg
```

### 本地库使用示例

#### Web GUI中使用

1. 启动界面：`python app.py`
2. 在"从本地风格库选择"下拉菜单中选择风格图片
3. 在"从本地内容库选择"下拉菜单中选择内容图片
4. 开始处理

#### 命令行中使用

```bash
# 使用本地库的图片
python train.py \
    --content data/content/my_photo.jpg \
    --style data/style/monet_water_lilies.jpg \
    --output results
```

### 推荐的风格图片

#### 经典艺术作品

1. **印象派**:
   - 莫奈《睡莲》、《日出·印象》
   - 雷诺阿《煎饼磨坊的舞会》

2. **后印象派**:
   - 梵高《星空》、《向日葵》
   - 塞尚《圣维克多山》

3. **立体派**:
   - 毕加索《哭泣的女人》、《亚威农的少女》
   - 布拉克的作品

4. **浮世绘**:
   - 葛饰北斋《神奈川冲浪里》
   - 歌川广重的风景画

5. **表现主义**:
   - 蒙克《呐喊》
   - 康定斯基的抽象作品

#### 获取艺术作品的网站

- [WikiArt](https://www.wikiart.org/) - 大量免费艺术作品
- [The Met Collection](https://www.metmuseum.org/art/collection) - 大都会博物馆
- [Google Arts & Culture](https://artsandculture.google.com/)
- [Rijksmuseum](https://www.rijksmuseum.nl/) - 荷兰国家博物馆

### 文件命名建议

使用有意义的文件名方便识别和管理：

```
data/style/
├── monet_water_lilies.jpg         # 莫奈-睡莲
├── vangogh_starry_night.jpg       # 梵高-星空
├── hokusai_great_wave.jpg         # 北斋-神奈川冲浪里
├── picasso_weeping_woman.jpg      # 毕加索-哭泣的女人
└── kandinsky_composition_8.jpg    # 康定斯基-构成8号

data/content/
├── portrait_01.jpg                # 人像照片
├── landscape_mountain.jpg         # 山景照片
├── city_night.jpg                 # 城市夜景
└── pet_cat.jpg                    # 宠物照片
```

---

## 5. WikiArt数据集

### 数据集结构

项目包含72个parquet文件，位于 `wikiart/data/`：
```
wikiart/data/
├── train-00000-of-00072.parquet
├── train-00001-of-00072.parquet
├── ...
└── train-00071-of-00072.parquet
```

### 支持的艺术风格

使用以下命令查看可用风格：

```bash
python dataset.py
```

常见风格包括：
- **Impressionism** (印象派): 莫奈、雷诺阿
- **Post-Impressionism** (后印象派): 梵高、塞尚
- **Cubism** (立体派): 毕加索、布拉克
- **Expressionism** (表现主义): 蒙克、康定斯基
- **Ukiyo-e** (浮世绘): 葛饰北斋
- **Realism** (写实主义)
- **Romanticism** (浪漫主义)
- 等等...

### WikiArt使用示例

```bash
# 使用印象派风格
python train.py --content photo.jpg --style-name Impressionism

# 使用梵高的作品
python train.py --content photo.jpg --artist "Vincent van Gogh"

# 随机选择一个风格
python train.py --content photo.jpg
```

---

## 6. 参数调优

### 参数影响说明

#### 风格权重 (style_weight)

控制风格的强度：

```bash
# 弱风格 (保留更多原图)
python train.py --content photo.jpg --style art.jpg --style-weight 1e5

# 平衡 (推荐)
python train.py --content photo.jpg --style art.jpg --style-weight 1e6

# 强风格 (艺术感更强)
python train.py --content photo.jpg --style art.jpg --style-weight 1e7
```

#### 迭代步数 (steps)

影响质量和耗时：

```bash
# 快速测试
python train.py --content photo.jpg --style art.jpg --steps 100

# 标准质量
python train.py --content photo.jpg --style art.jpg --steps 300

# 高质量
python train.py --content photo.jpg --style art.jpg --steps 500
```

#### 图像尺寸 (size)

影响细节和内存占用：

```bash
# 小尺寸 (快速、低内存)
python train.py --content photo.jpg --style art.jpg --size 256

# 中等尺寸 (推荐)
python train.py --content photo.jpg --style art.jpg --size 512

# 大尺寸 (需要GPU和大内存)
python train.py --content photo.jpg --style art.jpg --size 1024
```

### 推荐配置

根据不同场景选择配置：

| 场景 | steps | size | style_weight | 耗时(GPU) |
|------|-------|------|--------------|-----------|
| 快速测试 | 100 | 256 | 1e6 | ~30秒 |
| 日常使用 | 200 | 512 | 1e6 | ~1分钟 |
| 高质量输出 | 300 | 512 | 5e6 | ~2分钟 |
| 专业级 | 500 | 1024 | 5e6 | ~5分钟 |

---

## 7. 批量处理

### 批量处理文件夹

```bash
python batch_process.py \
    --mode folder \
    --content-dir data/content \
    --style-dir data/style \
    --output batch_results
```

### 使用WikiArt批量处理

```bash
python batch_process.py \
    --mode wikiart \
    --content-dir data/content \
    --styles "Impressionism,Cubism,Expressionism" \
    --output batch_results
```

### 参数对比实验

```bash
python experiments.py \
    --content photo.jpg \
    --style art.jpg \
    --experiment all
```

这会生成多组对比实验，包括：
- 不同风格权重的效果
- 不同迭代步数的效果
- 不同图像尺寸的效果

---

## 8. 技术原理 (为什么不需要训练)

### 工作原理

**神经风格迁移 ≠ 模型训练**

传统深度学习项目需要训练模型参数，但本项目不同：

#### 传统训练项目
```
数据集 → 训练模型参数 → 保存模型 → 推理
```

#### 本项目（风格迁移）
```
预训练VGG19(固定) + 内容图 + 风格图 → 优化图像像素 → 生成结果
```

### 关键区别

| 项目类型 | 优化对象 | 是否需要大量数据 | 是否保存模型 |
|---------|---------|----------------|-------------|
| 传统训练 | 网络参数（权重） | ✓ 需要 | ✓ 需要 |
| 风格迁移 | **图像像素** | ✗ 不需要 | ✗ 不需要 |

### 技术细节

1. **预训练VGG19**
   - 已在ImageNet上训练好
   - 权重固定，不更新
   - 仅用于提取特征

2. **优化目标**
   - 不是训练网络
   - 而是调整图像的像素值
   - 使图像同时保留内容和风格

3. **损失函数**
   ```
   总损失 = α × 内容损失 + β × 风格损失
   
   内容损失 = MSE(VGG(内容图), VGG(生成图))
   风格损失 = MSE(Gram(VGG(风格图)), Gram(VGG(生成图)))
   ```

4. **优化过程**
   - 使用L-BFGS优化器
   - 迭代更新：**生成图的像素**
   - VGG19网络权重保持不变

### 为什么这样设计？

- ✅ **快速**: 无需长时间训练，几分钟即可
- ✅ **灵活**: 每次可用不同的风格和内容
- ✅ **通用**: 一个预训练模型适用所有风格
- ✅ **简单**: 不需要准备训练数据集

---

## 9. 常见问题

### Q1: 显存不足 (CUDA out of memory)

**解决方案**:
```bash
# 减小图像尺寸
python train.py --content photo.jpg --style art.jpg --size 256

# 或使用CPU
python train.py --content photo.jpg --style art.jpg --device cpu
```

### Q2: 风格太强或太弱

**解决方案**:
```bash
# 风格太强，减小style_weight
python train.py --content photo.jpg --style art.jpg --style-weight 5e5

# 风格太弱，增大style_weight
python train.py --content photo.jpg --style art.jpg --style-weight 5e6
```

### Q3: 处理速度太慢

**解决方案**:
```bash
# 减少迭代步数
python train.py --content photo.jpg --style art.jpg --steps 100

# 减小图像尺寸
python train.py --content photo.jpg --style art.jpg --size 256

# 确认使用GPU
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}')"
```

### Q4: WikiArt数据集找不到

**解决方案**:

1. 检查文件是否存在：
```bash
ls -l wikiart/data/*.parquet
```

2. 如果没有数据集，可以手动上传风格图片：
```bash
python train.py --content photo.jpg --style your_art.jpg
```

### Q5: 结果有噪点或异常

**解决方案**:
```bash
# 增加迭代步数
python train.py --content photo.jpg --style art.jpg --steps 500

# 调整权重平衡
python train.py --content photo.jpg --style art.jpg \
    --style-weight 1e6 --content-weight 1
```

### Q6: GUI界面打不开

**解决方案**:
```bash
# 检查gradio是否安装
pip install gradio

# 重新启动
python app.py

# 如果自动打开失败，手动访问显示的URL
# 通常是 http://127.0.0.1:7860
```

---

## 完整工作流程示例

### 新手推荐流程

1. **安装环境**
   ```bash
   pip install -r requirements.txt
   ```

2. **启动GUI**
   ```bash
   python app.py
   ```

3. **上传图片并处理**
   - 上传你的照片
   - 从WikiArt数据集选择艺术风格或上传自己的艺术作品
   - 使用默认参数开始处理

4. **下载结果**

### 进阶用户流程

1. **准备多张图片**
   ```bash
   mkdir -p data/content data/style
   # 将图片放入对应目录
   ```

2. **批量处理**
   ```bash
   python batch_process.py \
       --mode folder \
       --content-dir data/content \
       --style-dir data/style
   ```

3. **参数实验**
   ```bash
   python experiments.py \
       --content best_photo.jpg \
       --style best_art.jpg
   ```

4. **整理结果**
   ```bash
   # 结果在 batch_results/ 目录
   ```

---

## 技术支持

如有问题，请检查：
1. Python版本是否≥3.8
2. PyTorch是否正确安装
3. 图片路径是否正确
4. 是否有足够的内存/显存

**享受艺术创作的乐趣！** 🎨
