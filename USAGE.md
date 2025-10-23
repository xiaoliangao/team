# 使用指南

## 快速导航

1. [环境配置](#环境配置)
2. [Web界面使用](#web界面使用)
3. [命令行使用](#命令行使用)
4. [参数调优](#参数调优)
5. [常见问题](#常见问题)

---

## 环境配置

```bash
# 安装依赖
pip install -r requirements.txt

# 检查CUDA（可选）
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}')"
```

---

## Web界面使用

### 启动

```bash
python app.py
```

浏览器自动打开，提供三个功能标签：

### 📤 风格迁移
1. 上传内容图和风格图（或从本地库/WikiArt选择）
2. 调整参数：风格强度(1-10)、步数(100-500)、尺寸(256/512/1024)
3. 开始处理，等待完成后下载

### 🎯 风格推荐
1. 上传图片
2. 点击分析，获取Top 5-10推荐风格
3. 选择推荐的风格进行迁移

### 🎬 视频处理
1. 上传视频和风格图
2. 设置参数（建议：256px, 150步, 启用帧间一致性）
3. 先限制最大帧数测试（如100帧）
4. 查看进度，完成后下载

**注意**：视频处理耗时长，10秒视频约需30-60分钟（GPU）

---

## 命令行使用

### 图像风格迁移

#### 基本命令

```bash
# 使用本地图片
python train.py --content photo.jpg --style art.jpg

# 使用WikiArt数据集
python train.py --content photo.jpg --style-name Impressionism

# 指定艺术家
python train.py --content photo.jpg --artist "Vincent van Gogh"

# 高质量输出
python train.py --content photo.jpg --style art.jpg --steps 500 --size 1024
```

### 视频风格迁移

#### 基本命令

```bash
# 最简单的用法（文件在 data/ 对应目录中）
python video_style_transfer.py my_video.mp4 starry_night.jpg

# 使用完整路径
python video_style_transfer.py data/content/video.mp4 data/style/style.jpg

# 自定义输出路径
python video_style_transfer.py video.mp4 style.jpg -o my_output.mp4

# 快速测试（只处理前50帧）
python video_style_transfer.py video.mp4 style.jpg -f 50

# 高质量处理
python video_style_transfer.py video.mp4 style.jpg -s 300 --size 1024

# 禁用帧间一致性（更快但可能闪烁）
python video_style_transfer.py video.mp4 style.jpg --no-consistency
```

#### 视频处理参数

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `video` | 输入视频文件（必需） | - |
| `style` | 风格图像文件（必需） | - |
| `-o, --output` | 输出视频路径 | 自动生成时间戳 |
| `-s, --steps` | 优化步数 | 150 |
| `--size` | 处理图像大小 | 512 |
| `-f, --max-frames` | 最大处理帧数 | 无限制 |
| `--style-weight` | 风格权重 | 1e6 |
| `--content-weight` | 内容权重 | 1.0 |
| `--temporal-weight` | 时间一致性权重 | 1e4 |
| `--no-consistency` | 禁用帧间一致性 | False |
| `--device` | 计算设备 | 自动检测 |

#### 视频处理示例

```bash
# 查看帮助
python video_style_transfer.py -h

# 快速预览（低分辨率，少量帧）
python video_style_transfer.py video.mp4 style.jpg --size 256 -f 30

# 平衡质量（推荐）
python video_style_transfer.py video.mp4 style.jpg -s 150 --size 512

# 高质量输出
python video_style_transfer.py video.mp4 style.jpg -s 300 --size 1024

# 超快速测试（无一致性优化）
python video_style_transfer.py video.mp4 style.jpg -f 20 --no-consistency

# 完整配置示例
python video_style_transfer.py \
  data/content/my_video.mp4 \
  data/style/starry_night.jpg \
  -o data/outputs/result.mp4 \
  -s 200 \
  --size 768 \
  -f 300 \
  --style-weight 1e6 \
  --temporal-weight 5e3
```

#### 自动文件命名

如果不指定输出路径，系统会自动生成带时间戳的文件名：
```
data/outputs/styled_{视频名}_{风格名}_{时间戳}.mp4
```

例如：
```
data/outputs/styled_my_video_starry_night_20251023_143052.mp4
```

### 主要参数

#### 图像处理参数

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--content` | 内容图片路径 | 必需 |
| `--style` | 风格图片路径 | None |
| `--style-name` | WikiArt风格名称 | None |
| `--steps` | 迭代步数 | 300 |
| `--style-weight` | 风格权重 | 1e6 |
| `--size` | 图像尺寸 | 512 |
| `--output` | 输出目录 | output |

### 图片库管理

```bash
# 查看
python manage_styles.py list                    # 风格图片
python manage_styles.py list-content            # 内容图片

# 添加
python manage_styles.py add artwork.jpg
python manage_styles.py add-content photo.jpg

# 删除
python manage_styles.py remove old_style.jpg
```

---

## 参数调优

### 图像处理推荐配置

| 场景 | 步数 | 尺寸 | 风格权重 | 耗时(GPU) |
|------|------|------|----------|-----------|
| 快速测试 | 100 | 256 | 1e6 | ~30秒 |
| 日常使用 | 200 | 512 | 1e6 | ~1分钟 |
| 高质量 | 300 | 512 | 5e6 | ~2分钟 |
| 专业级 | 500 | 1024 | 5e6 | ~5分钟 |

### 视频处理推荐配置

| 场景 | 步数 | 尺寸 | 帧间一致性 | 最大帧数 | 预估时间(GPU) |
|------|------|------|-----------|---------|--------------|
| 快速预览 | 100 | 256 | 禁用 | 30 | ~5分钟 |
| 测试运行 | 150 | 512 | 启用 | 100 | ~30分钟 |
| 标准质量 | 150 | 512 | 启用 | 无限制 | ~2-5小时* |
| 高质量 | 200 | 768 | 启用 | 无限制 | ~5-10小时* |
| 专业级 | 300 | 1024 | 启用 | 无限制 | ~10-20小时* |

*时间取决于视频长度和帧率，按30fps、30秒视频估算

### 视频处理时间估算

- **256px, 100步**: 每帧约3-5秒
- **512px, 150步**: 每帧约8-12秒
- **1024px, 300步**: 每帧约30-50秒

**示例**：处理一个30秒、30fps的视频（900帧）
- 快速模式(256px, 100步): 约1-1.5小时
- 标准模式(512px, 150步): 约2.5-3.5小时  
- 高质量(1024px, 300步): 约8-12小时

### 参数效果

**风格权重**：
- `1e5` - 弱风格，保留更多原图
- `1e6` - 平衡（推荐）
- `1e7` - 强风格，艺术感更强

**迭代步数**：
- `100` - 快速预览
- `300` - 标准质量
- `500` - 高质量

**图像尺寸**：
- `256` - 快速，低内存
- `512` - 推荐
- `1024` - 需GPU和大内存

---

## 常见问题

### 图像处理

#### Q1: 显存不足
```bash
# 减小尺寸
python train.py --content photo.jpg --style art.jpg --size 256

# 使用CPU
python train.py --content photo.jpg --style art.jpg --device cpu
```

#### Q2: 风格太强/太弱
```bash
# 风格太强
python train.py --content photo.jpg --style art.jpg --style-weight 5e5

# 风格太弱
python train.py --content photo.jpg --style art.jpg --style-weight 5e6
```

#### Q3: 处理太慢
```bash
# 减少步数和尺寸
python train.py --content photo.jpg --style art.jpg --steps 100 --size 256
```

### 视频处理

#### Q4: 视频处理中断怎么办？
视频处理支持**断点续传**功能：
- 每处理10帧自动保存进度
- 重新运行相同命令会从上次中断处继续
- 检查点保存在 `data/outputs/work_*/checkpoint.json`

```bash
# 如果中断，直接重新运行相同命令即可继续
python video_style_transfer.py video.mp4 style.jpg -o output.mp4
```

#### Q5: 视频处理太慢
```bash
# 方法1: 减小分辨率和步数
python video_style_transfer.py video.mp4 style.jpg --size 256 -s 100

# 方法2: 先处理少量帧测试
python video_style_transfer.py video.mp4 style.jpg -f 50

# 方法3: 禁用帧间一致性（快2-3倍，但可能闪烁）
python video_style_transfer.py video.mp4 style.jpg --no-consistency
```

#### Q6: 视频输出有闪烁
```bash
# 确保启用帧间一致性（默认启用）
python video_style_transfer.py video.mp4 style.jpg

# 增加时间一致性权重
python video_style_transfer.py video.mp4 style.jpg --temporal-weight 5e4
```

#### Q7: 找不到视频或风格图像
确保文件在正确的目录中：
```bash
# 视频放在这里
data/content/your_video.mp4

# 风格图像放在这里
data/style/your_style.jpg

# 然后使用文件名即可
python video_style_transfer.py your_video.mp4 your_style.jpg
```

或使用完整路径：
```bash
python video_style_transfer.py /path/to/video.mp4 /path/to/style.jpg
```

#### Q8: 如何估算处理时间？
运行脚本后会显示每帧处理时间和预计剩余时间：
```
帧 10/300 完成 (3.3%) - 耗时: 8.5s - 预计剩余: 41.2分钟
```

建议先用 `-f 10` 处理10帧测试，根据耗时估算总时间。

### 通用问题

#### Q9: GUI打不开
```bash
pip install gradio
python app.py
# 手动访问：http://127.0.0.1:7860
```

---

## 工作流程

### 新手入门

#### 图像风格迁移
1. `pip install -r requirements.txt`
2. `python app.py`
3. 上传图片，使用默认参数
4. 下载结果

#### 视频风格迁移（命令行）
1. 准备视频文件，放在 `data/content/` 目录
2. 准备风格图像，放在 `data/style/` 目录
3. 快速测试：
   ```bash
   python video_style_transfer.py your_video.mp4 style.jpg -f 30
   ```
4. 满意后处理完整视频：
   ```bash
   python video_style_transfer.py your_video.mp4 style.jpg
   ```

### 进阶使用

#### 图像批量处理
1. 准备图片库（`data/content/`, `data/style/`）
2. 批量处理：`python batch_process.py --mode folder`
3. 参数实验：尝试不同配置
4. 整理结果

#### 视频处理最佳实践
1. **第一步：快速预览**
   ```bash
   # 处理前30帧，低分辨率
   python video_style_transfer.py video.mp4 style.jpg --size 256 -f 30
   ```

2. **第二步：参数调优**
   - 查看预览效果
   - 根据需要调整风格权重、步数等参数
   - 再次测试少量帧

3. **第三步：完整处理**
   ```bash
   # 使用优化后的参数处理完整视频
   python video_style_transfer.py video.mp4 style.jpg -s 150 --size 512
   ```

4. **第四步：管理输出**
   - 输出保存在 `data/outputs/`
   - 自动生成带时间戳的文件名
   - 可以手动指定输出名称

### 目录结构

```
project/
├── data/
│   ├── content/          # 放置内容图像和视频
│   ├── style/            # 放置风格图像
│   └── outputs/          # 处理结果输出
│       └── work_*/       # 视频处理工作目录（自动创建）
├── train.py              # 图像风格迁移命令行
├── video_style_transfer.py  # 视频风格迁移命令行
└── app.py               # Web界面
```

---

**享受创作！** 🎨