# 本地图片库组织示例

本文档展示如何有效组织和管理你的本地图片库。

## 📁 推荐的目录结构

### 基础结构

```
data/
├── content/                    # 内容图片库
│   ├── README.md
│   ├── portrait_01.jpg
│   ├── landscape_mountain.jpg
│   └── city_night.jpg
├── style/                      # 风格图片库
│   ├── README.md
│   ├── monet_water_lilies.jpg
│   ├── vangogh_starry_night.jpg
│   └── hokusai_great_wave.jpg
└── outputs/                    # 输出结果
    └── result_*.jpg
```

### 进阶结构（分类管理）

```
data/
├── content/
│   ├── README.md
│   ├── portraits/              # 人像照片
│   │   ├── person_01.jpg
│   │   └── person_02.jpg
│   ├── landscapes/             # 风景照片
│   │   ├── mountain.jpg
│   │   ├── lake.jpg
│   │   └── forest.jpg
│   ├── urban/                  # 城市照片
│   │   ├── city_night.jpg
│   │   └── street.jpg
│   └── animals/                # 动物照片
│       ├── cat.jpg
│       └── dog.jpg
├── style/
│   ├── README.md
│   ├── impressionism/          # 印象派
│   │   ├── monet_water_lilies.jpg
│   │   ├── monet_sunrise.jpg
│   │   └── renoir_dance.jpg
│   ├── post_impressionism/     # 后印象派
│   │   ├── vangogh_starry_night.jpg
│   │   ├── vangogh_sunflowers.jpg
│   │   └── cezanne_mountain.jpg
│   ├── cubism/                 # 立体派
│   │   ├── picasso_weeping_woman.jpg
│   │   └── braque_violin.jpg
│   ├── ukiyo_e/                # 浮世绘
│   │   ├── hokusai_great_wave.jpg
│   │   └── hiroshige_bridge.jpg
│   └── expressionism/          # 表现主义
│       ├── munch_scream.jpg
│       └── kandinsky_composition.jpg
└── outputs/
    ├── 2024_01_15/             # 按日期组织
    │   ├── portrait_monet_*.jpg
    │   └── landscape_vangogh_*.jpg
    └── experiments/            # 实验结果
        └── comparison_*.jpg
```

## 🎨 风格图片收藏建议

### 必备经典作品（10张起步）

1. **印象派** (2张)
   - 莫奈《睡莲》 - 柔和的色彩和光影
   - 雷诺阿《煎饼磨坊的舞会》 - 温暖的人物场景

2. **后印象派** (3张)
   - 梵高《星空》 - 经典的旋转笔触
   - 梵高《向日葵》 - 明亮的黄色主题
   - 塞尚《圣维克多山》 - 几何化的风景

3. **立体派** (2张)
   - 毕加索《哭泣的女人》 - 独特的解构风格
   - 毕加索《亚威农的少女》 - 多角度的人物

4. **浮世绘** (1张)
   - 葛饰北斋《神奈川冲浪里》 - 日本传统艺术

5. **表现主义** (1张)
   - 蒙克《呐喊》 - 强烈的情感表达

6. **其他** (1张)
   - 康定斯基的抽象作品 - 纯色彩和形状

### 进阶收藏（50+张）

按风格类别扩充：

- **印象派**: 莫奈、雷诺阿、德加、毕沙罗等
- **后印象派**: 梵高、塞尚、高更、修拉等
- **立体派**: 毕加索、布拉克、格里斯等
- **野兽派**: 马蒂斯、德兰等
- **表现主义**: 蒙克、康定斯基、克利等
- **浮世绘**: 葛饰北斋、歌川广重等
- **文艺复兴**: 达芬奇、米开朗基罗、拉斐尔等
- **现代艺术**: 蒙德里安、波洛克、罗斯科等

## 📷 内容图片准备建议

### 最佳实践

1. **分辨率**: 至少 1024x1024 像素
2. **格式**: JPG（压缩率80-95%）
3. **光线**: 光线充足，避免过暗或过曝
4. **构图**: 主体明确，背景不要太复杂
5. **清晰度**: 避免模糊或运动模糊

### 不同类型照片的处理建议

| 照片类型 | 推荐风格 | 风格强度 | 迭代步数 |
|---------|---------|---------|---------|
| 人像照 | 印象派、立体派 | 3-5 | 200-250 |
| 风景照 | 印象派、后印象派 | 5-7 | 250-300 |
| 建筑照 | 立体派、现代艺术 | 5-8 | 250-300 |
| 动物照 | 表现主义、野兽派 | 4-6 | 200-250 |
| 静物照 | 任意风格 | 6-9 | 200-300 |

## 🔧 文件命名规范

### 风格图片命名

格式: `艺术家_作品名_年份.jpg`

示例：
```
monet_water_lilies_1906.jpg
vangogh_starry_night_1889.jpg
picasso_weeping_woman_1937.jpg
hokusai_great_wave_1833.jpg
```

或简化版: `艺术家_作品名.jpg`

```
monet_water_lilies.jpg
vangogh_starry_night.jpg
picasso_weeping_woman.jpg
hokusai_great_wave.jpg
```

### 内容图片命名

格式: `类型_描述_日期.jpg`

示例：
```
portrait_girl_20240115.jpg
landscape_mountain_20240115.jpg
city_tokyo_night_20240115.jpg
animal_cat_20240115.jpg
```

或简化版: `类型_描述.jpg`

```
portrait_girl.jpg
landscape_mountain.jpg
city_tokyo_night.jpg
animal_cat.jpg
```

## 📦 批量添加图片

### 使用脚本批量添加

创建一个批量添加脚本 `add_batch.sh`:

```bash
#!/bin/bash

# 批量添加风格图片
for file in ~/Downloads/artworks/*.jpg; do
    python manage_styles.py add "$file"
done

# 批量添加内容图片
for file in ~/Pictures/photos/*.jpg; do
    python manage_styles.py add-content "$file"
done
```

使用：
```bash
chmod +x add_batch.sh
./add_batch.sh
```

### 直接复制文件

也可以直接将图片复制到对应文件夹：

```bash
# 复制风格图片
cp ~/Downloads/artworks/*.jpg data/style/

# 复制内容图片
cp ~/Pictures/photos/*.jpg data/content/
```

## 💾 备份建议

### 重要文件备份

```bash
# 备份整个data目录
tar -czf data_backup_$(date +%Y%m%d).tar.gz data/

# 只备份图片库（不包括outputs）
tar -czf images_backup_$(date +%Y%m%d).tar.gz data/content/ data/style/
```

### 定期清理

```bash
# 清理outputs中的旧文件（保留最近30天）
find data/outputs -name "*.jpg" -mtime +30 -delete

# 查看各文件夹大小
du -sh data/*
```

## 🌐 资源下载脚本示例

创建 `download_artworks.py` 自动下载艺术作品：

```python
"""
示例：从WikiArt下载艺术作品（需要requests和beautifulsoup4）
注意：请遵守网站的使用条款和版权规定
"""

import requests
from pathlib import Path

# 这只是示例，实际使用时需要检查网站的robots.txt和使用条款
# 建议手动下载或使用官方API

def download_artwork(url, filename):
    """下载艺术作品"""
    try:
        response = requests.get(url)
        if response.status_code == 200:
            filepath = Path('data/style') / filename
            filepath.write_bytes(response.content)
            print(f"✅ 下载成功: {filename}")
        else:
            print(f"❌ 下载失败: {url}")
    except Exception as e:
        print(f"❌ 错误: {e}")

# 示例使用
# download_artwork('https://example.com/artwork.jpg', 'monet_water_lilies.jpg')
```

## 📊 统计信息

查看图片库统计信息：

```bash
# 统计图片数量
echo "风格图片数: $(ls -1 data/style/*.jpg 2>/dev/null | wc -l)"
echo "内容图片数: $(ls -1 data/content/*.jpg 2>/dev/null | wc -l)"
echo "输出结果数: $(ls -1 data/outputs/*.jpg 2>/dev/null | wc -l)"

# 统计总大小
du -sh data/style
du -sh data/content
du -sh data/outputs
```

---

**有序的图片库管理让创作更高效！** 📁✨
