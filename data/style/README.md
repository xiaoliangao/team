# 风格图片库 (Style Images)

这个文件夹用于存放你的艺术风格图片。

## 📁 文件夹用途

将你喜欢的艺术作品图片放在这里，可以在风格迁移时使用。

## 🎨 支持的图片格式

- JPG / JPEG
- PNG
- BMP
- WebP

## 📝 使用方法

### 方法1：命令行使用

```bash
# 使用本地风格图片
python train.py --content data/content/photo.jpg --style data/style/art.jpg
```

### 方法2：Web GUI使用

1. 启动 Web 界面：`python app.py`
2. 在界面中从"本地风格库"下拉菜单选择你上传的风格图片
3. 或直接上传新的风格图片

### 方法3：批量处理

```bash
# 批量处理多张图片
python batch_process.py \
    --mode folder \
    --content-dir data/content \
    --style-dir data/style \
    --output batch_results
```

## 💡 建议

### 风格图片的选择

- **艺术作品**: 油画、水彩画、素描等艺术作品效果最佳
- **高质量**: 建议使用高分辨率图片（至少 512x512）
- **清晰度**: 避免模糊或过度压缩的图片
- **多样性**: 收集不同风格的作品，如印象派、立体派、浮世绘等

### 推荐的艺术风格

1. **印象派**: 莫奈的《睡莲》、雷诺阿的作品
2. **后印象派**: 梵高的《星空》、塞尚的静物
3. **立体派**: 毕加索的作品
4. **浮世绘**: 葛饰北斋的《神奈川冲浪里》
5. **抽象派**: 康定斯基、蒙德里安的作品
6. **现代艺术**: 各种现代艺术作品

### 文件命名建议

使用有意义的文件名方便识别：

```
data/style/
├── monet_water_lilies.jpg        # 莫奈-睡莲
├── vangogh_starry_night.jpg      # 梵高-星空
├── hokusai_great_wave.jpg        # 北斋-神奈川冲浪里
├── picasso_weeping_woman.jpg     # 毕加索-哭泣的女人
└── kandinsky_composition.jpg     # 康定斯基-构成
```

## 🔧 管理工具

使用风格图片管理脚本：

```bash
# 列出所有风格图片
python manage_styles.py list

# 添加风格图片（会复制到此文件夹）
python manage_styles.py add path/to/artwork.jpg

# 预览风格图片
python manage_styles.py preview monet_water_lilies.jpg

# 删除风格图片
python manage_styles.py remove old_style.jpg
```

## 📊 示例结构

```
data/style/
├── README.md                    # 本说明文件
├── impressionism/               # 印象派
│   ├── monet_01.jpg
│   └── renoir_01.jpg
├── post_impressionism/          # 后印象派
│   ├── vangogh_01.jpg
│   └── cezanne_01.jpg
├── cubism/                      # 立体派
│   └── picasso_01.jpg
└── ukiyo_e/                     # 浮世绘
    └── hokusai_01.jpg
```

## ⚠️ 注意事项

1. **版权**: 确保你有权使用这些艺术作品图片
2. **文件大小**: 单个文件建议不超过 10MB
3. **备份**: 重要的艺术图片请做好备份
4. **清理**: 定期清理不再使用的图片以节省空间

## 🌐 资源推荐

### 免费艺术资源网站

- [WikiArt](https://www.wikiart.org/) - 大量艺术作品
- [The Met Collection](https://www.metmuseum.org/art/collection) - 大都会博物馆
- [Google Arts & Culture](https://artsandculture.google.com/) - 谷歌艺术与文化
- [Rijksmuseum](https://www.rijksmuseum.nl/) - 荷兰国家博物馆

---

**开始收集你喜欢的艺术作品，创造独特的风格迁移效果吧！** 🎨
