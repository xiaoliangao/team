import os
import json
import time
import pandas as pd

import gradio as gr
import torch
from PIL import Image

from neural_style_transfer import NeuralStyleTransfer
from dataset import StyleImageLoader


class StyleTransferApp:
    """风格迁移应用类"""

    def __init__(self):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.nst = None

        # 以当前文件为基准的项目根目录（确保从任意 CWD 导入时路径正确）
        self.base_dir = os.path.dirname(os.path.abspath(__file__))

        # 创建输出目录（绝对路径）
        self.outputs_dir = os.path.join(self.base_dir, 'data', 'outputs')
        os.makedirs(self.outputs_dir, exist_ok=True)

        # 本地风格图片库路径
        self.local_style_dir = os.path.join(self.base_dir, 'data', 'style')
        os.makedirs(self.local_style_dir, exist_ok=True)
        
        # 本地内容图片库路径
        self.local_content_dir = os.path.join(self.base_dir, 'data', 'content')
        os.makedirs(self.local_content_dir, exist_ok=True)

        # WikiArt 数据集路径（绝对路径）
        self.wikiart_path = os.path.join(self.base_dir, 'wikiart', 'data')
        self.has_wikiart = os.path.exists(self.wikiart_path)
        # 从元数据中加载完整风格列表
        self.wikiart_styles = self._load_wikiart_styles_from_metadata()
        # 构建风格名 -> 标签索引的映射（parquet 中可能存储为数值标签）
        self.wikiart_style_to_label = {}
        for i, name in enumerate(self.wikiart_styles):
            if name == '随机':
                continue
            # parquet 中的标签通常是从 0 开始对应 metadata 列表中的顺序
            self.wikiart_style_to_label[name] = i - 1
        
        # 艺术风格中英文对照
        self.style_translations = {
            'Abstract_Expressionism': '抽象表现主义',
            'Action_painting': '行动绘画',
            'Analytical_Cubism': '分析立体主义',
            'Art_Nouveau': '新艺术运动',
            'Baroque': '巴洛克',
            'Color_Field_Painting': '色域绘画',
            'Contemporary_Realism': '当代写实主义',
            'Cubism': '立体主义',
            'Early_Renaissance': '早期文艺复兴',
            'Expressionism': '表现主义',
            'Fauvism': '野兽派',
            'High_Renaissance': '盛期文艺复兴',
            'Impressionism': '印象派',
            'Mannerism_Late_Renaissance': '风格主义',
            'Minimalism': '极简主义',
            'Naive_Art_Primitivism': '素朴艺术',
            'New_Realism': '新写实主义',
            'Northern_Renaissance': '北方文艺复兴',
            'Pointillism': '点彩派',
            'Pop_Art': '波普艺术',
            'Post_Impressionism': '后印象派',
            'Realism': '写实主义',
            'Rococo': '洛可可',
            'Romanticism': '浪漫主义',
            'Symbolism': '象征主义',
            'Synthetic_Cubism': '综合立体主义',
            'Ukiyo_e': '浮世绘',
            '随机': '随机'
        }
        
        # 创建带翻译的风格选项
        self.wikiart_styles_display = self._create_translated_styles()

    def _create_translated_styles(self):
        """创建带中文翻译的风格选项"""
        translated = {}
        for style in self.wikiart_styles:
            # 将下划线替换为空格以匹配翻译字典
            style_key = style.replace(' ', '_')
            if style_key in self.style_translations:
                display_text = f"{self.style_translations[style_key]} ({style})"
                translated[display_text] = style
            else:
                # 如果没有翻译，直接使用原名
                translated[style] = style
        return translated
    
    def _get_original_style_from_display(self, display_text):
        """从显示文本中提取原始风格名称"""
        for display, original in self.wikiart_styles_display.items():
            if display == display_text:
                return original
        return display_text

    def get_local_styles(self):
        """获取本地风格图片库中的所有图片"""
        styles = []
        if os.path.exists(self.local_style_dir):
            for file in os.listdir(self.local_style_dir):
                if file.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.webp')):
                    styles.append(file)
        return sorted(styles)
    
    def load_local_style(self, style_filename):
        """从本地风格库加载图片"""
        try:
            if not style_filename:
                return None, "⚠️ 请选择一个风格图片"
            
            style_path = os.path.join(self.local_style_dir, style_filename)
            if not os.path.exists(style_path):
                return None, f"❌ 文件不存在: {style_filename}"
            
            style_img = Image.open(style_path).convert('RGB')
            info = f"📁 本地风格库: {style_filename}\n路径: {style_path}"
            
            return style_img, info
        except Exception as e:
            return None, f"❌ 加载失败: {str(e)}"

    def get_local_contents(self):
        """获取本地内容图片库中的所有图片"""
        contents = []
        if os.path.exists(self.local_content_dir):
            for file in os.listdir(self.local_content_dir):
                if file.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.webp')):
                    contents.append(file)
        return sorted(contents)
    
    def load_local_content(self, content_filename):
        """从本地内容库加载图片"""
        try:
            if not content_filename:
                return None, "⚠️ 请选择一个内容图片"
            
            content_path = os.path.join(self.local_content_dir, content_filename)
            if not os.path.exists(content_path):
                return None, f"❌ 文件不存在: {content_filename}"
            
            content_img = Image.open(content_path).convert('RGB')
            info = f"📁 本地内容库: {content_filename}\n路径: {content_path}"
            
            return content_img, info
        except Exception as e:
            return None, f"❌ 加载失败: {str(e)}"


    def process_style_transfer(
        self,
        content_image,
        style_image,
        style_strength,
        num_steps,
        image_size,
        progress=gr.Progress(),
    ):
        """执行风格迁移"""
        try:
            if content_image is None:
                return None, "❌ 请上传内容图片！"
            if style_image is None:
                return None, "❌ 请选择或上传风格图片！"

            progress(0, desc="初始化模型...")

            # 初始化模型
            self.nst = NeuralStyleTransfer(device=self.device)
            self.nst.imsize = image_size

            # 保存临时文件
            content_path = os.path.join(self.outputs_dir, 'temp_content.jpg')
            style_path = os.path.join(self.outputs_dir, 'temp_style.jpg')

            content_image.save(content_path)
            style_image.save(style_path)

            progress(0.1, desc="加载图片...")

            # 加载图片
            content_img = self.nst.load_image(content_path)
            style_img = self.nst.load_image(style_path)

            progress(0.2, desc="开始风格迁移...")

            # 计算风格权重（1-10映射到1e5-1e7）
            style_weight = 10 ** (5 + style_strength * 0.2)

            start_time = time.time()

            # 执行风格迁移
            output = self.nst.run_style_transfer(
                content_img,
                style_img,
                num_steps=num_steps,
                style_weight=style_weight,
                content_weight=1,
            )

            elapsed_time = time.time() - start_time

            progress(0.9, desc="生成结果图片...")

            result_img = self.nst.show_image(output)

            # 保存结果
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            output_filename = f'result_{timestamp}.jpg'
            output_path = os.path.join(self.outputs_dir, output_filename)
            result_img.save(output_path)

            info_text = f"""
✅ 风格迁移完成！

📊 处理信息:
- 设备: {self.device.upper()}
- 图像尺寸: {image_size}x{image_size}
- 迭代步数: {num_steps}
- 风格强度: {style_strength}/10
- 风格权重: {style_weight:.2e}
- 处理时间: {elapsed_time:.2f}秒

💾 结果已保存: {output_path}
"""

            progress(1.0, desc="完成！")

            return result_img, info_text

        except Exception as e:
            error_msg = f"❌ 处理失败: {str(e)}\n\n请检查:\n1. 图片格式是否正确\n2. 内存是否充足\n3. 参数设置是否合理"
            return None, error_msg

    def load_wikiart_style(self, style_name_display):
        """从 WikiArt 加载随机或指定风格的图片"""
        try:
            if not self.has_wikiart:
                return None, "❌ WikiArt 数据集不可用"

            # 从显示文本提取原始风格名称
            style_name = self._get_original_style_from_display(style_name_display)

            parquet_files = [f for f in os.listdir(self.wikiart_path) if f.endswith('.parquet')]
            if not parquet_files:
                return None, "❌ 未找到任何 WikiArt parquet 文件于 wikiart/data/"

            # 随机选择
            if style_name == "随机":
                parquet_path = os.path.join(self.wikiart_path, parquet_files[0])
                loader = StyleImageLoader(parquet_path, image_size=512)
                style_tensor, metadata = loader.get_random_image(style=None)
            else:
                # 在本地分片中查找包含该风格的分片（只读取 style 列来加速）
                found = False
                selected_parquet = None
                matched_style_value = None

                def _normalize(s):
                    return ''.join(ch.lower() for ch in str(s) if ch.isalnum())

                target_norm = _normalize(style_name)

                # 计算可能的数值标签（如果 metadata 可用）
                desired_label = None
                if style_name in self.wikiart_style_to_label:
                    desired_label = self.wikiart_style_to_label[style_name]

                for pf in parquet_files:
                    path = os.path.join(self.wikiart_path, pf)
                    try:
                        df = pd.read_parquet(path, columns=['style'])
                        styles = list(df['style'].dropna().unique())

                        # 先检查数值标签是否存在
                        if desired_label is not None and any((s == desired_label) or (str(s) == str(desired_label)) for s in styles):
                            selected_parquet = path
                            matched_style_value = desired_label
                            found = True
                            break

                        # 尝试精确字符串或归一化字符串匹配
                        for s in styles:
                            try:
                                if s == style_name or _normalize(s) == target_norm:
                                    selected_parquet = path
                                    matched_style_value = s
                                    found = True
                                    break
                            except Exception:
                                continue
                        if found:
                            break
                    except Exception:
                        continue

                if not found:
                    msg = f"⚠️ 未在本地数据集中找到风格 '{style_name}'。"
                    return None, msg

                loader = StyleImageLoader(selected_parquet, image_size=512)
                # 使用匹配到的实际值（数值或字符串）进行加载
                style_to_use = matched_style_value if matched_style_value is not None else style_name
                style_tensor, metadata = loader.get_random_image(style=style_to_use)

            # 转换为 PIL
            import torchvision.transforms as transforms
            to_pil = transforms.ToPILImage()
            style_img = to_pil(style_tensor)

            info = f"📚 WikiArt 风格:\n"
            for key, value in metadata.items():
                info += f"- {key}: {value}\n"

            return style_img, info

        except Exception as e:
            return None, f"❌ 加载失败: {str(e)}"

    def _load_wikiart_styles_from_metadata(self):
        """从 wikiart/dataset_infos.json 中读取风格列表"""
        meta_path = os.path.join(self.base_dir, 'wikiart', 'dataset_infos.json')
        try:
            if os.path.exists(meta_path):
                with open(meta_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)

                for k, v in data.items():
                    if isinstance(v, dict) and 'features' in v and 'style' in v['features']:
                        style_info = v['features']['style']
                        if 'names' in style_info:
                            return ['随机'] + style_info['names']

                if 'style' in data:
                    style_info = data['style']
                    if isinstance(style_info, dict) and 'names' in style_info:
                        return ['随机'] + style_info['names']
        except Exception:
            pass

        # 兜底默认
        return ['随机', 'Impressionism', 'Post_Impressionism', 'Cubism', 'Expressionism', 'Ukiyo_e', 'Realism']


def create_interface():
    """创建 Gradio 界面"""

    app = StyleTransferApp()

    css = """
    .gradio-container {
        font-family: 'Arial', sans-serif;
    }
    .result-image {
        border: 2px solid #4CAF50;
        border-radius: 10px;
    }
    """

    with gr.Blocks(css=css, title="神经风格迁移") as interface:

        gr.Markdown("""
        # 🎨 神经风格迁移 (Neural Style Transfer)

        将艺术作品的风格应用到你的照片上！上传图片即可开始。
        """)

        with gr.Tab("📤 风格迁移"):
            with gr.Row():
                with gr.Column():
                    gr.Markdown("### 1️⃣ 选择或上传内容图片")
                    
                    # 本地内容库选择
                    local_contents = app.get_local_contents()
                    if local_contents:
                        with gr.Accordion("📂 从本地内容库选择", open=False):
                            local_content_dropdown = gr.Dropdown(
                                choices=local_contents,
                                label="选择内容图片",
                                interactive=True
                            )
                            load_content_btn = gr.Button("📥 加载选中的内容图片", variant="secondary")
                            content_info = gr.Textbox(label="图片信息", lines=2, interactive=False)
                    else:
                        gr.Markdown("💡 提示：将照片放入 `data/content/` 文件夹可以在此快速选择")
                    
                    gr.Markdown("**或者直接上传内容图片：**")
                    content_input = gr.Image(label="内容图片（你的照片）", type="pil", height=300, sources=["upload"])

                with gr.Column():
                    gr.Markdown("### 2️⃣ 选择或上传风格图片")
                    
                    # 本地风格库选择
                    local_styles = app.get_local_styles()
                    if local_styles:
                        with gr.Accordion("📂 从本地风格库选择", open=True):
                            local_style_dropdown = gr.Dropdown(
                                choices=local_styles,
                                label="选择风格图片",
                                interactive=True
                            )
                            load_local_style_btn = gr.Button("🎨 加载选中的风格", variant="secondary")
                            local_style_info = gr.Textbox(label="风格信息", lines=2, interactive=False)
                    else:
                        gr.Markdown("💡 提示：将艺术作品放入 `data/style/` 文件夹可以在此快速选择")
                    
                    # WikiArt 数据集选择
                    if app.has_wikiart:
                        with gr.Accordion("📚 从 WikiArt 数据集选择", open=False):
                            wikiart_style = gr.Dropdown(
                                choices=list(app.wikiart_styles_display.keys()), 
                                value=list(app.wikiart_styles_display.keys())[0], 
                                label="选择艺术风格"
                            )
                            load_wikiart_btn = gr.Button("🎨 加载选中的风格", variant="secondary")
                            wikiart_info = gr.Textbox(label="风格信息", lines=4, interactive=False)
                    
                    gr.Markdown("**或者直接上传风格图片：**")
                    style_input = gr.Image(label="风格图片（艺术作品）", type="pil", height=300, sources=["upload"])

            gr.Markdown("### 3️⃣ 调整参数")

            with gr.Row():
                style_strength = gr.Slider(minimum=1, maximum=10, value=5, step=1, label="风格强度（值越大风格越强）")

                num_steps = gr.Slider(minimum=50, maximum=500, value=200, step=50, label="迭代步数（步数越多质量越好但耗时越长）")

                image_size = gr.Radio(choices=[256, 512, 1024], value=512, label="图像尺寸")

            process_btn = gr.Button("🚀 开始风格迁移", variant="primary", size="lg")

            gr.Markdown("### 4️⃣ 结果")

            with gr.Row():
                output_image = gr.Image(label="生成结果", type="pil", elem_classes=["result-image"])
                info_output = gr.Textbox(label="处理信息", lines=15)
        
        # 绑定本地风格库加载事件
        if local_styles:
            load_local_style_btn.click(
                fn=app.load_local_style,
                inputs=[local_style_dropdown],
                outputs=[style_input, local_style_info]
            )
            
            # 下拉选择变化时自动加载
            local_style_dropdown.change(
                fn=app.load_local_style,
                inputs=[local_style_dropdown],
                outputs=[style_input, local_style_info]
            )
        
        # 绑定本地内容库加载事件
        if local_contents:
            load_content_btn.click(
                fn=app.load_local_content,
                inputs=[local_content_dropdown],
                outputs=[content_input, content_info]
            )
            
            # 下拉选择变化时自动加载
            local_content_dropdown.change(
                fn=app.load_local_content,
                inputs=[local_content_dropdown],
                outputs=[content_input, content_info]
            )
        
        # 绑定 WikiArt 加载按钮事件，并在下拉值变化时自动加载（用户只选择即可自动填充）
        if app.has_wikiart:
            load_wikiart_btn.click(
                fn=app.load_wikiart_style,
                inputs=[wikiart_style],
                outputs=[style_input, wikiart_info]
            )

            # 当用户在下拉中选择风格时自动加载对应风格图片（无需额外点击）
            wikiart_style.change(
                fn=app.load_wikiart_style,
                inputs=[wikiart_style],
                outputs=[style_input, wikiart_info]
            )

        with gr.Tab("ℹ️ 使用说明"):
            gr.Markdown("""
            ## 使用指南

            ### 基本步骤

            1. **上传内容图片**: 选择你想要转换风格的照片
            2. **选择艺术风格**: 
               - 从 WikiArt 数据集的下拉菜单中选择风格，然后点击"加载选中的风格"按钮
               - 或者直接上传自己的风格图片
            3. **调整参数**:
               - **风格强度**: 1-10，推荐5。值越大，生成图片的艺术风格越强
               - **迭代步数**: 50-500，推荐200。步数越多质量越好但耗时越长
               - **图像尺寸**: 256/512/1024，推荐512。尺寸越大质量越好但需要更多内存
            4. **点击开始**: 等待处理完成
            5. **下载结果**: 右键保存图片

            ### 技术信息

            - **算法**: 基于 VGG19 的神经风格迁移
            - **框架**: PyTorch
            - **当前设备**: {}
            - **WikiArt 数据集**: {}

            ### 结果保存

            所有生成的结果都会自动保存在 `data/outputs/` 目录下。
            """.format("GPU (CUDA)" if torch.cuda.is_available() else "CPU", "已加载" if app.has_wikiart else "未加载"))

        process_btn.click(
            fn=app.process_style_transfer,
            inputs=[content_input, style_input, style_strength, num_steps, image_size],
            outputs=[output_image, info_output],
        )

        gr.Markdown("""
        ---

        💡 **提示**: 详细使用说明请查看 [USAGE.md](USAGE.md)

        Made with ❤️ using PyTorch and Gradio
        """)

    return interface


if __name__ == "__main__":
    interface = create_interface()
    interface.launch(share=False, server_name="0.0.0.0", server_port=7861, show_error=True)
