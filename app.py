import os
import json
import time
import pandas as pd
from pathlib import Path

import gradio as gr
import torch
from PIL import Image

from neural_style_transfer import NeuralStyleTransfer
from dataset import StyleImageLoader
from style_recommendation import StyleRecommendationSystem
from video_style_transfer import VideoStyleTransfer


class StyleTransferApp:
    """风格迁移应用类"""

    def __init__(self):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.nst = None
        
        # 初始化风格推荐系统
        self.style_recommender = StyleRecommendationSystem(device=self.device)
        
        # 初始化视频风格迁移
        self.video_transfer = VideoStyleTransfer(device=self.device)

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
    
    def get_local_videos(self):
        """获取本地内容库中的所有视频文件"""
        videos = []
        if os.path.exists(self.local_content_dir):
            for file in os.listdir(self.local_content_dir):
                if file.lower().endswith(('.mp4', '.avi', '.mov', '.mkv', '.flv', '.wmv', '.webm')):
                    videos.append(file)
        return sorted(videos)
    
    def load_local_video(self, video_filename):
        """从本地内容库加载视频"""
        try:
            if not video_filename:
                return None, "⚠️ 请选择一个视频文件"
            
            video_path = os.path.join(self.local_content_dir, video_filename)
            if not os.path.exists(video_path):
                return None, f"❌ 文件不存在: {video_filename}"
            
            info = f"📁 本地视频库: {video_filename}\n路径: {video_path}"
            
            return video_path, info
        except Exception as e:
            return None, f"❌ 加载失败: {str(e)}"
    
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

    def recommend_styles_for_image(self, content_image, top_k=5):
        """为内容图像推荐风格"""
        try:
            if content_image is None:
                return None, gr.update(), content_image, "❌ 请上传内容图片！"
            
            # 获取推荐
            recommendations = self.style_recommender.recommend_styles(content_image, top_k=top_k)
            
            # 提取色彩特征
            color_features = self.style_recommender.extract_color_features(content_image)
            
            # 构建推荐文本
            result_text = "🎨 **风格推荐结果**\n\n"
            result_text += "**图像色彩分析:**\n"
            result_text += f"- 平均色调: {color_features['avg_hue']:.3f}\n"
            result_text += f"- 平均饱和度: {color_features['avg_saturation']:.3f}\n"
            result_text += f"- 平均亮度: {color_features['avg_brightness']:.3f}\n"
            result_text += f"- 颜色多样性: {color_features['color_diversity']:.3f}\n\n"
            result_text += "---\n\n"
            result_text += f"**Top {len(recommendations)} 推荐风格:**\n\n"
            
            for i, rec in enumerate(recommendations, 1):
                result_text += f"**{i}. {rec['style_cn']} ({rec['style']})**\n"
                result_text += f"   匹配度: {rec['score']:.1%}\n"
                result_text += f"   {rec['description']}\n\n"
            
            # 返回推荐列表（用于下拉选择）
            style_names = [f"{rec['style_cn']} ({rec['style']})" for rec in recommendations]
            
            # 返回: 推荐文本, 下拉选择更新, 内容图片(传递到风格迁移), 提示信息
            return result_text, gr.update(choices=style_names, value=style_names[0] if style_names else None), content_image, "✅ 推荐完成！可以选择推荐的风格，然后点击下方按钮跳转到风格迁移"
            
        except Exception as e:
            return f"❌ 推荐失败: {str(e)}", gr.update(), None, f"❌ 推荐失败: {str(e)}"
    
    def apply_recommended_style(self, content_image, selected_style_display):
        """应用推荐的风格到风格迁移标签页"""
        try:
            if content_image is None:
                return None, None, "⚠️ 没有内容图片"
            
            if not selected_style_display:
                return content_image, None, "⚠️ 请先选择一个推荐的风格"
            
            # 加载选中的风格图片
            style_img, info = self.load_wikiart_style(selected_style_display)
            
            if style_img is None:
                return content_image, None, f"❌ 加载风格失败: {info}"
            
            return content_image, style_img, f"✅ 已加载推荐风格！请切换到 '📤 风格迁移' 标签页开始处理。\n\n{info}"
            
        except Exception as e:
            return content_image, None, f"❌ 应用风格失败: {str(e)}"

    def process_video_style_transfer(
        self,
        video_file,
        style_image,
        num_steps,
        style_strength,
        image_size,
        max_frames,
        use_consistency,
        progress=gr.Progress()
    ):
        """处理视频风格迁移"""
        try:
            if video_file is None:
                return None, "❌ 请上传视频文件！"
            if style_image is None:
                return None, "❌ 请选择或上传风格图片！"
            
            progress(0, desc="准备处理视频...")
            
            # 保存临时文件
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            video_path = video_file  # Gradio已经提供了临时文件路径
            style_path = os.path.join(self.outputs_dir, f'temp_style_{timestamp}.jpg')
            
            # 输出路径使用 .mp4 扩展名（H.264编码）
            output_filename = f'styled_video_{timestamp}.mp4'
            output_path = os.path.join(self.outputs_dir, output_filename)
            
            style_image.save(style_path)
            
            # 计算风格权重
            style_weight = 10 ** (5 + style_strength * 0.2)
            
            # 进度回调
            def update_progress(prog, desc):
                progress(prog, desc=desc)
            
            start_time = time.time()
            
            # 处理视频
            result = self.video_transfer.process_video(
                video_path=video_path,
                style_image_path=style_path,
                output_path=output_path,
                num_steps=num_steps,
                style_weight=style_weight,
                image_size=image_size,
                max_frames=max_frames if max_frames > 0 else None,
                use_consistency=use_consistency,
                progress_callback=update_progress
            )
            
            elapsed_time = time.time() - start_time
            
            # 验证输出文件
            if not os.path.exists(output_path) or os.path.getsize(output_path) == 0:
                error_msg = f"❌ 视频文件生成失败！\n\n请检查:\n1. 是否有足够的磁盘空间\n2. 系统是否安装了必要的视频编解码器\n3. 尝试使用较小的参数设置"
                return None, error_msg
            
            # 创建预览 GIF（用于在浏览器中快速查看）
            gif_path = os.path.splitext(output_path)[0] + '_preview.gif'
            gif_created = False
            try:
                progress(0.95, desc="生成预览...")
                gif_created = self.video_transfer.create_preview_gif(
                    output_path, gif_path, num_frames=15, fps=5
                )
            except Exception as e:
                print(f"⚠️ 预览 GIF 生成失败: {str(e)}")
            
            # 清理临时帧文件
            progress(0.98, desc="清理临时文件...")
            try:
                # 清理工作目录（包含所有帧文件）
                work_dir_name = f'work_{Path(output_path).stem}'
                work_dir = os.path.join(self.outputs_dir, work_dir_name)
                
                if os.path.exists(work_dir):
                    import shutil
                    try:
                        shutil.rmtree(work_dir)
                        print(f"✅ 已删除工作目录: {work_dir}")
                    except Exception as e:
                        print(f"⚠️ 删除工作目录失败: {str(e)}")
                
                # 清理其他临时文件
                import glob
                
                # 删除临时风格文件
                temp_style_pattern = os.path.join(self.outputs_dir, 'temp_style_*.jpg')
                for temp_file in glob.glob(temp_style_pattern):
                    try:
                        os.remove(temp_file)
                        print(f"已删除临时风格文件: {temp_file}")
                    except Exception as e:
                        print(f"⚠️ 删除文件失败 {temp_file}: {str(e)}")
                
                # 删除检查点文件（如果还有残留）
                checkpoint_pattern = os.path.join(self.outputs_dir, '*_checkpoint_*.mp4')
                for checkpoint_file in glob.glob(checkpoint_pattern):
                    try:
                        os.remove(checkpoint_file)
                        print(f"已删除检查点文件: {checkpoint_file}")
                    except Exception as e:
                        print(f"⚠️ 删除检查点文件失败 {checkpoint_file}: {str(e)}")
                        
            except Exception as e:
                print(f"⚠️ 清理临时文件时出错: {str(e)}")
            
            # 构建结果信息
            file_size_mb = os.path.getsize(output_path) / (1024 * 1024)
            info_text = f"""
✅ 视频风格迁移完成！

📊 处理信息:
- 设备: {self.device.upper()}
- 总帧数: {result['total_frames']}
- 帧率: {result['fps']} FPS
- 分辨率: {result['resolution']}
- 文件大小: {file_size_mb:.2f} MB
- 迭代步数: {num_steps}
- 风格强度: {style_strength}/10
- 风格权重: {style_weight:.2e}
- 帧间一致性: {'启用' if use_consistency else '禁用'}
- 总耗时: {elapsed_time/60:.1f} 分钟
- 每帧耗时: {result['time_per_frame']:.1f} 秒

💾 文件保存位置:
- 视频: {output_path}
{'- 预览GIF: ' + gif_path if gif_created else ''}

💡 提示: 如果视频无法在浏览器中播放，请点击下载按钮保存到本地使用视频播放器观看
"""
            
            progress(1.0, desc="完成！")
            
            # 返回视频路径（Gradio会自动处理）
            return output_path, info_text
            
        except Exception as e:
            import traceback
            error_details = traceback.format_exc()
            error_msg = f"""❌ 处理失败: {str(e)}

详细错误信息:
{error_details}

请检查:
1. 视频格式是否正确（建议使用 .mp4）
2. 内存是否充足
3. 参数设置是否合理
4. 系统是否安装了视频编解码器（macOS: brew install ffmpeg）
"""
            return None, error_msg


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
    /* 隐藏 Gradio 底部的 "Built with Gradio" 和其他链接 */
    footer {
        display: none !important;
    }
    .footer {
        display: none !important;
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
                    
                    gr.Markdown("**或者直接上传/拍摄内容图片：**")
                    content_input = gr.Image(label="内容图片（你的照片）", type="pil", height=300, sources=["upload", "webcam"])

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
                    style_input = gr.Image(label="风格图片（艺术作品）", type="pil", height=300, sources=["upload", "webcam"])

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

        with gr.Tab("🎯 风格推荐"):
            gr.Markdown("""
            ## 智能风格推荐系统
            
            基于图像的色调、饱和度、亮度等特征，为你的照片推荐最适合的艺术风格！
            """)
            
            with gr.Row():
                with gr.Column():
                    gr.Markdown("### 上传图片")
                    recommend_input = gr.Image(label="内容图片", type="pil", height=400, sources=["upload", "webcam"])
                    
                    with gr.Row():
                        recommend_btn = gr.Button("🔍 分析并推荐风格", variant="primary", size="lg")
                        top_k_slider = gr.Slider(minimum=3, maximum=10, value=5, step=1, 
                                               label="推荐数量", interactive=True)
                
                with gr.Column():
                    gr.Markdown("### 推荐结果")
                    recommendation_output = gr.Textbox(label="风格推荐", lines=20, interactive=False)
                    recommended_styles = gr.Dropdown(label="选择推荐的风格", choices=[], interactive=True)
                    
                    # 添加应用风格按钮
                    apply_style_btn = gr.Button("✨ 应用选中的风格并跳转到风格迁移", variant="primary", size="lg")
                    apply_status = gr.Textbox(label="状态", lines=3, interactive=False)
            
            # 隐藏状态，用于存储推荐时的内容图片
            recommend_content_state = gr.State()
            
            gr.Markdown("""
            ### 使用推荐的风格
            
            1. 上传你的照片
            2. 点击"分析并推荐风格"按钮
            3. 查看推荐结果和匹配度
            4. 从下拉菜单选择推荐的风格
            5. 点击"应用选中的风格"按钮，内容和风格会自动填充到"风格迁移"标签页
            6. 切换到"📤 风格迁移"标签页开始处理
            """)
            
            # 绑定推荐按钮
            recommend_btn.click(
                fn=app.recommend_styles_for_image,
                inputs=[recommend_input, top_k_slider],
                outputs=[recommendation_output, recommended_styles, recommend_content_state, apply_status]
            )
            
            # 绑定应用风格按钮
            apply_style_btn.click(
                fn=app.apply_recommended_style,
                inputs=[recommend_content_state, recommended_styles],
                outputs=[content_input, style_input, apply_status]
            )
        
        with gr.Tab("🎬 视频风格迁移"):
            with gr.Row():
                with gr.Column():
                    gr.Markdown("### 1️⃣ 选择或上传视频")
                    
                    # 本地视频库选择
                    local_videos = app.get_local_videos()
                    if local_videos:
                        with gr.Accordion("📂 从本地视频库选择", open=False):
                            local_video_dropdown = gr.Dropdown(
                                choices=local_videos,
                                label="选择视频文件",
                                interactive=True
                            )
                            load_video_btn = gr.Button("📥 加载选中的视频", variant="secondary")
                            video_load_info = gr.Textbox(label="视频信息", lines=2, interactive=False)
                    else:
                        gr.Markdown("💡 提示：将视频放入 `data/content/` 文件夹可以在此快速选择")
                    
                    gr.Markdown("**或者直接上传/录制视频：**")
                    video_input = gr.Video(label="输入视频", sources=["upload", "webcam"])

                with gr.Column():
                    gr.Markdown("### 2️⃣ 选择或上传风格图片")
                    
                    # 本地风格库选择
                    if local_styles:
                        with gr.Accordion("📂 从本地风格库选择", open=True):
                            video_style_dropdown = gr.Dropdown(
                                choices=local_styles,
                                label="选择风格图片",
                                interactive=True
                            )
                            load_video_style_btn = gr.Button("🎨 加载选中的风格", variant="secondary")
                            video_style_info = gr.Textbox(label="风格信息", lines=2, interactive=False)
                    else:
                        gr.Markdown("💡 提示：将艺术作品放入 `data/style/` 文件夹可以在此快速选择")
                    
                    # WikiArt 数据集选择（视频用）
                    if app.has_wikiart:
                        with gr.Accordion("📚 从 WikiArt 数据集选择", open=False):
                            video_wikiart_style = gr.Dropdown(
                                choices=list(app.wikiart_styles_display.keys()), 
                                value=list(app.wikiart_styles_display.keys())[0], 
                                label="选择艺术风格"
                            )
                            load_video_wikiart_btn = gr.Button("🎨 加载选中的风格", variant="secondary")
                            video_wikiart_info = gr.Textbox(label="风格信息", lines=4, interactive=False)
                    
                    gr.Markdown("**或者直接上传/拍摄风格图片：**")
                    video_style_input = gr.Image(label="风格图片（艺术作品）", type="pil", height=300, sources=["upload", "webcam"])

            gr.Markdown("### 3️⃣ 调整参数")

            with gr.Row():
                video_style_strength = gr.Slider(
                    minimum=1, maximum=10, value=5, step=1,
                    label="风格强度（值越大风格越强）"
                )
                
                video_num_steps = gr.Slider(
                    minimum=50, maximum=300, value=150, step=25,
                    label="迭代步数（每帧）（步数越多质量越好但耗时越长）"
                )
                
                video_image_size = gr.Radio(
                    choices=[256, 512],
                    value=256,
                    label="处理尺寸（较小尺寸处理更快）"
                )
            
            with gr.Row():
                video_max_frames = gr.Slider(
                    minimum=0, maximum=500, value=0, step=10,
                    label="最大处理帧数（0表示处理全部帧）"
                )
                
                video_consistency = gr.Checkbox(
                    value=True,
                    label="启用帧间一致性优化（减少闪烁）"
                )
            
            video_process_btn = gr.Button("🚀 开始视频风格迁移", variant="primary", size="lg")
            
            gr.Markdown("### 4️⃣ 处理结果")
            
            with gr.Row():
                video_output = gr.Video(
                    label="处理后的视频",
                    format="mp4",
                    autoplay=False
                )
                video_info_output = gr.Textbox(label="处理信息", lines=15)
            
            gr.Markdown("""
            ### 💡 提示
            
            - **处理时间**: 取决于视频长度、帧率和参数设置。通常每帧需要5-30秒
            - **断点续传**: 如果处理中断，重新运行将从上次的进度继续
            - **内存使用**: 较大的图像尺寸需要更多GPU/CPU内存
            - **帧间一致性**: 建议开启以避免视频闪烁
            - **测试建议**: 首次使用时建议限制最大帧数进行测试
            
            ### 🎬 视频播放问题解决方案
            
            如果视频在网页中无法播放：
            
            1. **下载到本地**: 点击视频右上角的下载按钮，使用本地播放器观看
            2. **检查浏览器兼容性**: 某些浏览器对 H.264 编码的支持可能不同
            3. **使用其他播放器**: 推荐使用 VLC、QuickTime 等专业播放器
            4. **检查文件大小**: 确保视频文件已完整生成（查看处理信息中的文件大小）
            5. **尝试刷新页面**: 有时需要刷新浏览器页面才能正确加载视频
            
            **常见原因**:
            - 视频编码格式与浏览器不兼容
            - 视频文件较大，加载时间较长
            - 浏览器缓存问题
            
            **推荐做法**: 
            - 处理完成后立即下载视频到本地
            - 文件保存在 `data/outputs/` 目录，可以直接访问
            """)
            
            # 绑定视频本地库加载事件
            if local_videos:
                def handle_video_load(video_filename):
                    """处理视频加载，返回可用于Video组件的路径"""
                    video_path, info = app.load_local_video(video_filename)
                    return video_path, info
                
                load_video_btn.click(
                    fn=handle_video_load,
                    inputs=[local_video_dropdown],
                    outputs=[video_input, video_load_info]
                )
                
                # 下拉选择变化时自动加载
                local_video_dropdown.change(
                    fn=handle_video_load,
                    inputs=[local_video_dropdown],
                    outputs=[video_input, video_load_info]
                )
            
            # 绑定视频风格加载（本地风格库）
            if local_styles:
                load_video_style_btn.click(
                    fn=app.load_local_style,
                    inputs=[video_style_dropdown],
                    outputs=[video_style_input, video_style_info]
                )
                
                # 下拉选择变化时自动加载
                video_style_dropdown.change(
                    fn=app.load_local_style,
                    inputs=[video_style_dropdown],
                    outputs=[video_style_input, video_style_info]
                )
            
            # 绑定视频风格加载（WikiArt数据集）
            if app.has_wikiart:
                load_video_wikiart_btn.click(
                    fn=app.load_wikiart_style,
                    inputs=[video_wikiart_style],
                    outputs=[video_style_input, video_wikiart_info]
                )
                
                # 下拉选择变化时自动加载
                video_wikiart_style.change(
                    fn=app.load_wikiart_style,
                    inputs=[video_wikiart_style],
                    outputs=[video_style_input, video_wikiart_info]
                )
            
            # 绑定视频处理按钮
            video_process_btn.click(
                fn=app.process_video_style_transfer,
                inputs=[
                    video_input,
                    video_style_input,
                    video_num_steps,
                    video_style_strength,
                    video_image_size,
                    video_max_frames,
                    video_consistency
                ],
                outputs=[video_output, video_info_output]
            )

        with gr.Tab("ℹ️ 使用说明"):
            gr.Markdown("""
            ## 使用指南

            ### 📤 风格迁移

            1. **上传内容图片**: 选择你想要转换风格的照片
               - 📁 从本地库选择已保存的图片
               - 📤 上传新图片
               - 📷 使用摄像头实时拍摄
            2. **选择艺术风格**: 
               - 从 WikiArt 数据集的下拉菜单中选择风格，然后点击"加载选中的风格"按钮
               - 或者直接上传自己的风格图片
               - 或者使用摄像头拍摄作为风格图片
            3. **调整参数**:
               - **风格强度**: 1-10，推荐5。值越大，生成图片的艺术风格越强
               - **迭代步数**: 50-500，推荐200。步数越多质量越好但耗时越长
               - **图像尺寸**: 256/512/1024，推荐512。尺寸越大质量越好但需要更多内存
            4. **点击开始**: 等待处理完成
            5. **下载结果**: 右键保存图片

            ### 🎯 风格推荐

            1. **上传照片**: 上传你想要处理的内容图片
               - 📤 上传图片文件
               - 📷 使用摄像头拍摄
            2. **设置推荐数量**: 选择要获得多少个风格推荐（3-10个）
            3. **点击分析**: 系统将分析图片的色彩特征
            4. **查看推荐**: 根据匹配度查看推荐的艺术风格
            5. **应用风格**: 在下拉菜单选择推荐的风格，然后切换到"风格迁移"标签页

            **推荐原理**:
            - 分析图片的色调、饱和度、亮度
            - 计算颜色分布和多样性
            - 匹配最适合的艺术风格特征
            - 提供详细的匹配度说明

            ### 🎬 视频风格迁移

            1. **选择或上传视频**: 
               - 从本地视频库的下拉菜单中选择（视频需放在 `data/content/` 目录）
               - 📤 上传视频文件
               - 📷 使用摄像头实时录制
            2. **选择风格**: 
               - 从本地风格库选择（风格图片放在 `data/style/` 目录）
               - 从 WikiArt 数据集选择
               - 📤 上传风格图片
               - 📷 使用摄像头拍摄作为风格
            3. **调整参数**:
               - **迭代步数**: 每帧的优化步数，推荐150
               - **风格强度**: 1-10，控制风格效果强度
               - **处理尺寸**: 256或512，较小尺寸处理更快
               - **最大帧数**: 限制处理的帧数，0表示处理全部
               - **帧间一致性**: 建议开启以减少闪烁
            4. **开始处理**: 点击按钮开始，可以看到进度
            5. **下载视频**: 处理完成后下载结果

            **视频处理特点**:
            - ✅ **断点续传**: 如果中断，重新运行会继续之前的进度
            - ✅ **帧间一致性**: 使用时间一致性优化，减少闪烁
            - ✅ **进度显示**: 实时显示处理进度和预计剩余时间
            - ✅ **自动保存**: 每10帧自动保存检查点

            ### 命令行使用

            #### 图像风格迁移
            ```bash
            # 基本用法
            python train.py --content photo.jpg --style art.jpg

            # 使用WikiArt数据集
            python train.py --content photo.jpg --style-name Impressionism

            # 高质量输出
            python train.py --content photo.jpg --style art.jpg --steps 500 --size 1024
            ```

            #### 视频风格迁移
            ```bash
            # 基本用法
            python video_style_transfer.py my_video.mp4 starry_night.jpg

            # 快速测试（只处理前50帧）
            python video_style_transfer.py video.mp4 style.jpg -f 50

            # 高质量处理
            python video_style_transfer.py video.mp4 style.jpg -s 300 --size 1024

            # 禁用帧间一致性（更快但可能闪烁）
            python video_style_transfer.py video.mp4 style.jpg --no-consistency
            ```

            ### 参数推荐配置

            #### 图像处理

            | 场景 | 步数 | 尺寸 | 风格权重 | 耗时(GPU) |
            |------|------|------|----------|-----------|
            | 快速测试 | 100 | 256 | 1e6 | ~30秒 |
            | 日常使用 | 200 | 512 | 1e6 | ~1分钟 |
            | 高质量 | 300 | 512 | 5e6 | ~2分钟 |
            | 专业级 | 500 | 1024 | 5e6 | ~5分钟 |

            #### 视频处理

            | 场景 | 步数 | 尺寸 | 帧间一致性 | 最大帧数 | 预估时间(GPU) |
            |------|------|------|-----------|---------|--------------|
            | 快速预览 | 100 | 256 | 禁用 | 30 | ~5分钟 |
            | 测试运行 | 150 | 512 | 启用 | 100 | ~30分钟 |
            | 标准质量 | 150 | 512 | 启用 | 无限制 | ~2-5小时* |
            | 高质量 | 200 | 768 | 启用 | 无限制 | ~5-10小时* |

            *时间取决于视频长度和帧率

            ### 常见问题

            **Q: 显存不足怎么办？**
            - 减小图像尺寸（使用256或512）
            - 使用CPU模式（较慢）

            **Q: 风格太强/太弱？**
            - 调整风格强度滑块（1-10）
            - 命令行中使用 `--style-weight` 参数

            **Q: 视频处理中断了怎么办？**
            - 视频处理支持断点续传
            - 重新运行相同命令会从上次中断处继续
            - 每10帧自动保存检查点

            **Q: 视频输出有闪烁？**
            - 确保启用"帧间一致性"选项（默认启用）
            - 可以尝试增加迭代步数

            ### 技术信息

            - **算法**: 基于 VGG19 的神经风格迁移
            - **框架**: PyTorch
            - **当前设备**: {}
            - **WikiArt 数据集**: {}
            - **新功能**:
              - 智能风格推荐系统（基于ResNet50特征提取）
              - 视频风格迁移（支持帧间一致性优化）

            ### 结果保存

            所有生成的结果都会自动保存在 `data/outputs/` 目录下。

            ### 目录结构

            ```
            project/
            ├── data/
            │   ├── content/          # 放置内容图像和视频
            │   ├── style/            # 放置风格图像
            │   └── outputs/          # 处理结果输出
            ├── train.py              # 图像风格迁移命令行
            ├── video_style_transfer.py  # 视频风格迁移命令行
            ├── app.py               # Web界面
            └── USAGE.md             # 完整使用文档
            ```

            ### 📖 更多信息

            - 完整的命令行参数说明请查看项目目录下的 `USAGE.md` 文件
            - 所有处理结果保存在 `data/outputs/` 目录
            - 支持的图像格式：JPG, PNG, BMP, WEBP
            - 支持的视频格式：MP4, AVI, MOV, MKV, FLV, WMV, WEBM
            """.format("GPU (CUDA)" if torch.cuda.is_available() else "CPU", "已加载" if app.has_wikiart else "未加载"))

        process_btn.click(
            fn=app.process_style_transfer,
            inputs=[content_input, style_input, style_strength, num_steps, image_size],
            outputs=[output_image, info_output],
        )

        gr.Markdown("""
        ---

        💡 **提示**: 
        - 以上为简要使用说明，更详细的功能介绍请查看 "ℹ️ 使用说明" 标签页
        - 完整的命令行使用文档请查看项目目录下的 `USAGE.md` 文件
        - 所有结果保存在 `data/outputs/` 目录
        """)

    return interface


if __name__ == "__main__":
    interface = create_interface()
    interface.launch(
        share=False, 
        server_name="0.0.0.0", 
        server_port=7861, 
        show_error=True,
        show_api=False  
    )
