"""
视频风格迁移 - 支持逐帧处理、帧间一致性优化和断点续传
"""

import os
import json
import time
from pathlib import Path
from typing import Optional, Callable, Dict, List
import cv2
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
from tqdm import tqdm

from neural_style_transfer import NeuralStyleTransfer


class VideoStyleTransfer:
    """视频风格迁移类"""
    
    def __init__(self, device='cuda' if torch.cuda.is_available() else 'cpu'):
        """
        初始化视频风格迁移
        
        Args:
            device: 计算设备
        """
        self.device = device
        self.nst = NeuralStyleTransfer(device=device)
        
    def extract_frames(self, video_path: str, output_dir: str, 
                       max_frames: Optional[int] = None) -> Dict:
        """
        从视频中提取帧
        
        Args:
            video_path: 视频文件路径
            output_dir: 帧输出目录
            max_frames: 最大提取帧数（None表示提取所有帧）
            
        Returns:
            视频信息字典
        """
        # 创建输出目录
        frames_dir = os.path.join(output_dir, 'frames')
        os.makedirs(frames_dir, exist_ok=True)
        
        # 打开视频
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            raise ValueError(f"无法打开视频文件: {video_path}")
        
        # 获取视频属性
        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps == 0 or fps is None:
            fps = 30.0  # 默认帧率
        fps = float(fps)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        # 限制帧数
        frames_to_extract = min(total_frames, max_frames) if max_frames else total_frames
        
        print(f"视频信息:")
        print(f"  分辨率: {width}x{height}")
        print(f"  帧率: {fps} FPS")
        print(f"  总帧数: {total_frames}")
        print(f"  提取帧数: {frames_to_extract}")
        
        # 提取帧
        frame_paths = []
        for i in tqdm(range(frames_to_extract), desc="提取视频帧"):
            ret, frame = cap.read()
            if not ret:
                break
            
            # 保存帧
            frame_path = os.path.join(frames_dir, f'frame_{i:06d}.jpg')
            cv2.imwrite(frame_path, frame)
            frame_paths.append(frame_path)
        
        cap.release()
        
        # 保存视频信息
        video_info = {
            'fps': fps,
            'width': width,
            'height': height,
            'total_frames': total_frames,
            'extracted_frames': len(frame_paths),
            'frame_paths': frame_paths
        }
        
        info_path = os.path.join(output_dir, 'video_info.json')
        with open(info_path, 'w') as f:
            json.dump(video_info, f, indent=2)
        
        return video_info
    
    def process_frame_with_consistency(self, 
                                      current_frame: torch.Tensor,
                                      style_img: torch.Tensor,
                                      prev_output: Optional[torch.Tensor] = None,
                                      num_steps: int = 200,
                                      style_weight: float = 1e6,
                                      content_weight: float = 1.0,
                                      temporal_weight: float = 1e4) -> torch.Tensor:
        """
        处理单帧，带帧间一致性优化
        
        Args:
            current_frame: 当前帧
            style_img: 风格图像
            prev_output: 上一帧的输出（用于保持一致性）
            num_steps: 优化步数
            style_weight: 风格权重
            content_weight: 内容权重
            temporal_weight: 时间一致性权重
            
        Returns:
            处理后的帧
        """
        # 如果有前一帧，使用它作为初始化（提供更好的一致性）
        if prev_output is not None:
            input_img = prev_output.clone()
        else:
            input_img = current_frame.clone()
        
        # 构建模型
        model, style_losses, content_losses = self.nst.get_style_model_and_losses(
            style_img, current_frame)
        
        # 优化器
        optimizer = torch.optim.LBFGS([input_img.requires_grad_()])
        
        run = [0]
        while run[0] <= num_steps:
            def closure():
                with torch.no_grad():
                    input_img.clamp_(0, 1)
                
                optimizer.zero_grad()
                model(input_img)
                
                # 计算损失
                style_score = sum(sl.loss for sl in style_losses) * style_weight
                content_score = sum(cl.loss for cl in content_losses) * content_weight
                
                # 添加时间一致性损失（如果有前一帧）
                temporal_score = 0
                if prev_output is not None:
                    temporal_score = F.mse_loss(input_img, prev_output) * temporal_weight
                
                loss = style_score + content_score + temporal_score
                loss.backward()
                
                run[0] += 1
                return loss
            
            optimizer.step(closure)
        
        with torch.no_grad():
            input_img.clamp_(0, 1)
        
        return input_img
    
    def process_video(self, 
                     video_path: str,
                     style_image_path: str,
                     output_path: str,
                     num_steps: int = 200,
                     style_weight: float = 1e6,
                     content_weight: float = 1.0,
                     temporal_weight: float = 1e4,
                     image_size: int = 512,
                     max_frames: Optional[int] = None,
                     use_consistency: bool = True,
                     progress_callback: Optional[Callable] = None) -> Dict:
        """
        处理整个视频
        
        Args:
            video_path: 输入视频路径
            style_image_path: 风格图像路径
            output_path: 输出视频路径
            num_steps: 每帧的优化步数
            style_weight: 风格权重
            content_weight: 内容权重
            temporal_weight: 时间一致性权重
            image_size: 处理图像大小
            max_frames: 最大处理帧数
            use_consistency: 是否使用帧间一致性优化
            progress_callback: 进度回调函数
            
        Returns:
            处理信息字典
        """
        start_time = time.time()
        
        # 设置NST图像大小
        self.nst.imsize = image_size
        
        # 创建工作目录
        work_dir = os.path.join(os.path.dirname(output_path), 
                               f'work_{Path(output_path).stem}')
        os.makedirs(work_dir, exist_ok=True)
        
        # 检查是否有断点续传数据
        checkpoint_path = os.path.join(work_dir, 'checkpoint.json')
        checkpoint = self._load_checkpoint(checkpoint_path)
        
        # 提取帧（如果尚未提取）
        if checkpoint and 'video_info' in checkpoint:
            print("从断点恢复...")
            video_info = checkpoint['video_info']
            start_frame = checkpoint.get('last_processed_frame', 0) + 1
        else:
            print("提取视频帧...")
            video_info = self.extract_frames(video_path, work_dir, max_frames)
            start_frame = 0
        
        # 加载风格图像
        print("加载风格图像...")
        style_img = self.nst.load_image(style_image_path)
        
        # 创建输出帧目录
        output_frames_dir = os.path.join(work_dir, 'styled_frames')
        os.makedirs(output_frames_dir, exist_ok=True)
        
        # 处理每一帧
        frame_paths = video_info['frame_paths']
        total_frames = len(frame_paths)
        
        prev_output = None
        processed_frames = []
        
        print(f"\n开始处理视频（从第 {start_frame} 帧开始）...")
        
        for i in range(start_frame, total_frames):
            frame_start_time = time.time()
            
            # 加载当前帧
            current_frame = self.nst.load_image(frame_paths[i])
            
            # 处理帧
            if use_consistency and prev_output is not None:
                output = self.process_frame_with_consistency(
                    current_frame, style_img, prev_output,
                    num_steps, style_weight, content_weight, temporal_weight
                )
            else:
                output = self.nst.run_style_transfer(
                    current_frame, style_img, num_steps,
                    style_weight, content_weight
                )
            
            # 保存输出帧
            output_frame_path = os.path.join(output_frames_dir, f'styled_{i:06d}.jpg')
            output_pil = self.nst.show_image(output)
            output_pil.save(output_frame_path, quality=95)
            processed_frames.append(output_frame_path)
            
            # 保存为下一帧的参考
            prev_output = output.detach()
            
            # 计算进度和预估时间
            frame_time = time.time() - frame_start_time
            progress = (i + 1) / total_frames
            remaining_frames = total_frames - (i + 1)
            eta = remaining_frames * frame_time
            
            # 更新进度
            if progress_callback:
                progress_callback(progress, f"处理第 {i+1}/{total_frames} 帧")
            
            print(f"帧 {i+1}/{total_frames} 完成 "
                  f"({progress*100:.1f}%) - "
                  f"耗时: {frame_time:.1f}s - "
                  f"预计剩余: {eta/60:.1f}分钟")
            
            # 保存检查点
            if (i + 1) % 10 == 0:  # 每10帧保存一次
                self._save_checkpoint(checkpoint_path, {
                    'video_info': video_info,
                    'last_processed_frame': i,
                    'processed_frames': processed_frames,
                    'params': {
                        'num_steps': num_steps,
                        'style_weight': style_weight,
                        'content_weight': content_weight,
                        'temporal_weight': temporal_weight,
                        'image_size': image_size
                    }
                })
        
        # 合成视频
        print("\n合成最终视频...")
        self._create_video_from_frames(
            processed_frames,
            output_path,
            float(video_info['fps']),
            (image_size, image_size)
        )
        
        total_time = time.time() - start_time
        
        # 清理检查点
        if os.path.exists(checkpoint_path):
            os.remove(checkpoint_path)
        
        # 返回处理信息
        return {
            'input_video': video_path,
            'output_video': output_path,
            'style_image': style_image_path,
            'total_frames': total_frames,
            'fps': video_info['fps'],
            'resolution': f"{image_size}x{image_size}",
            'processing_time': total_time,
            'time_per_frame': total_time / total_frames,
            'params': {
                'num_steps': num_steps,
                'style_weight': style_weight,
                'temporal_weight': temporal_weight,
                'use_consistency': use_consistency
            }
        }
    
    def _create_video_from_frames(self, frame_paths: List[str], 
                                 output_path: str, fps: float,
                                 frame_size: tuple):
        """
        从帧序列创建视频
        
        Args:
            frame_paths: 帧文件路径列表
            output_path: 输出视频路径
            fps: 帧率
            frame_size: 帧大小 (width, height)
        """
        if not frame_paths:
            raise ValueError("没有帧可以合成视频")
        
        # 确保fps是有效值
        fps = float(fps)
        if fps <= 0:
            fps = 30.0
        
        # 确保输出目录存在
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # 确保输出文件是.mp4格式
        if not output_path.endswith('.mp4'):
            output_path = os.path.splitext(output_path)[0] + '.mp4'
        
        # 优先尝试使用FFmpeg（更可靠）
        print("尝试使用FFmpeg创建视频...")
        success = self._create_video_with_ffmpeg(frame_paths, output_path, fps, frame_size)
        
        if success:
            return
        
        # 如果FFmpeg失败，尝试OpenCV
        print("\n⚠️ FFmpeg不可用，尝试使用OpenCV编码器...")
        
        # 使用 H.264 编码器，兼容性更好
        # 尝试多个编码器以确保兼容性
        codecs = [
            ('mp4v', '.mp4'),  # MPEG-4 (macOS兼容性更好)
            ('avc1', '.mp4'),  # H.264
            ('XVID', '.avi'),  # Xvid
        ]
        
        success = False
        out = None
        
        for codec, ext in codecs:
            try:
                # 调整输出文件扩展名
                if not output_path.endswith(ext):
                    temp_output_path = os.path.splitext(output_path)[0] + ext
                else:
                    temp_output_path = output_path
                
                fourcc = cv2.VideoWriter_fourcc(*codec)
                out = cv2.VideoWriter(temp_output_path, fourcc, fps, frame_size)
                
                if out.isOpened():
                    print(f"使用编码器: {codec}")
                    
                    frames_written = 0
                    for frame_path in tqdm(frame_paths, desc="合成视频"):
                        # 读取图像
                        frame = cv2.imread(frame_path)
                        if frame is not None:
                            # 调整大小（如果需要）
                            if frame.shape[1] != frame_size[0] or frame.shape[0] != frame_size[1]:
                                frame = cv2.resize(frame, frame_size, interpolation=cv2.INTER_LANCZOS4)
                            
                            # 确保图像是正确的格式 (BGR, uint8)
                            if frame.dtype != np.uint8:
                                frame = (frame * 255).astype(np.uint8) if frame.max() <= 1.0 else frame.astype(np.uint8)
                            
                            # 确保是3通道BGR图像
                            if len(frame.shape) == 2:
                                frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
                            elif frame.shape[2] == 4:
                                frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
                            
                            out.write(frame)
                            frames_written += 1
                    
                    out.release()
                    
                    # 验证文件是否成功创建
                    if os.path.exists(temp_output_path) and os.path.getsize(temp_output_path) > 0:
                        print(f"✅ 视频已保存: {temp_output_path}")
                        print(f"   帧数: {frames_written}, 文件大小: {os.path.getsize(temp_output_path)/1024/1024:.2f} MB")
                        
                        # 如果临时路径与目标路径不同，重命名
                        if temp_output_path != output_path:
                            if os.path.exists(output_path):
                                os.remove(output_path)
                            os.rename(temp_output_path, output_path)
                            print(f"   已重命名为: {output_path}")
                        
                        success = True
                        break
                    else:
                        print(f"⚠️ 编码器 {codec} 创建的文件无效，尝试下一个...")
                        if out:
                            out.release()
                        if os.path.exists(temp_output_path):
                            os.remove(temp_output_path)
                else:
                    print(f"⚠️ 无法使用编码器 {codec}，尝试下一个...")
                    
            except Exception as e:
                print(f"⚠️ 编码器 {codec} 失败: {str(e)}")
                if out:
                    out.release()
                continue
        
        if not success:
            raise RuntimeError(
                "无法创建视频文件。请确保系统已安装必要的视频编解码器。\n"
                "在 macOS 上，请通过 Homebrew 安装: brew install ffmpeg\n"
                "或者安装完整的opencv: pip install opencv-contrib-python"
            )
    
    def _create_video_with_ffmpeg(self, frame_paths: List[str], 
                                   output_path: str, fps: float,
                                   frame_size: tuple) -> bool:
        """
        使用FFmpeg创建视频(推荐方案)
        
        Args:
            frame_paths: 帧文件路径列表
            output_path: 输出视频路径
            fps: 帧率
            frame_size: 帧大小 (width, height)
            
        Returns:
            是否成功
        """
        try:
            import subprocess
            import shutil
            
            # 检查ffmpeg是否可用
            if not shutil.which('ffmpeg'):
                print("⚠️ FFmpeg未安装")
                print("   在macOS上安装: brew install ffmpeg")
                print("   在Ubuntu上安装: sudo apt-get install ffmpeg")
                return False
            
            # 检查帧文件的实际尺寸
            first_frame = cv2.imread(frame_paths[0])
            if first_frame is None:
                print(f"⚠️ 无法读取第一帧: {frame_paths[0]}")
                return False
            
            actual_height, actual_width = first_frame.shape[:2]
            print(f"   检测到帧尺寸: {actual_width}x{actual_height}")
            
            # 使用实际尺寸或指定尺寸
            if (actual_width, actual_height) != frame_size:
                print(f"   将调整为: {frame_size[0]}x{frame_size[1]}")
                scale_filter = f'scale={frame_size[0]}:{frame_size[1]}'
            else:
                scale_filter = None
            
            # 创建临时目录并复制/链接帧文件
            work_dir = os.path.dirname(frame_paths[0])
            
            # 创建符号链接以符合ffmpeg的命名要求
            print("   准备帧文件...")
            link_paths = []
            for i, frame_path in enumerate(frame_paths):
                link_path = os.path.join(work_dir, f'ffmpeg_frame_{i:06d}.jpg')
                if os.path.exists(link_path):
                    os.remove(link_path)
                try:
                    os.symlink(os.path.abspath(frame_path), link_path)
                    link_paths.append(link_path)
                except:
                    # 如果符号链接失败,尝试复制
                    shutil.copy2(frame_path, link_path)
                    link_paths.append(link_path)
            
            # 使用ffmpeg命令
            input_pattern = os.path.join(work_dir, 'ffmpeg_frame_%06d.jpg')
            cmd = [
                'ffmpeg',
                '-y',  # 覆盖输出文件
                '-framerate', str(fps),
                '-i', input_pattern,
                '-c:v', 'libx264',  # H.264编码器
                '-preset', 'medium',  # 编码速度（fast/medium/slow）
                '-crf', '23',  # 质量 (18-28推荐，越小质量越高但文件越大)
                '-pix_fmt', 'yuv420p',  # 像素格式(yuv420p兼容性最好)
                '-movflags', '+faststart',  # 优化网络播放
            ]
            
            # 添加缩放滤镜（如果需要）
            if scale_filter:
                cmd.extend(['-vf', scale_filter])
            
            cmd.append(output_path)
            
            print(f"   执行FFmpeg命令...")
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            # 清理临时文件
            for link_path in link_paths:
                if os.path.exists(link_path):
                    try:
                        os.remove(link_path)
                    except:
                        pass
            
            if result.returncode == 0 and os.path.exists(output_path):
                file_size = os.path.getsize(output_path) / 1024 / 1024
                print(f"✅ 视频已保存: {output_path}")
                print(f"   帧数: {len(frame_paths)}, 文件大小: {file_size:.2f} MB")
                
                # 验证视频可以被读取
                cap = cv2.VideoCapture(output_path)
                if cap.isOpened():
                    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                    cap.release()
                    print(f"   验证成功: 视频包含 {frame_count} 帧")
                    return True
                else:
                    print(f"⚠️ 视频文件创建成功但无法打开，可能存在编码问题")
                    return False
            else:
                print(f"⚠️ FFmpeg执行失败")
                if result.stderr:
                    # 只显示最后几行错误信息
                    error_lines = result.stderr.strip().split('\n')
                    print("   错误信息:")
                    for line in error_lines[-5:]:
                        print(f"   {line}")
                return False
                
        except Exception as e:
            print(f"⚠️ FFmpeg方案失败: {str(e)}")
            import traceback
            traceback.print_exc()
            return False
    
    def _save_checkpoint(self, checkpoint_path: str, data: Dict):
        """保存检查点"""
        with open(checkpoint_path, 'w') as f:
            json.dump(data, f, indent=2)
    
    def _load_checkpoint(self, checkpoint_path: str) -> Optional[Dict]:
        """加载检查点"""
        if os.path.exists(checkpoint_path):
            try:
                with open(checkpoint_path, 'r') as f:
                    return json.load(f)
            except Exception as e:
                print(f"加载检查点失败: {e}")
                return None
        return None
    
    def create_preview_gif(self, video_path: str, output_path: str, 
                          num_frames: int = 10, fps: int = 5, 
                          max_width: int = 480):
        """
        从视频创建预览 GIF
        
        Args:
            video_path: 输入视频路径
            output_path: 输出 GIF 路径
            num_frames: GIF 帧数
            fps: GIF 帧率
            max_width: 最大宽度（用于压缩）
        """
        try:
            from PIL import Image
            
            cap = cv2.VideoCapture(video_path)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            
            if total_frames == 0:
                print("⚠️ 视频文件为空或无法读取")
                cap.release()
                return False
            
            # 均匀采样帧
            frame_indices = np.linspace(0, total_frames-1, min(num_frames, total_frames), dtype=int)
            
            frames = []
            for idx in frame_indices:
                cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
                ret, frame = cap.read()
                if ret:
                    # 转换颜色空间
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    
                    # 调整大小以减小文件
                    h, w = frame.shape[:2]
                    if w > max_width:
                        scale = max_width / w
                        new_w = max_width
                        new_h = int(h * scale)
                        frame = cv2.resize(frame, (new_w, new_h))
                    
                    frames.append(Image.fromarray(frame))
            
            cap.release()
            
            if frames:
                # 保存为 GIF
                duration = int(1000 / fps)  # 毫秒
                frames[0].save(
                    output_path,
                    save_all=True,
                    append_images=frames[1:],
                    duration=duration,
                    loop=0,
                    optimize=True
                )
                print(f"✅ 预览 GIF 已保存: {output_path}")
                return True
            else:
                print("⚠️ 无法提取视频帧")
                return False
                
        except Exception as e:
            print(f"⚠️ 创建预览 GIF 失败: {str(e)}")
            return False


# ==================== 命令行接口 ====================
if __name__ == "__main__":
    import sys
    import argparse
    from datetime import datetime
    
    # 获取项目根目录
    base_dir = os.path.dirname(os.path.abspath(__file__))
    
    # 设置默认路径
    default_content_dir = os.path.join(base_dir, 'data', 'content')
    default_style_dir = os.path.join(base_dir, 'data', 'style')
    default_output_dir = os.path.join(base_dir, 'data', 'outputs')
    
    # 确保目录存在
    os.makedirs(default_output_dir, exist_ok=True)
    
    # 创建参数解析器
    parser = argparse.ArgumentParser(
        description='视频风格迁移 - 将艺术风格应用到视频中',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 使用相对路径（自动在 data/ 目录中查找）
  python video_style_transfer.py my_video.mp4 starry_night.jpg
  
  # 使用完整路径
  python video_style_transfer.py data/content/video.mp4 data/style/style.jpg
  
  # 自定义输出路径和参数
  python video_style_transfer.py video.mp4 style.jpg -o my_output.mp4 -s 200 -f 200
  
  # 调整图像大小和关闭帧间一致性
  python video_style_transfer.py video.mp4 style.jpg --size 1024 --no-consistency
        """)
    
    parser.add_argument('video', type=str,
                       help='输入视频文件名（在 data/content/ 中）或完整路径')
    parser.add_argument('style', type=str,
                       help='风格图像文件名（在 data/style/ 中）或完整路径')
    parser.add_argument('-o', '--output', type=str, default=None,
                       help='输出视频路径（默认: data/outputs/styled_TIMESTAMP.mp4）')
    parser.add_argument('-s', '--steps', type=int, default=150,
                       help='优化步数（默认: 150，更多步数质量更好但速度更慢）')
    parser.add_argument('--size', type=int, default=512,
                       help='处理图像大小（默认: 512，越大越清晰但越慢）')
    parser.add_argument('-f', '--max-frames', type=int, default=None,
                       help='最大处理帧数（默认: 无限制，用于快速测试）')
    parser.add_argument('--style-weight', type=float, default=1e6,
                       help='风格权重（默认: 1e6）')
    parser.add_argument('--content-weight', type=float, default=1.0,
                       help='内容权重（默认: 1.0）')
    parser.add_argument('--temporal-weight', type=float, default=1e4,
                       help='时间一致性权重（默认: 1e4）')
    parser.add_argument('--no-consistency', action='store_true',
                       help='禁用帧间一致性优化（处理更快但可能闪烁）')
    parser.add_argument('--device', type=str, default=None,
                       help='计算设备: cuda 或 cpu（默认: 自动检测）')
    
    args = parser.parse_args()
    
    # 处理视频路径
    if os.path.exists(args.video):
        video_path = args.video
    else:
        video_path = os.path.join(default_content_dir, args.video)
        if not os.path.exists(video_path):
            print(f"❌ 错误: 找不到视频文件")
            print(f"   尝试路径: {args.video}")
            print(f"   尝试路径: {video_path}")
            print(f"\n💡 提示: 请将视频放在 {default_content_dir} 目录中")
            sys.exit(1)
    
    # 处理风格图像路径
    if os.path.exists(args.style):
        style_path = args.style
    else:
        style_path = os.path.join(default_style_dir, args.style)
        if not os.path.exists(style_path):
            print(f"❌ 错误: 找不到风格图像")
            print(f"   尝试路径: {args.style}")
            print(f"   尝试路径: {style_path}")
            print(f"\n💡 提示: 请将风格图像放在 {default_style_dir} 目录中")
            sys.exit(1)
    
    # 处理输出路径
    if args.output:
        if os.path.isabs(args.output):
            output_path = args.output
        else:
            output_path = os.path.join(default_output_dir, args.output)
    else:
        # 生成带时间戳的输出文件名
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        video_name = os.path.splitext(os.path.basename(video_path))[0]
        style_name = os.path.splitext(os.path.basename(style_path))[0]
        output_filename = f"styled_{video_name}_{style_name}_{timestamp}.mp4"
        output_path = os.path.join(default_output_dir, output_filename)
    
    # 确保输出目录存在
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # 设置设备
    device = args.device if args.device else ('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 显示配置信息
    print("=" * 80)
    print("🎨 视频风格迁移")
    print("=" * 80)
    print(f"📹 输入视频: {video_path}")
    print(f"🖼️  风格图像: {style_path}")
    print(f"💾 输出路径: {output_path}")
    print(f"⚙️  处理参数:")
    print(f"   - 优化步数: {args.steps}")
    print(f"   - 图像大小: {args.size}x{args.size}")
    print(f"   - 最大帧数: {args.max_frames if args.max_frames else '无限制'}")
    print(f"   - 风格权重: {args.style_weight:.0e}")
    print(f"   - 内容权重: {args.content_weight}")
    print(f"   - 时间一致性: {'禁用' if args.no_consistency else f'启用 (权重: {args.temporal_weight:.0e})'}")
    print(f"   - 计算设备: {device.upper()}")
    print("=" * 80)
    print()
    
    # 初始化视频风格迁移
    vst = VideoStyleTransfer(device=device)
    
    try:
        # 处理视频
        result = vst.process_video(
            video_path=video_path,
            style_image_path=style_path,
            output_path=output_path,
            num_steps=args.steps,
            style_weight=args.style_weight,
            content_weight=args.content_weight,
            temporal_weight=args.temporal_weight,
            image_size=args.size,
            max_frames=args.max_frames,
            use_consistency=not args.no_consistency
        )
        
        # 显示结果
        print("\n" + "=" * 80)
        print("✅ 处理完成!")
        print("=" * 80)
        print(f"📹 输入视频: {result['input_video']}")
        print(f"🎬 输出视频: {result['output_video']}")
        print(f"📊 视频信息:")
        print(f"   - 总帧数: {result['total_frames']}")
        print(f"   - 帧率: {result['fps']} FPS")
        print(f"   - 分辨率: {result['resolution']}")
        print(f"⏱️  处理时间:")
        print(f"   - 总耗时: {result['processing_time']/60:.1f} 分钟")
        print(f"   - 每帧耗时: {result['time_per_frame']:.1f} 秒")
        print(f"🎨 风格参数:")
        print(f"   - 优化步数: {result['params']['num_steps']}")
        print(f"   - 风格权重: {result['params']['style_weight']:.0e}")
        print(f"   - 帧间一致性: {'启用' if result['params']['use_consistency'] else '禁用'}")
        print("=" * 80)
        print(f"\n✨ 您可以在这里查看结果: {output_path}")
        
    except KeyboardInterrupt:
        print("\n\n⚠️  用户中断处理")
        print("💡 处理进度已保存，下次运行将从断点继续")
        sys.exit(0)
    except Exception as e:
        print(f"\n\n❌ 处理出错: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
