"""
风格推荐系统 - 基于图像特征智能推荐艺术风格
"""

import torch
import numpy as np
from PIL import Image
from torchvision import transforms, models
import torch.nn.functional as F
from typing import List, Dict, Tuple
import json
import os


class StyleRecommendationSystem:
    """风格推荐系统"""
    
    def __init__(self, device='cuda' if torch.cuda.is_available() else 'cpu'):
        """
        初始化推荐系统
        
        Args:
            device: 计算设备
        """
        self.device = device
        
        # 加载预训练的特征提取器（使用ResNet50）
        self.feature_extractor = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
        self.feature_extractor = torch.nn.Sequential(*list(self.feature_extractor.children())[:-1])
        self.feature_extractor = self.feature_extractor.to(device).eval()
        
        # 图像预处理
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                              std=[0.229, 0.224, 0.225])
        ])
        
        # 风格特征库（将在运行时构建）
        self.style_profiles = self._initialize_style_profiles()
        
    def _initialize_style_profiles(self) -> Dict[str, Dict]:
        """
        初始化风格特征配置文件
        每个风格都有其倾向的色调、亮度、饱和度和内容特征
        """
        return {
            'Impressionism': {
                'name_cn': '印象派',
                'hue_range': (0.15, 0.65),  # 偏向暖色和绿色
                'brightness_range': (0.5, 0.9),  # 较明亮
                'saturation_range': (0.5, 0.9),  # 较高饱和度
                'content_keywords': ['landscape', 'nature', 'outdoor', 'flower', 'garden'],
                'description': '适合明亮、色彩丰富的自然风光和户外场景'
            },
            'Post_Impressionism': {
                'name_cn': '后印象派',
                'hue_range': (0.0, 1.0),  # 全色域
                'brightness_range': (0.3, 0.8),
                'saturation_range': (0.6, 1.0),  # 高饱和度
                'content_keywords': ['landscape', 'portrait', 'nature', 'sky'],
                'description': '适合需要强烈表现力的风景和肖像'
            },
            'Cubism': {
                'name_cn': '立体主义',
                'hue_range': (0.0, 0.3),  # 偏冷色调
                'brightness_range': (0.3, 0.7),
                'saturation_range': (0.3, 0.6),  # 中等饱和度
                'content_keywords': ['portrait', 'still life', 'geometric', 'building'],
                'description': '适合几何结构明显、有建筑感的场景'
            },
            'Expressionism': {
                'name_cn': '表现主义',
                'hue_range': (0.0, 0.2),  # 暗沉色调
                'brightness_range': (0.2, 0.6),  # 较暗
                'saturation_range': (0.5, 0.9),
                'content_keywords': ['portrait', 'emotion', 'people', 'dark'],
                'description': '适合需要表达强烈情感的人物和场景'
            },
            'Abstract_Expressionism': {
                'name_cn': '抽象表现主义',
                'hue_range': (0.0, 1.0),
                'brightness_range': (0.2, 0.8),
                'saturation_range': (0.4, 1.0),
                'content_keywords': ['abstract', 'color', 'texture'],
                'description': '适合任何需要艺术化表达的内容'
            },
            'Ukiyo_e': {
                'name_cn': '浮世绘',
                'hue_range': (0.5, 0.7),  # 蓝绿色调
                'brightness_range': (0.4, 0.8),
                'saturation_range': (0.6, 0.9),
                'content_keywords': ['landscape', 'nature', 'water', 'mountain', 'wave'],
                'description': '适合山水、海浪等自然景观'
            },
            'Realism': {
                'name_cn': '写实主义',
                'hue_range': (0.0, 1.0),
                'brightness_range': (0.3, 0.7),
                'saturation_range': (0.3, 0.7),  # 中等饱和度
                'content_keywords': ['portrait', 'people', 'life', 'realistic'],
                'description': '适合需要保持真实感的人物和生活场景'
            },
            'Romanticism': {
                'name_cn': '浪漫主义',
                'hue_range': (0.0, 0.4),  # 偏暖色
                'brightness_range': (0.3, 0.7),
                'saturation_range': (0.5, 0.8),
                'content_keywords': ['landscape', 'nature', 'dramatic', 'sunset'],
                'description': '适合戏剧性的自然景观和日落场景'
            },
            'Baroque': {
                'name_cn': '巴洛克',
                'hue_range': (0.0, 0.15),  # 深色调
                'brightness_range': (0.2, 0.5),  # 较暗
                'saturation_range': (0.4, 0.7),
                'content_keywords': ['portrait', 'dramatic', 'indoor', 'classical'],
                'description': '适合需要戏剧性光影效果的室内和古典场景'
            },
            'Symbolism': {
                'name_cn': '象征主义',
                'hue_range': (0.5, 0.8),  # 冷色调
                'brightness_range': (0.3, 0.6),
                'saturation_range': (0.4, 0.8),
                'content_keywords': ['mystical', 'dream', 'fantasy', 'symbolic'],
                'description': '适合梦幻、神秘的场景'
            },
            'Pop_Art': {
                'name_cn': '波普艺术',
                'hue_range': (0.0, 1.0),
                'brightness_range': (0.5, 1.0),  # 明亮
                'saturation_range': (0.8, 1.0),  # 极高饱和度
                'content_keywords': ['portrait', 'colorful', 'bold', 'modern'],
                'description': '适合色彩鲜艳、对比强烈的现代场景'
            },
            'Minimalism': {
                'name_cn': '极简主义',
                'hue_range': (0.0, 0.2),
                'brightness_range': (0.4, 0.9),
                'saturation_range': (0.1, 0.4),  # 低饱和度
                'content_keywords': ['simple', 'minimal', 'clean', 'geometric'],
                'description': '适合简洁、几何感强的场景'
            }
        }
    
    def extract_color_features(self, image: Image.Image) -> Dict[str, float]:
        """
        提取图像的色彩特征
        
        Args:
            image: PIL图像
            
        Returns:
            色彩特征字典
        """
        # 转换为RGB
        img_rgb = image.convert('RGB')
        img_array = np.array(img_rgb) / 255.0
        
        # 转换为HSV以分析色调、饱和度、亮度
        img_hsv = image.convert('HSV')
        hsv_array = np.array(img_hsv) / 255.0
        
        # 计算平均色调、饱和度、亮度
        avg_hue = np.mean(hsv_array[:, :, 0])
        avg_saturation = np.mean(hsv_array[:, :, 1])
        avg_brightness = np.mean(hsv_array[:, :, 2])
        
        # 计算色调分布（分成12个区间）
        hue_hist, _ = np.histogram(hsv_array[:, :, 0], bins=12, range=(0, 1))
        hue_hist = hue_hist / hue_hist.sum()  # 归一化
        
        # 计算主色调（占比最大的色调区间）
        dominant_hue_idx = np.argmax(hue_hist)
        dominant_hue = dominant_hue_idx / 12.0
        
        # 计算颜色多样性（熵）
        color_diversity = -np.sum(hue_hist * np.log(hue_hist + 1e-10))
        
        return {
            'avg_hue': float(avg_hue),
            'avg_saturation': float(avg_saturation),
            'avg_brightness': float(avg_brightness),
            'dominant_hue': float(dominant_hue),
            'color_diversity': float(color_diversity),
            'hue_distribution': hue_hist.tolist()
        }
    
    def extract_content_features(self, image: Image.Image) -> torch.Tensor:
        """
        提取图像的内容特征（使用ResNet）
        
        Args:
            image: PIL图像
            
        Returns:
            特征向量
        """
        img_tensor = self.transform(image).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            features = self.feature_extractor(img_tensor)
            features = features.squeeze(-1).squeeze(-1)  # 去除空间维度
            features = F.normalize(features, p=2, dim=1)  # L2归一化
        
        return features
    
    def calculate_style_score(self, color_features: Dict[str, float], 
                             style_profile: Dict) -> float:
        """
        计算内容图与某个风格的匹配分数
        
        Args:
            color_features: 图像色彩特征
            style_profile: 风格配置
            
        Returns:
            匹配分数 (0-1)
        """
        score = 0.0
        weights_sum = 0.0
        
        # 色调匹配（权重：0.3）
        hue_range = style_profile['hue_range']
        if hue_range[0] <= color_features['avg_hue'] <= hue_range[1]:
            hue_score = 1.0
        else:
            # 计算距离
            if color_features['avg_hue'] < hue_range[0]:
                hue_score = 1.0 - (hue_range[0] - color_features['avg_hue'])
            else:
                hue_score = 1.0 - (color_features['avg_hue'] - hue_range[1])
            hue_score = max(0.0, hue_score)
        score += hue_score * 0.3
        weights_sum += 0.3
        
        # 饱和度匹配（权重：0.25）
        sat_range = style_profile['saturation_range']
        if sat_range[0] <= color_features['avg_saturation'] <= sat_range[1]:
            sat_score = 1.0
        else:
            if color_features['avg_saturation'] < sat_range[0]:
                sat_score = 1.0 - (sat_range[0] - color_features['avg_saturation'])
            else:
                sat_score = 1.0 - (color_features['avg_saturation'] - sat_range[1])
            sat_score = max(0.0, sat_score)
        score += sat_score * 0.25
        weights_sum += 0.25
        
        # 亮度匹配（权重：0.25）
        bright_range = style_profile['brightness_range']
        if bright_range[0] <= color_features['avg_brightness'] <= bright_range[1]:
            bright_score = 1.0
        else:
            if color_features['avg_brightness'] < bright_range[0]:
                bright_score = 1.0 - (bright_range[0] - color_features['avg_brightness'])
            else:
                bright_score = 1.0 - (color_features['avg_brightness'] - bright_range[1])
            bright_score = max(0.0, bright_score)
        score += bright_score * 0.25
        weights_sum += 0.25
        
        # 颜色多样性（权重：0.2）
        # 高多样性适合色彩丰富的风格
        diversity_styles = ['Impressionism', 'Post_Impressionism', 'Abstract_Expressionism']
        if any(s in style_profile.get('name_cn', '') for s in ['印象', '抽象', '波普']):
            diversity_score = color_features['color_diversity'] / 2.5  # 归一化到0-1
            diversity_score = min(1.0, diversity_score)
        else:
            # 对于其他风格，多样性不是关键因素
            diversity_score = 0.5
        score += diversity_score * 0.2
        weights_sum += 0.2
        
        return score / weights_sum if weights_sum > 0 else 0.0
    
    def recommend_styles(self, image: Image.Image, top_k: int = 5) -> List[Dict]:
        """
        为图像推荐适合的艺术风格
        
        Args:
            image: 输入的内容图像
            top_k: 返回前k个推荐
            
        Returns:
            推荐列表，每项包含风格名、分数和描述
        """
        # 提取特征
        color_features = self.extract_color_features(image)
        
        # 计算每个风格的匹配分数
        recommendations = []
        for style_name, style_profile in self.style_profiles.items():
            score = self.calculate_style_score(color_features, style_profile)
            
            recommendations.append({
                'style': style_name,
                'style_cn': style_profile['name_cn'],
                'score': score,
                'description': style_profile['description'],
                'color_features': {
                    'hue': color_features['avg_hue'],
                    'saturation': color_features['avg_saturation'],
                    'brightness': color_features['avg_brightness']
                }
            })
        
        # 按分数排序
        recommendations.sort(key=lambda x: x['score'], reverse=True)
        
        return recommendations[:top_k]
    
    def explain_recommendation(self, recommendation: Dict, color_features: Dict) -> str:
        """
        解释推荐理由
        
        Args:
            recommendation: 推荐结果
            color_features: 图像色彩特征
            
        Returns:
            解释文本
        """
        style = recommendation['style']
        profile = self.style_profiles[style]
        score = recommendation['score']
        
        explanation = f"**{recommendation['style_cn']} ({style})**\n"
        explanation += f"匹配度: {score:.1%}\n\n"
        explanation += f"{profile['description']}\n\n"
        explanation += "**匹配分析:**\n"
        
        # 色调分析
        hue_val = color_features['avg_hue']
        hue_range = profile['hue_range']
        if hue_range[0] <= hue_val <= hue_range[1]:
            explanation += f"✓ 色调匹配良好 ({hue_val:.2f})\n"
        else:
            explanation += f"○ 色调 ({hue_val:.2f}) 与推荐范围略有差异\n"
        
        # 饱和度分析
        sat_val = color_features['avg_saturation']
        sat_range = profile['saturation_range']
        if sat_range[0] <= sat_val <= sat_range[1]:
            explanation += f"✓ 饱和度匹配良好 ({sat_val:.2f})\n"
        else:
            explanation += f"○ 饱和度 ({sat_val:.2f}) 可调整\n"
        
        # 亮度分析
        bright_val = color_features['avg_brightness']
        bright_range = profile['brightness_range']
        if bright_range[0] <= bright_val <= bright_range[1]:
            explanation += f"✓ 亮度匹配良好 ({bright_val:.2f})\n"
        else:
            explanation += f"○ 亮度 ({bright_val:.2f}) 可调整\n"
        
        return explanation


# ==================== 测试代码 ====================
if __name__ == "__main__":
    import os
    from pathlib import Path
    
    # 初始化推荐系统
    recommender = StyleRecommendationSystem()
    
    # 测试图像路径
    test_image_path = "data/content/test.jpg"
    
    if os.path.exists(test_image_path):
        # 加载测试图像
        image = Image.open(test_image_path)
        
        # 获取推荐
        recommendations = recommender.recommend_styles(image, top_k=5)
        
        # 提取色彩特征用于解释
        color_features = recommender.extract_color_features(image)
        
        print("=" * 70)
        print("图像色彩特征分析")
        print("=" * 70)
        print(f"平均色调: {color_features['avg_hue']:.3f}")
        print(f"平均饱和度: {color_features['avg_saturation']:.3f}")
        print(f"平均亮度: {color_features['avg_brightness']:.3f}")
        print(f"颜色多样性: {color_features['color_diversity']:.3f}")
        print()
        
        print("=" * 70)
        print(f"TOP-{len(recommendations)} 风格推荐")
        print("=" * 70)
        
        for i, rec in enumerate(recommendations, 1):
            print(f"\n{i}. {recommender.explain_recommendation(rec, color_features)}")
            print("-" * 70)
    else:
        print(f"测试图像不存在: {test_image_path}")
        print("请将测试图像放在 data/content/ 目录下")
