"""
配置文件 - 集中管理所有参数配置
"""

import os


class Config:
    """配置类"""
    
    # ==================== 路径配置 ====================
    # 项目根目录
    PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
    
    # 数据目录
    DATA_DIR = os.path.join(PROJECT_ROOT, "data")
    CONTENT_DIR = os.path.join(DATA_DIR, "content")
    STYLE_DIR = os.path.join(DATA_DIR, "style")
    OUTPUT_DIR = os.path.join(DATA_DIR, "outputs")
    
    # WikiArt数据集
    WIKIART_DIR = os.path.join(PROJECT_ROOT, "wikiart", "data")
    WIKIART_FILE = os.path.join(WIKIART_DIR, "train-00000-of-00072.parquet")
    
    # ==================== 模型配置 ====================
    # 设备
    DEVICE = 'auto'  # 'auto', 'cuda', 'cpu'
    
    # 图像尺寸
    IMAGE_SIZE = 512
    IMAGE_SIZE_FAST = 256  # 快速测试用
    
    # VGG层配置
    CONTENT_LAYERS = ['conv_4']
    STYLE_LAYERS = ['conv_1', 'conv_2', 'conv_3', 'conv_4', 'conv_5']
    
    # ImageNet归一化参数
    NORMALIZATION_MEAN = [0.485, 0.456, 0.406]
    NORMALIZATION_STD = [0.229, 0.224, 0.225]
    
    # ==================== 训练配置 ====================
    # 优化步数
    NUM_STEPS = 300
    NUM_STEPS_FAST = 100  # 快速测试
    NUM_STEPS_BATCH = 200  # 批量处理
    
    # 损失权重
    STYLE_WEIGHT = 1e6
    CONTENT_WEIGHT = 1
    
    # 保存间隔
    SAVE_EVERY = 50
    
    # ==================== 批量处理配置 ====================
    # 每个内容图片应用的风格数量
    NUM_STYLES_PER_CONTENT = 5
    
    # 推荐的艺术风格列表
    RECOMMENDED_STYLES = [
        'Impressionism',      # 印象派
        'Post_Impressionism', # 后印象派
        'Cubism',            # 立体派
        'Expressionism',     # 表现主义
        'Abstract_Expressionism',  # 抽象表现主义
        'Ukiyo_e',           # 浮世绘
        'Realism',           # 写实主义
        'Romanticism',       # 浪漫主义
        'Baroque',           # 巴洛克
        'Symbolism',         # 象征主义
    ]
    
    # 著名艺术家列表
    RECOMMENDED_ARTISTS = [
        'Vincent van Gogh',      # 梵高
        'Pablo Picasso',         # 毕加索
        'Claude Monet',          # 莫奈
        'Edvard Munch',          # 蒙克
        'Wassily Kandinsky',     # 康定斯基
        'Paul Cezanne',          # 塞尚
        'Henri Matisse',         # 马蒂斯
        'Salvador Dali',         # 达利
        'Jackson Pollock',       # 波洛克
        'Katsushika Hokusai',    # 葛饰北斋
    ]
    
    # ==================== 输出配置 ====================
    # 图像质量
    JPEG_QUALITY = 95
    
    # 对比图配置
    COMPARISON_FIGSIZE = (15, 5)
    COMPARISON_DPI = 150
    
    # GIF配置
    GIF_DURATION = 200  # 毫秒
    
    # 视频配置
    VIDEO_FPS = 10
    
    # ==================== 风格推荐配置 ====================
    # 推荐系统特征提取模型
    RECOMMENDATION_MODEL = 'resnet50'
    
    # 推荐数量范围
    MIN_RECOMMENDATIONS = 3
    MAX_RECOMMENDATIONS = 10
    DEFAULT_RECOMMENDATIONS = 5
    
    # 色彩特征权重
    HUE_WEIGHT = 0.3
    SATURATION_WEIGHT = 0.25
    BRIGHTNESS_WEIGHT = 0.25
    DIVERSITY_WEIGHT = 0.2
    
    # ==================== 视频处理配置 ====================
    # 视频处理默认参数
    VIDEO_DEFAULT_STEPS = 150
    VIDEO_DEFAULT_SIZE = 256
    VIDEO_MAX_FRAMES_DEFAULT = 0  # 0表示处理全部
    
    # 时间一致性权重
    TEMPORAL_WEIGHT = 1e4
    
    # 检查点保存间隔（帧数）
    CHECKPOINT_INTERVAL = 10
    
    # 视频输出格式
    VIDEO_CODEC = 'mp4v'
    VIDEO_EXTENSION = '.mp4'
    
    # ==================== 实验配置 ====================
    # 风格权重实验范围
    STYLE_WEIGHT_RANGE = [1e5, 5e5, 1e6, 5e6, 1e7]
    
    # 内容权重实验范围
    CONTENT_WEIGHT_RANGE = [0.5, 1, 2, 5, 10]
    
    # 步数实验范围
    STEPS_RANGE = [100, 200, 300, 500, 1000]
    
    @classmethod
    def create_directories(cls):
        """创建必要的目录"""
        directories = [
            cls.DATA_DIR,
            cls.CONTENT_DIR,
            cls.STYLE_DIR,
            cls.OUTPUT_DIR,
        ]
        
        for directory in directories:
            os.makedirs(directory, exist_ok=True)
        
        print("目录结构已创建：")
        for directory in directories:
            print(f"  {directory}")
    
    @classmethod
    def print_config(cls):
        """打印当前配置"""
        print("=" * 70)
        print("当前配置")
        print("=" * 70)
        
        print("\n路径配置：")
        print(f"  项目根目录: {cls.PROJECT_ROOT}")
        print(f"  内容目录: {cls.CONTENT_DIR}")
        print(f"  风格目录: {cls.STYLE_DIR}")
        print(f"  输出目录: {cls.OUTPUT_DIR}")
        print(f"  WikiArt: {cls.WIKIART_FILE}")
        
        print("\n模型配置：")
        print(f"  设备: {cls.DEVICE}")
        print(f"  图像尺寸: {cls.IMAGE_SIZE}")
        print(f"  内容层: {cls.CONTENT_LAYERS}")
        print(f"  风格层: {cls.STYLE_LAYERS}")
        
        print("\n训练配置：")
        print(f"  优化步数: {cls.NUM_STEPS}")
        print(f"  风格权重: {cls.STYLE_WEIGHT}")
        print(f"  内容权重: {cls.CONTENT_WEIGHT}")
        
        print("\n推荐风格: {cls.RECOMMENDED_STYLES[:5]}...")
        print("\n" + "=" * 70)
    
    @classmethod
    def get_device(cls):
        """获取设备"""
        import torch
        
        if cls.DEVICE == 'auto':
            return 'cuda' if torch.cuda.is_available() else 'cpu'
        return cls.DEVICE


# 快速配置预设
class FastConfig(Config):
    """快速测试配置"""
    IMAGE_SIZE = 256
    NUM_STEPS = 100
    NUM_STEPS_BATCH = 100


class HighQualityConfig(Config):
    """高质量配置"""
    IMAGE_SIZE = 1024
    NUM_STEPS = 500
    STYLE_WEIGHT = 5e6


# ==================== 测试代码 ====================
if __name__ == "__main__":
    # 打印配置
    Config.print_config()
    
    # 创建目录
    print("\n创建目录结构...")
    Config.create_directories()
    
    print("\n快速配置：")
    print(f"  图像尺寸: {FastConfig.IMAGE_SIZE}")
    print(f"  优化步数: {FastConfig.NUM_STEPS}")
    
    print("\n高质量配置：")
    print(f"  图像尺寸: {HighQualityConfig.IMAGE_SIZE}")
    print(f"  优化步数: {HighQualityConfig.NUM_STEPS}")
