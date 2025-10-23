import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms, models
from PIL import Image
import matplotlib.pyplot as plt
import copy

class NeuralStyleTransfer:
    def __init__(self, device='cuda' if torch.cuda.is_available() else 'cpu'):
        """
        初始化神经风格迁移
        
        Args:
            device: 'cuda' 或 'cpu'
        """
        self.device = device
        self.imsize = 512 if torch.cuda.is_available() else 256
        
        # 图像预处理
        self.loader = transforms.Compose([
            transforms.Resize((self.imsize, self.imsize)),
            transforms.ToTensor()
        ])
        
        self.unloader = transforms.ToPILImage()
        
        # VGG19模型 (使用weights参数替代deprecated的pretrained)
        self.vgg = models.vgg19(weights=models.VGG19_Weights.IMAGENET1K_V1).features.to(device).eval()
        
        # 归一化参数（ImageNet标准）
        self.normalization_mean = torch.tensor([0.485, 0.456, 0.406]).to(device)
        self.normalization_std = torch.tensor([0.229, 0.224, 0.225]).to(device)
        
        # 风格层和内容层
        self.content_layers = ['conv_4']
        self.style_layers = ['conv_1', 'conv_2', 'conv_3', 'conv_4', 'conv_5']
        
    def load_image(self, image_path):
        """加载图片"""
        image = Image.open(image_path).convert('RGB')
        image = self.loader(image).unsqueeze(0)
        return image.to(self.device, torch.float)
    
    def show_image(self, tensor, title=None, save_plot=False):
        """
        显示张量图片
        
        Args:
            tensor: 图片张量
            title: 标题
            save_plot: 是否保存matplotlib图表
        
        Returns:
            PIL Image对象
        """
        image = tensor.cpu().clone()
        image = image.squeeze(0)
        image = self.unloader(image)
        
        if save_plot:
            plt.figure(figsize=(8, 8))
            plt.imshow(image)
            if title:
                plt.title(title)
            plt.axis('off')
            plt.tight_layout()
        
        return image
    
    def gram_matrix(self, input_tensor):
        """
        计算Gram矩阵（用于风格损失）
        
        Args:
            input_tensor: (batch, channels, height, width)
        Returns:
            Gram矩阵
        """
        batch, channels, h, w = input_tensor.size()
        features = input_tensor.view(batch * channels, h * w)
        G = torch.mm(features, features.t())
        return G.div(batch * channels * h * w)
    
    def get_style_model_and_losses(self, style_img, content_img):
        """构建模型并设置损失层"""
        # 归一化层
        normalization = Normalization(self.normalization_mean, 
                                     self.normalization_std).to(self.device)
        
        content_losses = []
        style_losses = []
        
        model = nn.Sequential(normalization)
        
        i = 0
        for layer in self.vgg.children():
            if isinstance(layer, nn.Conv2d):
                i += 1
                name = f'conv_{i}'
            elif isinstance(layer, nn.ReLU):
                name = f'relu_{i}'
                layer = nn.ReLU(inplace=False)
            elif isinstance(layer, nn.MaxPool2d):
                name = f'pool_{i}'
            elif isinstance(layer, nn.BatchNorm2d):
                name = f'bn_{i}'
            else:
                raise RuntimeError(f'Unrecognized layer: {layer.__class__.__name__}')
            
            model.add_module(name, layer)
            
            if name in self.content_layers:
                target = model(content_img).detach()
                content_loss = ContentLoss(target)
                model.add_module(f"content_loss_{i}", content_loss)
                content_losses.append(content_loss)
            
            if name in self.style_layers:
                target_feature = model(style_img).detach()
                style_loss = StyleLoss(target_feature)
                model.add_module(f"style_loss_{i}", style_loss)
                style_losses.append(style_loss)
        
        # 裁剪到最后一个损失层之后
        for i in range(len(model) - 1, -1, -1):
            if isinstance(model[i], ContentLoss) or isinstance(model[i], StyleLoss):
                break
        model = model[:(i + 1)]
        
        return model, style_losses, content_losses
    
    def run_style_transfer(self, content_img, style_img, num_steps=300,
                          style_weight=1000000, content_weight=1):
        """
        执行风格迁移
        
        Args:
            content_img: 内容图片张量
            style_img: 风格图片张量
            num_steps: 优化步数
            style_weight: 风格权重
            content_weight: 内容权重
        Returns:
            生成的图片张量
        """
        print('Building the style transfer model...')
        model, style_losses, content_losses = self.get_style_model_and_losses(
            style_img, content_img)
        
        # 从内容图片开始优化
        input_img = content_img.clone()
        
        # 使用LBFGS优化器
        optimizer = optim.LBFGS([input_img.requires_grad_()])
        
        print('Optimizing...')
        run = [0]
        while run[0] <= num_steps:
            
            def closure():
                # 限制像素值范围
                with torch.no_grad():
                    input_img.clamp_(0, 1)
                
                optimizer.zero_grad()
                model(input_img)
                
                style_score = 0
                content_score = 0
                
                for sl in style_losses:
                    style_score += sl.loss
                for cl in content_losses:
                    content_score += cl.loss
                
                style_score *= style_weight
                content_score *= content_weight
                
                loss = style_score + content_score
                loss.backward()
                
                run[0] += 1
                if run[0] % 50 == 0:
                    print(f"Step [{run[0]}]:")
                    print(f'Style Loss: {style_score.item():.4f} '
                          f'Content Loss: {content_score.item():.4f}')
                    print()
                
                return style_score + content_score
            
            optimizer.step(closure)
        
        # 最终裁剪
        with torch.no_grad():
            input_img.clamp_(0, 1)
        
        return input_img


class Normalization(nn.Module):
    """归一化层"""
    def __init__(self, mean, std):
        super(Normalization, self).__init__()
        self.mean = mean.clone().view(-1, 1, 1)
        self.std = std.clone().view(-1, 1, 1)
    
    def forward(self, img):
        return (img - self.mean) / self.std


class ContentLoss(nn.Module):
    """内容损失"""
    def __init__(self, target):
        super(ContentLoss, self).__init__()
        self.target = target.detach()
    
    def forward(self, input):
        self.loss = nn.functional.mse_loss(input, self.target)
        return input


class StyleLoss(nn.Module):
    """风格损失"""
    def __init__(self, target_feature):
        super(StyleLoss, self).__init__()
        self.target = self.gram_matrix(target_feature).detach()
    
    def forward(self, input):
        G = self.gram_matrix(input)
        self.loss = nn.functional.mse_loss(G, self.target)
        return input
    
    def gram_matrix(self, input):
        batch, channels, h, w = input.size()
        features = input.view(batch * channels, h * w)
        G = torch.mm(features, features.t())
        return G.div(batch * channels * h * w)


# ==================== 使用示例 ====================
if __name__ == "__main__":
    # 初始化
    nst = NeuralStyleTransfer()
    
    # 加载图片（请替换为你的图片路径）
    content_img = nst.load_image('content.jpg')
    style_img = nst.load_image('style.jpg')
    
    # 显示原始图片
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 3, 1)
    nst.show_image(content_img, title='Content Image')
    
    plt.subplot(1, 3, 2)
    nst.show_image(style_img, title='Style Image')
    
    # 执行风格迁移
    output = nst.run_style_transfer(
        content_img, 
        style_img,
        num_steps=300,
        style_weight=1000000,
        content_weight=1
    )
    
    # 显示结果
    plt.subplot(1, 3, 3)
    result_img = nst.show_image(output, title='Output Image')
    
    plt.show()
    
    # 保存结果
    result_img.save('output.jpg')
    print("Result saved as 'output.jpg'")