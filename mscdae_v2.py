import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms
from torchvision.transforms.functional import resize
import cv2
import os
import matplotlib.pyplot as plt
from PIL import Image
import random
from datetime import datetime
import logging

# Setup logging
try:
    # Ensure log directory exists
    log_path = "logs/mscdae_v2_train.log"
    log_dir = os.path.dirname(os.path.abspath(log_path))
    os.makedirs(log_dir, exist_ok=True)
    
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_path, mode='a'),  # 追加模式
            logging.StreamHandler()
        ]
    )
    # Force output of initial log to confirm logging system is working
    logging.info("Logging system initialized")
except Exception as e:
    print(f"Error setting up logging system: {str(e)}")

logger = logging.getLogger("MscdaeModule")
# Confirm logger is working properly
logger.info("Mscdae module initialized")

class SaltPepperNoise(object):
    """为图像添加椒盐噪声"""
    def __init__(self, prob=0.05):
        self.prob = prob
        
    def __call__(self, img_tensor):
        # 转换为numpy数组
        img_np = img_tensor.numpy()
        # 添加椒盐噪声
        noise_mask = np.random.random(img_np.shape) < self.prob
        salt_mask = np.random.random(img_np.shape) < 0.5
        # 添加白色像素(salt)
        img_np[noise_mask & salt_mask] = 1.0
        # 添加黑色像素(pepper)
        img_np[noise_mask & ~salt_mask] = 0.0
        # 转回tensor
        return torch.from_numpy(img_np)

class NormalizeWeber(object):
    """使用韦伯法则进行光照归一化"""
    def __init__(self, epsilon=1e-6):
        self.epsilon = epsilon
        
    def __call__(self, img_tensor):
        # 计算每个通道的亮度均值
        mean_intensity = torch.mean(img_tensor) + self.epsilon
        # 应用韦伯法则: I_normalized = (I - mean) / mean
        normalized = (img_tensor - mean_intensity) / mean_intensity
        # 将值缩放到[0,1]范围
        min_val = torch.min(normalized)
        max_val = torch.max(normalized)
        normalized = (normalized - min_val) / (max_val - min_val + self.epsilon)
        return normalized

class GaussianPyramid(object):
    """生成图像的高斯金字塔"""
    def __init__(self, levels=3):
        self.levels = levels
        
    def __call__(self, img_tensor):
        # 转换为numpy数组
        img_np = img_tensor.numpy()[0]  # 取第一个通道，假设为灰度图
        pyramid = [img_np]
        
        for i in range(1, self.levels):
            img_np = cv2.pyrDown(img_np)
            pyramid.append(img_np)
            
        # 转回tensor
        pyramid_tensors = [torch.from_numpy(img).unsqueeze(0) for img in pyramid]
        return pyramid_tensors

class PatchExtractor(object):
    """从图像中提取固定大小的图像块"""
    def __init__(self, patch_size=64, stride=32):
        self.patch_size = patch_size
        self.stride = stride
        
    def __call__(self, img_tensor):
        patches = []
        c, h, w = img_tensor.shape
        
        for i in range(0, h - self.patch_size + 1, self.stride):
            for j in range(0, w - self.patch_size + 1, self.stride):
                patch = img_tensor[:, i:i+self.patch_size, j:j+self.patch_size]
                patches.append(patch)
                
        return patches

class DefectDataset(Dataset):
    """缺陷检测数据集"""
    def __init__(self, image_dir, transform=None, patch_size=64, stride=32):
        self.image_dir = image_dir
        self.transform = transform
        self.patch_extractor = PatchExtractor(patch_size, stride)
        self.image_files = [f for f in os.listdir(image_dir) if f.endswith(('.png', '.jpg', '.jpeg'))]
        
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        img_path = os.path.join(self.image_dir, self.image_files[idx])
        # 读取图像并转换为灰度
        img = Image.open(img_path).convert('L')
        # 转换为tensor
        img_tensor = transforms.ToTensor()(img)
        
        if self.transform:
            img_tensor = self.transform(img_tensor)
            
        return img_tensor

class ConvolutionalDenoisingAutoencoder(nn.Module):
    """卷积去噪自编码器模型"""
    def __init__(self, input_channels=1):
        super(ConvolutionalDenoisingAutoencoder, self).__init__()
        
        # 编码器
        self.encoder = nn.Sequential(
            nn.Conv2d(input_channels, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(True)
        )
        
        # 解码器
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(True),
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(True),
            nn.Conv2d(32, input_channels, kernel_size=3, stride=1, padding=1),
            nn.Sigmoid()  # 输出范围在[0,1]
        )
        
    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

class MSCDAE:
    """多尺度卷积去噪自编码器"""
    def __init__(self, levels=3, patch_size=64, stride=32, batch_size=32, epochs=50, learning_rate=0.001,
                 noise_prob=0.05, gamma=3.0, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.levels = levels
        self.patch_size = patch_size
        self.stride = stride
        self.batch_size = batch_size
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.noise_prob = noise_prob
        self.gamma = gamma  # 阈值参数
        self.device = device
        
        # 为每个金字塔层级创建一个CDAE模型
        self.models = [ConvolutionalDenoisingAutoencoder().to(self.device) for _ in range(levels)]
        self.optimizers = [optim.Adam(model.parameters(), lr=learning_rate) for model in self.models]
        self.criterion = nn.MSELoss()
        
        # 存储每个层级的重建误差统计
        self.error_stats = [{'mean': None, 'std': None, 'threshold': None} for _ in range(levels)]
        
        logger.info(f"Initialize MSCDAE model: pyramid level = {levels}, device = {device}")
        
    def train(self, image_dir):
        """训练模型"""
        logger.info(f"Start training: image_dir={image_dir}")
        
        # 数据预处理和变换
        transform = transforms.Compose([
            transforms.Resize((256, 256)),  # 调整为固定大小
            NormalizeWeber(),  # 光照归一化
        ])
        
        noise_transform = transforms.Compose([
            SaltPepperNoise(prob=self.noise_prob)  # 添加椒盐噪声
        ])
        
        # 创建数据集
        dataset = DefectDataset(image_dir, transform=transform, patch_size=self.patch_size, stride=self.stride)
        dataloader = DataLoader(dataset, batch_size=1, shuffle=True)  # 每次加载一张图像
        
        # 对每个金字塔层级训练一个CDAE模型
        for level in range(self.levels):
            logger.info(f"Training pyramid level {level + 1}/{self.levels}")
            model = self.models[level]
            optimizer = self.optimizers[level]
            model.train()
            
            all_errors = []  # 收集所有重建误差
            
            for epoch in range(self.epochs):
                epoch_loss = 0.0
                batch_count = 0
                
                for i, img in enumerate(dataloader):
                    # 缩放到当前金字塔层级
                    scale_factor = 0.5 ** level
                    scaled_size = (int(img.shape[2] * scale_factor), int(img.shape[3] * scale_factor))
                    scaled_img = resize(img, scaled_size)
                    
                    # 提取图像块
                    patches = []
                    for h in range(0, scaled_img.shape[2] - self.patch_size + 1, self.stride):
                        for w in range(0, scaled_img.shape[3] - self.patch_size + 1, self.stride):
                            patch = scaled_img[:, :, h:h+self.patch_size, w:w+self.patch_size]
                            patches.append(patch)
                    
                    if not patches:
                        continue
                    
                    # 将图像块组合成一个批次
                    batch = torch.cat(patches, dim=0).to(self.device)
                    
                    # 添加噪声
                    noisy_batch = []
                    # noisy_batch = torch.stack([noise_transform(patch.squeeze(0)).unsqueeze(0) for patch in patches], dim=0).to(self.device)
                    
                    for patch in patches:
                        # 确保patch维度正确 [1, C, H, W]
                        if patch.dim() == 4:
                            p = patch.squeeze(0)  # 变成 [C, H, W]
                        else:
                            p = patch
                        
                        # 添加噪声
                        noisy_p = noise_transform(p)
                        
                        # 确保正确的维度
                        if noisy_p.dim() == 3:
                            noisy_p = noisy_p.unsqueeze(0)  # 变回 [1, C, H, W]
                        
                        noisy_batch.append(noisy_p)
                    
                    noisy_batch = torch.cat(noisy_batch, dim=0).to(self.device)

                    # 训练步骤
                    optimizer.zero_grad()
                    outputs = model(noisy_batch)
                    loss = self.criterion(outputs, batch)
                    loss.backward()
                    optimizer.step()
                    
                    # 收集重建误差用于统计
                    with torch.no_grad():
                        for j in range(batch.size(0)):
                            clean_patch = batch[j].unsqueeze(0)
                            noisy_patch = noisy_batch[j].unsqueeze(0)
                            output = model(noisy_patch)
                            error = torch.mean((output - clean_patch) ** 2).item()
                            all_errors.append(error)
                    
                    epoch_loss += loss.item() * batch.size(0)
                    batch_count += batch.size(0)
                
                # 输出每个epoch的训练损失
                avg_loss = epoch_loss / batch_count if batch_count > 0 else 0
                logger.info(f"  Epoch {epoch+1}/{self.epochs}, Loss: {avg_loss:.6f}")
            
            # 计算重建误差的统计信息
            if all_errors:
                mean_error = np.mean(all_errors)
                std_error = np.std(all_errors)
                threshold = mean_error + self.gamma * std_error
                
                self.error_stats[level] = {
                    'mean': mean_error,
                    'std': std_error,
                    'threshold': threshold
                }
                
                logger.info(f"Level {level + 1} Statistics: Mean={mean_error:.6f}, Std={std_error:.6f}, Threshold={threshold:.6f}")
        
        logger.info("Training Completed")
        return self
    
    def save_model(self, save_dir):
        """保存模型和统计信息"""
        os.makedirs(save_dir, exist_ok=True)
        
        # 保存模型参数
        for level, model in enumerate(self.models):
            torch.save(model.state_dict(), os.path.join(save_dir, f"model_level_{level}.pth"))
        
        # 保存重建误差统计信息
        np.save(os.path.join(save_dir, "error_stats.npy"), self.error_stats)
        
        # 保存配置信息
        config = {
            'levels': self.levels,
            'patch_size': self.patch_size,
            'stride': self.stride,
            'gamma': self.gamma
        }
        np.save(os.path.join(save_dir, "config.npy"), config)
        
        logger.info(f"Model is saved to: {save_dir}")
    
    def load_model(self, save_dir):
        """加载已保存的模型和统计信息"""
        # 加载模型参数
        for level in range(self.levels):
            self.models[level].load_state_dict(torch.load(os.path.join(save_dir, f"model_level_{level}.pth"), map_location=self.device))
        
        # 加载重建误差统计信息
        self.error_stats = np.load(os.path.join(save_dir, "error_stats.npy"), allow_pickle=True).tolist()
        
        # 加载配置信息
        config = np.load(os.path.join(save_dir, "config.npy"), allow_pickle=True).item()
        self.levels = config['levels']
        self.patch_size = config['patch_size']
        self.stride = config['stride']
        self.gamma = config['gamma']
        
        logger.info(f"Load model from {save_dir}")
        return self
    
    def detect(self, image_path, output_dir=None):
        """检测图像中的缺陷"""
        logger.info(f"Detect image: {image_path}")
        
        # 数据预处理
        transform = transforms.Compose([
            transforms.Resize((256, 256)),  # 调整为固定大小
            NormalizeWeber(),  # 光照归一化
        ])
        
        # 读取图像
        img = Image.open(image_path).convert('L')
        img_tensor = transform(transforms.ToTensor()(img)).unsqueeze(0)  # [1, 1, H, W]
        
        # 保存原始图像用于可视化
        original_img = img_tensor.clone()
        
        # 对每个金字塔层级进行缺陷检测
        residual_maps = []
        segmentation_maps = []
        
        for level in range(self.levels):
            logger.info(f"Process pyramid level {level + 1}/{self.levels}")
            model = self.models[level]
            model.eval()
            
            # 缩放到当前金字塔层级
            scale_factor = 0.5 ** level
            scaled_size = (int(img_tensor.shape[2] * scale_factor), int(img_tensor.shape[3] * scale_factor))
            scaled_img = resize(img_tensor, scaled_size)
            
            # 创建空的残差图
            residual_map = torch.zeros((1, 1, scaled_img.shape[2], scaled_img.shape[3])).to(self.device)
            count_map = torch.zeros((1, 1, scaled_img.shape[2], scaled_img.shape[3])).to(self.device)
            
            # 滑动窗口提取图像块并计算重建误差
            with torch.no_grad():
                for h in range(0, scaled_img.shape[2] - self.patch_size + 1, self.stride):
                    for w in range(0, scaled_img.shape[3] - self.patch_size + 1, self.stride):
                        patch = scaled_img[:, :, h:h+self.patch_size, w:w+self.patch_size].to(self.device)
                        output = model(patch)
                        
                        # 计算重建残差
                        error = (output - patch) ** 2
                        
                        # 更新残差图和计数图
                        residual_map[:, :, h:h+self.patch_size, w:w+self.patch_size] += error
                        count_map[:, :, h:h+self.patch_size, w:w+self.patch_size] += 1
            
            # 处理计数为0的位置
            count_map[count_map == 0] = 1
            # 计算平均残差
            residual_map = residual_map / count_map
            
            # 调整残差图到原始大小
            residual_map_full = resize(residual_map, (img_tensor.shape[2], img_tensor.shape[3]))
            residual_maps.append(residual_map_full)
            
            # 应用阈值进行缺陷分割
            threshold = self.error_stats[level]['threshold']
            segmentation_map = (residual_map_full > threshold).float()
            segmentation_maps.append(segmentation_map)
        
        # 融合不同层级的分割结果
        # 使用逻辑运算：相邻层使用AND，最终结果使用OR
        final_segmentation = torch.zeros_like(segmentation_maps[0])
        
        # 如果只有一个层级，直接使用该层级的分割图
        if self.levels == 1:
            final_segmentation = segmentation_maps[0]
        else:
            # 首先对相邻层级进行AND运算
            and_results = []
            for i in range(self.levels - 1):
                and_result = segmentation_maps[i] * segmentation_maps[i + 1]  # 逻辑AND
                and_results.append(and_result)
            
            # 然后对AND结果进行OR运算
            for and_result in and_results:
                final_segmentation = torch.max(final_segmentation, and_result)  # 逻辑OR
        
        # 计算统计信息
        defect_ratio = torch.sum(final_segmentation) / (final_segmentation.shape[2] * final_segmentation.shape[3])
        defect_ratio = defect_ratio.item()
        
        # 判断是否为NC产品
        is_nc = defect_ratio > 0.01  # 如果缺陷像素比例>1%，则判断为NC产品
        
        result = {
            'is_nc': is_nc,
            'defect_ratio': defect_ratio,
            'segmentation': final_segmentation.cpu().numpy()
        }
        
        # 如果指定了输出目录，则保存可视化结果
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            
            # 保存原始图像
            original_np = original_img.squeeze().cpu().numpy()
            plt.figure(figsize=(10, 10))
            plt.imshow(original_np, cmap='gray')
            plt.title("Original Image")
            plt.savefig(os.path.join(output_dir, "original.png"))
            plt.close()
            
            # 保存残差图
            for i, residual_map in enumerate(residual_maps):
                residual_np = residual_map.squeeze().cpu().numpy()
                plt.figure(figsize=(10, 10))
                plt.imshow(residual_np, cmap='jet')
                plt.colorbar()
                plt.title(f"Residual Map - Level {i+1}")
                plt.savefig(os.path.join(output_dir, f"residual_level_{i+1}.png"))
                plt.close()
            
            # 保存分割图
            for i, segmentation_map in enumerate(segmentation_maps):
                segmentation_np = segmentation_map.squeeze().cpu().numpy()
                plt.figure(figsize=(10, 10))
                plt.imshow(segmentation_np, cmap='gray')
                plt.title(f"Segmentation - Level {i+1}")
                plt.savefig(os.path.join(output_dir, f"segmentation_level_{i+1}.png"))
                plt.close()
            
            # 保存最终分割结果
            final_np = final_segmentation.squeeze().cpu().numpy()
            plt.figure(figsize=(10, 10))
            plt.imshow(final_np, cmap='gray')
            plt.title(f"Final Segmentation (NC: {is_nc}, Defect Ratio: {defect_ratio:.6f})")
            plt.savefig(os.path.join(output_dir, "final_segmentation.png"))
            plt.close()
            
            # 在原始图像上标注缺陷区域
            original_rgb = np.stack([original_np, original_np, original_np], axis=2)
            overlay = original_rgb.copy()
            overlay[final_np > 0, 0] = 1.0  # 红色标注缺陷
            overlay[final_np > 0, 1:] = 0.0
            # 设置透明度
            alpha = 0.5  # 透明度值，范围在0到1之间
            overlay[final_np > 0] = alpha * overlay[final_np > 0] + (1 - alpha) * original_rgb[final_np > 0] 

            plt.figure(figsize=(10, 10))
            plt.imshow(overlay)
            plt.title(f"Defect Overlay (NC: {is_nc}, Defect Ratio: {defect_ratio:.6f})")
            plt.savefig(os.path.join(output_dir, "defect_overlay.png"))
            plt.close()
        
        logger.info(f"Test result: {'NC' if is_nc else 'AC'}, defect ratio: {defect_ratio:.6f}")
        return result
    
    def test_batch_images(self, image_dir, output_dir=None, threshold=0.01):
        """批量测试图像并统计结果"""
        logger.info(f"Batch test images: {image_dir}")
        
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        
        # 获取目录中的所有图像
        image_files = [f for f in os.listdir(image_dir) if f.endswith(('.png', '.jpg', '.jpeg'))]
        
        results = []
        nc_count = 0
        
        for image_file in image_files:
            image_path = os.path.join(image_dir, image_file)
            
            # 为每个图像创建输出子目录
            if output_dir:
                image_output_dir = os.path.join(output_dir, os.path.splitext(image_file)[0])
                os.makedirs(image_output_dir, exist_ok=True)
            else:
                image_output_dir = None
            
            # 检测缺陷
            result = self.detect(image_path, image_output_dir)
            
            # 统计结果
            if result['is_nc']:
                nc_count += 1
            
            results.append({
                'image': image_file,
                'is_nc': result['is_nc'],
                'defect_ratio': result['defect_ratio']
            })
        
        # 计算NC产品比例
        nc_ratio = nc_count / len(image_files) if image_files else 0
        
        # 保存汇总结果
        summary = {
            'total_images': len(image_files),
            'nc_count': nc_count,
            'nc_ratio': nc_ratio,
            'threshold': threshold,
            'results': results
        }
        
        if output_dir:
            # 保存汇总报告
            with open(os.path.join(output_dir, "summary_report.txt"), "w") as f:
                f.write(f"Test time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"Total number of images: {len(image_files)}\n")
                f.write(f"NC product quantity: {nc_count}\n")
                f.write(f"NC product ratio: {nc_ratio:.4f}\n")
                f.write(f"Defect threshold: {threshold}\n\n")
                f.write("Detailed results:\n")
                for result in results:
                    f.write(f"Image: {result['image']}, Result: {'NC' if result['is_nc'] else 'AC'}, Defect ratio: {result['defect_ratio']:.6f}\n")
            
            # 保存汇总图表
            if results:
                defect_ratios = [r['defect_ratio'] for r in results]
                plt.figure(figsize=(12, 6))
                plt.bar(range(len(defect_ratios)), defect_ratios)
                plt.axhline(y=threshold, color='r', linestyle='-', label=f'Threshold ({threshold})')
                plt.xlabel("Image Index")
                plt.ylabel("Defect Ratio")
                plt.title("Defect Ratio for Each Image")
                plt.legend()
                plt.savefig(os.path.join(output_dir, "defect_ratios.png"))
                plt.close()
        
        logger.info(f"Batch test completed: total number = {len(image_files)}, NC number = {nc_count}, NC ratio = {nc_ratio:.4f}")
        return summary

def apply_defect_mask(original_image, defect_mask, color=(0, 0, 255), alpha=0.5):
    """在原始图像上应用缺陷掩码，突出显示缺陷区域"""
    import numpy as np
    import cv2
    import torch
    from PIL import Image
    
    # 将原始图像转换为NumPy数组
    if isinstance(original_image, torch.Tensor):
        # 如果是张量，转换为NumPy数组并确保通道顺序正确(C,H,W -> H,W,C)
        original_np = original_image.permute(1, 2, 0).cpu().numpy()
    elif isinstance(original_image, Image.Image):
        # 如果是PIL图像，转换为NumPy数组
        original_np = np.array(original_image)
    else:
        original_np = original_image
    
    # 确保值范围在0-1之间
    if original_np.max() <= 1.0:
        original_np = (original_np * 255).astype(np.uint8)
    
    # 创建RGB图像(如果是灰度图)
    if len(original_np.shape) == 2 or original_np.shape[2] == 1:
        if len(original_np.shape) == 3 and original_np.shape[2] == 1:
            original_np = original_np.squeeze(2)
        original_np = cv2.cvtColor(original_np, cv2.COLOR_GRAY2BGR)
    
    # 创建与原图相同大小的掩码图像
    if isinstance(defect_mask, torch.Tensor):
        defect_mask = defect_mask.cpu().numpy()
    
    mask_np = cv2.resize(defect_mask.astype(np.uint8) * 255, 
                         (original_np.shape[1], original_np.shape[0]))
    
    # 创建一个彩色覆盖图
    overlay = original_np.copy()
    overlay[mask_np > 0] = color
    
    # 将覆盖图与原图混合
    highlighted = cv2.addWeighted(overlay, alpha, original_np, 1 - alpha, 0)
    
    return highlighted

def visualize_results(original_img, residual_maps, segmentation_maps, final_segmentation, 
                     is_nc, defect_ratio, output_path=None):
    """可视化检测结果"""
    import matplotlib.pyplot as plt
    import numpy as np
    import torch
    
    # 创建子图布局
    n_levels = len(residual_maps)
    n_rows = 2  # 原始图像和结果在第一行，残差图和分割图在第二行
    n_cols = max(2, n_levels)  # 至少2列，或者根据层级数量
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 5, n_rows * 5))
    
    # 确保axes是二维数组，即使n_cols=1
    if n_cols == 1:
        axes = axes.reshape(n_rows, 1)
    
    # 第一行：原始图像和最终分割结果
    # 显示原始图像
    if isinstance(original_img, torch.Tensor):
        original_np = original_img.squeeze().cpu().numpy()
    else:
        original_np = original_img
        
    axes[0, 0].imshow(original_np, cmap='gray')
    axes[0, 0].set_title("Original Image")
    axes[0, 0].axis('off')
    
    # 显示带有缺陷标记的原始图像
    if isinstance(final_segmentation, torch.Tensor):
        final_np = final_segmentation.squeeze().cpu().numpy()
    else:
        final_np = final_segmentation
        
    # 创建缺陷叠加图
    if len(original_np.shape) == 2:  # 灰度图
        original_rgb = np.stack([original_np, original_np, original_np], axis=2)
    else:  # 已经是RGB
        original_rgb = original_np
        
    marked_image = apply_defect_mask(original_rgb, final_np)
    axes[0, 1].imshow(marked_image)
    axes[0, 1].set_title(f"Defect marking (result: {'unqualified' if is_nc else 'qualified'}, defect ratio: {defect_ratio:.4f})")
    axes[0, 1].axis('off')
    
    # 填充剩余的第一行（如果有）
    for i in range(2, n_cols):
        axes[0, i].axis('off')
    
    # 第二行：每个层级的残差图和分割图
    for i in range(min(n_levels, n_cols)):
        # 获取残差图和分割图
        residual_np = residual_maps[i].squeeze().cpu().numpy()
        segmentation_np = segmentation_maps[i].squeeze().cpu().numpy()
        
        # 绘制残差图
        im = axes[1, i].imshow(residual_np, cmap='jet')
        axes[1, i].set_title(f"Level {i+1} residual plot")
        axes[1, i].axis('off')
        plt.colorbar(im, ax=axes[1, i], fraction=0.046, pad=0.04)
    
    # 填充剩余的第二行（如果有）
    for i in range(n_levels, n_cols):
        axes[1, i].axis('off')
    
    plt.tight_layout()
    
    # 保存结果（如果指定了输出路径）
    if output_path:
        plt.savefig(output_path)
        
    return fig

def detect_with_visualization(model, image_path, output_dir=None):
    """检测图像中的缺陷并提供可视化结果"""
    import os
    import torch
    from PIL import Image
    import matplotlib.pyplot as plt
    import torchvision.transforms as transforms
    import numpy as np
    
    # 数据预处理
    transform = transforms.Compose([
        transforms.Resize((256, 256)),  # 调整为固定大小
        transforms.ToTensor(),  # 转换为tensor
        NormalizeWeber(),  # 光照归一化
    ])
    
    # 读取图像
    img = Image.open(image_path).convert('L')
    img_tensor = transform(img).unsqueeze(0)  # [1, 1, H, W]
    
    # 保存原始图像用于可视化
    original_img = img_tensor.clone()
    
    # 对每个金字塔层级进行缺陷检测
    residual_maps = []
    segmentation_maps = []
    
    for level in range(model.levels):
        # 缩放到当前金字塔层级
        scale_factor = 0.5 ** level
        scaled_size = (int(img_tensor.shape[2] * scale_factor), int(img_tensor.shape[3] * scale_factor))
        scaled_img = resize(img_tensor, scaled_size)
        
        # 创建空的残差图
        residual_map = torch.zeros((1, 1, scaled_img.shape[2], scaled_img.shape[3])).to(model.device)
        count_map = torch.zeros((1, 1, scaled_img.shape[2], scaled_img.shape[3])).to(model.device)
        
        # 滑动窗口提取图像块并计算重建误差
        with torch.no_grad():
            for h in range(0, scaled_img.shape[2] - model.patch_size + 1, model.stride):
                for w in range(0, scaled_img.shape[3] - model.patch_size + 1, model.stride):
                    patch = scaled_img[:, :, h:h+model.patch_size, w:w+model.patch_size].to(model.device)
                    output = model.models[level](patch)
                    
                    # 计算重建残差
                    error = (output - patch) ** 2
                    
                    # 更新残差图和计数图
                    residual_map[:, :, h:h+model.patch_size, w:w+model.patch_size] += error
                    count_map[:, :, h:h+model.patch_size, w:w+model.patch_size] += 1
        
        # 处理计数为0的位置
        count_map[count_map == 0] = 1
        # 计算平均残差
        residual_map = residual_map / count_map
        
        # 调整残差图到原始大小
        residual_map_full = resize(residual_map, (img_tensor.shape[2], img_tensor.shape[3]))
        residual_maps.append(residual_map_full)
        
        # 应用阈值进行缺陷分割
        threshold = model.error_stats[level]['threshold']
        segmentation_map = (residual_map_full > threshold).float()
        segmentation_maps.append(segmentation_map)
    
    # 融合不同层级的分割结果
    final_segmentation = torch.zeros_like(segmentation_maps[0])
    
    # 如果只有一个层级，直接使用该层级的分割图
    if model.levels == 1:
        final_segmentation = segmentation_maps[0]
    else:
        # 首先对相邻层级进行AND运算
        and_results = []
        for i in range(model.levels - 1):
            and_result = segmentation_maps[i] * segmentation_maps[i + 1]  # 逻辑AND
            and_results.append(and_result)
        
        # 然后对AND结果进行OR运算
        for and_result in and_results:
            final_segmentation = torch.max(final_segmentation, and_result)  # 逻辑OR
    
    # 计算统计信息
    defect_ratio = torch.sum(final_segmentation) / (final_segmentation.shape[2] * final_segmentation.shape[3])
    defect_ratio = defect_ratio.item()
    
    # 判断是否为NC产品
    is_nc = defect_ratio > 0.01  # 如果缺陷像素比例>1%，则判断为NC产品
    
    # 可视化结果
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, f"{os.path.basename(image_path).split('.')[0]}_results.png")
        highlighted_path = os.path.join(output_dir, f"{os.path.basename(image_path).split('.')[0]}_highlighted.png")
    else:
        output_path = None
        highlighted_path = None
    
    # 创建可视化图表
    fig = visualize_results(
        original_img.squeeze().cpu().numpy(),
        residual_maps,
        segmentation_maps,
        final_segmentation,
        is_nc,
        defect_ratio,
        output_path
    )
    
    # 保存高亮的缺陷图像
    if highlighted_path:
        original_np = original_img.squeeze().cpu().numpy()
        final_np = final_segmentation.squeeze().cpu().numpy()
        # 确保图像是3通道的
        if len(original_np.shape) == 2:
            original_rgb = np.stack([original_np, original_np, original_np], axis=2)
        else:
            original_rgb = original_np
        
        highlighted = apply_defect_mask(original_rgb, final_np)
        plt.figure(figsize=(10, 10))
        plt.imshow(highlighted)
        plt.title(f"Defect marking (result: {'unqualified' if is_nc else 'qualified'}, defect ratio: {defect_ratio:.4f})")
        plt.axis('off')
        plt.savefig(highlighted_path)
        plt.close()
    
    # 返回检测结果
    result = {
        'is_nc': is_nc,
        'defect_ratio': defect_ratio,
        'segmentation': final_segmentation.cpu().numpy(),
        'residual_maps': [r.cpu().numpy() for r in residual_maps],
        'segmentation_maps': [s.cpu().numpy() for s in segmentation_maps],
        'visualization': fig
    }
    
    return result

def test_batch_images_with_visualization(model, image_dir, output_dir=None, threshold=0.01):
    """批量测试图像并提供可视化结果"""
    import os
    import logging
    import matplotlib.pyplot as plt
    import numpy as np
    from datetime import datetime
    
    logger = logging.getLogger(__name__)
    
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    
    # 获取目录中的所有图像
    image_files = [f for f in os.listdir(image_dir) if f.endswith(('.png', '.jpg', '.jpeg'))]
    
    results = []
    nc_count = 0
    
    # 创建汇总报告的数据
    defect_ratios = []
    image_names = []
    
    for image_file in image_files:
        image_path = os.path.join(image_dir, image_file)
        
        # 为每个图像创建输出子目录
        if output_dir:
            image_output_dir = os.path.join(output_dir, os.path.splitext(image_file)[0])
            os.makedirs(image_output_dir, exist_ok=True)
        else:
            image_output_dir = None
        
        # 检测缺陷并可视化
        result = detect_with_visualization(model, image_path, image_output_dir)
        
        # 统计结果
        if result['is_nc']:
            nc_count += 1
        
        # 收集报告数据
        defect_ratios.append(result['defect_ratio'])
        image_names.append(image_file)
        
        results.append({
            'image': image_file,
            'is_nc': result['is_nc'],
            'defect_ratio': result['defect_ratio']
        })
    
    # 计算NC产品比例
    nc_ratio = nc_count / len(image_files) if image_files else 0
    
    # 保存汇总结果
    summary = {
        'total_images': len(image_files),
        'nc_count': nc_count,
        'nc_ratio': nc_ratio,
        'threshold': threshold,
        'results': results
    }
    
    if output_dir:
        # 保存汇总报告
        with open(os.path.join(output_dir, "summary_report.txt"), "w") as f:
            f.write(f"Test time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Total number of images: {len(image_files)}\n")
            f.write(f"NC product quantity: {nc_count}\n")
            f.write(f"NC product ratio: {nc_ratio:.4f}\n")
            f.write(f"Defect threshold: {threshold}\n\n")
            f.write("Detailed results:\n")

        # 保存缺陷比例汇总图表
        if defect_ratios:
            plt.figure(figsize=(max(12, len(defect_ratios) * 0.5), 6))
            bars = plt.bar(range(len(defect_ratios)), defect_ratios)
            plt.axhline(y=threshold, color='r', linestyle='-', label=f'阈值 ({threshold})')
            
            # 添加颜色标记 - 红色表示不合格，绿色表示合格
            for i, bar in enumerate(bars):
                if defect_ratios[i] > threshold:
                    bar.set_color('r')
                else:
                    bar.set_color('g')
            
            plt.xlabel("Image Index")
            plt.ylabel("Defect ratio")
            plt.title("Defect ratio of each image")
            plt.legend()
            
            # 添加图像名称标签
            if len(image_names) <= 20:  # 如果图像数量不多，显示所有名称
                plt.xticks(range(len(image_names)), image_names, rotation=45)
            else:  # 否则只显示部分名称
                tick_interval = max(1, len(image_names) // 10)
                tick_indices = range(0, len(image_names), tick_interval)
                tick_labels = [image_names[i] for i in tick_indices]
                plt.xticks(tick_indices, tick_labels, rotation=45)
            
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, "defect_ratios_summary.png"))
            plt.close()
    
    logger.info(f"Batch test completed: total number = {len(image_files)}, number of unqualified = {nc_count}, unqualified ratio = {nc_ratio:.4f}")
    return summary

def main():
    # 使用绝对路径
    base_dir = "D:/aiml/defect_detector"

    # 创建目录结构
    # os.makedirs(f"{base_dir}/data/ac_samples", exist_ok=True)
    # os.makedirs(f"{base_dir}/data/test_samples", exist_ok=True)
    # os.makedirs(f"{base_dir}/models", exist_ok=True)
    # os.makedirs(f"{base_dir}/results", exist_ok=True)
    
    # 初始化模型
    model = MSCDAE(levels=3, patch_size=64, stride=32, batch_size=32, epochs=30, learning_rate=0.001, noise_prob=0.05, gamma=3.0)
    
    # 训练模型
    model.train(f"{base_dir}/data/train")
    
    # 保存模型
    model.save_model(f"{base_dir}/mscdae_v2_models")
    
    # 在测试样本上进行检测
    model.test_batch_images(f"{base_dir}/data/test", f"{base_dir}/data/mscdae2_test1_results")
    test_batch_images_with_visualization(model, f"{base_dir}/data/test", output_dir=f"{base_dir}/data/mscdae2_test2_results", threshold=0.01)

if __name__ == "__main__":
    main()