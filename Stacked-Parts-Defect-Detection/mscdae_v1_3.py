import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
import os
from PIL import Image
import numpy as np
import logging
from pathlib import Path

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class GaussianPyramid(nn.Module):
    def __init__(self, levels=3):
        super(GaussianPyramid, self).__init__()
        self.levels = levels
        
    def forward(self, x):
        # 生成高斯金字塔
        pyramid = [x]
        current = x
        for _ in range(1, self.levels):
            # 使用平均池化模拟高斯下采样
            current = F.avg_pool2d(current, kernel_size=2, stride=2)
            pyramid.append(current)
        return pyramid

class MultiScaleConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(MultiScaleConvBlock, self).__init__()
        # 高斯金字塔
        self.gaussian_pyramid = GaussianPyramid(levels=3)
        
        # 多尺度卷积
        self.conv_layers = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=k, padding=k//2),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True)
            ) for k in [1, 3, 5]
        ])
        
        # 计算输出通道数 (3个卷积核尺寸 * 3个金字塔级别 * out_channels)
        self.output_channels = 3 * 3 * out_channels

    def forward(self, x):
        # 获取高斯金字塔
        pyramid_features = self.gaussian_pyramid(x)
        
        # 存储多尺度特征
        multi_scale_features = []
        
        # 在每个金字塔层级应用卷积
        for level in pyramid_features:
            level_features = [conv(level) for conv in self.conv_layers]
            
            #报错
            # 确保所有特征图尺寸一致
            # if level != pyramid_features[0]:
            #     level_features = [F.interpolate(feat, size=pyramid_features[0].shape[2:]) 
            #                     for feat in level_features]
                
            # 另一种修改方式:
            if level.shape != pyramid_features[0].shape:
                level_features = [F.interpolate(feat, size=pyramid_features[0].shape[2:]) 
                                for feat in level_features]

            multi_scale_features.extend(level_features)
        
        # 特征融合
        return torch.cat(multi_scale_features, dim=1)

class MSCDAE(nn.Module):
    def __init__(self, input_channels=1):
        super(MSCDAE, self).__init__()
        
        # 定义每层的通道数
        self.encoder_channels = [input_channels, 16, 32]
        
        # 编码器
        self.encoder_block1 = MultiScaleConvBlock(self.encoder_channels[0], self.encoder_channels[1])
        self.pool1 = nn.MaxPool2d(2, 2)
        self.encoder_block2 = MultiScaleConvBlock(self.encoder_block1.output_channels, self.encoder_channels[2])
        self.pool2 = nn.MaxPool2d(2, 2)
        
        # 获取编码器最终输出通道数
        self.bottleneck_channels = self.encoder_block2.output_channels
        
        # 解码器
        self.upconv1 = nn.ConvTranspose2d(self.bottleneck_channels, 32, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.decoder_block1 = MultiScaleConvBlock(32, 16)
        self.upconv2 = nn.ConvTranspose2d(self.decoder_block1.output_channels, 16, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.decoder_block2 = MultiScaleConvBlock(16, 8)
        self.final_conv = nn.Conv2d(self.decoder_block2.output_channels, input_channels, kernel_size=1)
        
        # 添加Sigmoid激活保证输出在[0,1]范围
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # 添加噪声 (根据输入强度自适应)
        noise_level = 0.1 * torch.mean(x)
        noise = torch.randn_like(x) * noise_level
        x_noisy = torch.clamp(x + noise, 0, 1)
        
        # 编码
        e1 = self.encoder_block1(x_noisy)
        e1_pool = self.pool1(e1)
        e2 = self.encoder_block2(e1_pool)
        e2_pool = self.pool2(e2)
        
        # 解码
        d1 = self.upconv1(e2_pool)
        d1_block = self.decoder_block1(d1)
        d2 = self.upconv2(d1_block)
        d2_block = self.decoder_block2(d2)
        output = self.final_conv(d2_block)
        
        # 确保输出在[0,1]范围内
        return self.sigmoid(output)

class DefectDataset(Dataset):
    def __init__(self, image_dir, transform=None):
        self.image_dir = Path(image_dir)
        
        # 检查文件夹是否存在
        if not self.image_dir.exists():
            raise FileNotFoundError(f"图像目录 '{image_dir}' 不存在")
        
        # 获取支持的图像文件
        self.images = [f for f in os.listdir(image_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))]
        
        # 检查是否有图像文件
        if len(self.images) == 0:
            raise ValueError(f"图像目录 '{image_dir}' 中没有找到支持的图像文件(.png, .jpg, .jpeg, .bmp)")
        
        self.transform = transform or transforms.Compose([
            transforms.Grayscale(),
            transforms.Resize((256, 256)),
            transforms.ToTensor()
        ])
        
        logger.info(f"已加载 {len(self.images)} 张图像")

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = os.path.join(self.image_dir, self.images[idx])
        try:
            image = Image.open(img_path).convert('RGB')  # 确保图像是RGB格式
            image_tensor = self.transform(image)
            return image_tensor
        except Exception as e:
            logger.error(f"加载图像 '{img_path}' 时出错: {e}")
            # 返回一个空白图像作为替代
            return torch.zeros((1, 256, 256))

def train_mscdae(model, train_loader, criterion, optimizer, device, epochs=50, save_path='checkpoints'):
    # 创建保存检查点的目录
    save_dir = Path(save_path)
    save_dir.mkdir(exist_ok=True, parents=True)
    
    best_loss = float('inf')
    model.train()
    
    for epoch in range(epochs):
        total_loss = 0
        for batch_idx, batch in enumerate(train_loader):
            # 将数据移至设备
            batch = batch.to(device)
            
            optimizer.zero_grad()
            
            # 前向传播
            reconstructed = model(batch)
            
            # 计算损失
            loss = criterion(reconstructed, batch)
            
            # 反向传播
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
            # 打印批次进度
            if (batch_idx + 1) % 10 == 0 or (batch_idx + 1) == len(train_loader):
                logger.info(f'Epoch [{epoch+1}/{epochs}], Batch [{batch_idx+1}/{len(train_loader)}], Loss: {loss.item():.4f}')
        
        avg_loss = total_loss / len(train_loader)
        logger.info(f'Epoch [{epoch+1}/{epochs}], Average Loss: {avg_loss:.4f}')
        
        # 保存最佳模型
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': best_loss,
            }, save_dir / 'best_model.pth')
            logger.info(f'已保存最佳模型, Loss: {best_loss:.4f}')
        
        # 每10个epoch保存一次检查点
        if (epoch + 1) % 10 == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_loss,
            }, save_dir / f'checkpoint_epoch_{epoch+1}.pth')

def detect_defects(model, image, device, threshold_factor=2.0):
    # 确保模型在评估模式
    model.eval()
    
    # 将图像移至设备
    image = image.to(device)
    
    with torch.no_grad():
        # 获取重建图像
        reconstructed = model(image.unsqueeze(0)).squeeze(0)
        
        # 计算重建误差
        error_map = torch.abs(image - reconstructed)
        
        # 计算每个通道的误差统计
        if error_map.dim() > 2:  # 多通道图像
            # 转换为灰度误差图
            error_map = torch.mean(error_map, dim=0)
        
        # 设置自适应阈值
        threshold = error_map.mean() + threshold_factor * error_map.std()
        defect_mask = error_map > threshold
        
        # 返回结果
        return {
            'original': image.cpu(),
            'reconstructed': reconstructed.cpu(),
            'error_map': error_map.cpu(),
            'defect_mask': defect_mask.cpu(),
            'threshold': threshold.item()
        }

def load_model(model_path, model, device):
    # 加载模型检查点
    try:
        checkpoint = torch.load(model_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        logger.info(f"模型已从 '{model_path}' 加载, Epoch: {checkpoint['epoch']}, Loss: {checkpoint['loss']:.4f}")
        return True
    except Exception as e:
        logger.error(f"加载模型失败: {e}")
        return False

def main():
    # 设备选择
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"使用设备: {device}")
    
    # 设置参数，测试集目录
    image_dir = 'data/ac_samples'
    batch_size = 16
    epochs = 50
    learning_rate = 0.001
    
    try:
        # 数据集和数据加载器
        dataset = DefectDataset(image_dir)
        
        # 划分训练集和验证集 (80% 训练, 20% 验证)
        train_size = int(0.8 * len(dataset))
        val_size = len(dataset) - train_size
        train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
        
        logger.info(f"训练集: {train_size} 样本, 验证集: {val_size} 样本")
        
        # 模型初始化
        model = MSCDAE().to(device)
        logger.info(f"初始化模型: {model.__class__.__name__}")
        
        # 损失函数和优化器
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        
        # 训练
        logger.info("开始训练...")
        train_mscdae(model, train_loader, criterion, optimizer, device, epochs=epochs)
        
        # 保存最终模型
        torch.save({
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
        }, 'final_mscdae_model.pth')
        logger.info("训练完成，模型已保存")
        
        # 在验证集上评估
        logger.info("在验证集上评估...")
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(device)
                reconstructed = model(batch)
                loss = criterion(reconstructed, batch)
                val_loss += loss.item()
        
        avg_val_loss = val_loss / len(val_loader)
        logger.info(f"验证集平均损失: {avg_val_loss:.4f}")

    except Exception as e:
        logger.error(f"发生错误: {e}")
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    main()

# 执行
# python mscdae_v1_3.py