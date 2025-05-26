当然可以！下面我将融合`mscdae_v1.py`和`mscdae_v2.py`的优点，设计一个**更强大、更工程化、更易用**的`mscdae_v3.py`。  
主要融合思路如下：

- **多尺度特征提取**：每个金字塔层级采用`MultiScaleConvBlock`（多尺度卷积核），提升特征表达能力。
- **Patch切块+滑窗**：高分辨率层采用patch滑窗，低分辨率层可用整图，兼顾效率与细节。
- **数据增强**：支持椒盐噪声、高斯噪声、韦伯归一化等多种方式。
- **端到端结构**：每层金字塔为一个端到端MSCDAE网络，支持批量训练与推理。
- **批量检测、可视化、统计分析**：保留v2的工程化优势。
- **日志、异常处理、模型保存/加载**：参考v1的健壮性和可维护性。
- **配置参数化**：便于实验和部署。

---

下面是**完整的v3代码**，已尽量精简和模块化，便于理解和扩展。

---

```python
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from torchvision.transforms.functional import resize
from PIL import Image
import logging
import matplotlib.pyplot as plt
from datetime import datetime

# ========== 日志设置 ==========
log_path = "logs/mscdae_v3_train.log"
os.makedirs(os.path.dirname(log_path), exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_path, mode='a'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("MSCDAE_v3")

# ========== 数据增强 ==========
class SaltPepperNoise(object):
    def __init__(self, prob=0.05):
        self.prob = prob
    def __call__(self, img_tensor):
        img_np = img_tensor.numpy()
        noise_mask = np.random.random(img_np.shape) < self.prob
        salt_mask = np.random.random(img_np.shape) < 0.5
        img_np[noise_mask & salt_mask] = 1.0
        img_np[noise_mask & ~salt_mask] = 0.0
        return torch.from_numpy(img_np)

class GaussianNoise(object):
    def __init__(self, std=0.05):
        self.std = std
    def __call__(self, img_tensor):
        noise = torch.randn_like(img_tensor) * self.std
        return torch.clamp(img_tensor + noise, 0, 1)

class NormalizeWeber(object):
    def __init__(self, epsilon=1e-6):
        self.epsilon = epsilon
    def __call__(self, img_tensor):
        mean_intensity = torch.mean(img_tensor) + self.epsilon
        normalized = (img_tensor - mean_intensity) / mean_intensity
        min_val = torch.min(normalized)
        max_val = torch.max(normalized)
        normalized = (normalized - min_val) / (max_val - min_val + self.epsilon)
        return normalized

# ========== 数据集 ==========
class DefectDataset(Dataset):
    def __init__(self, image_dir, transform=None):
        self.image_dir = image_dir
        self.transform = transform
        self.image_files = [f for f in os.listdir(image_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))]
        if len(self.image_files) == 0:
            raise ValueError(f"No images found in {image_dir}")
    def __len__(self):
        return len(self.image_files)
    def __getitem__(self, idx):
        img_path = os.path.join(self.image_dir, self.image_files[idx])
        img = Image.open(img_path).convert('L')
        img_tensor = transforms.ToTensor()(img)
        if self.transform:
            img_tensor = self.transform(img_tensor)
        return img_tensor

# ========== 多尺度卷积块 ==========
class MultiScaleConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv_layers = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=k, padding=k//2),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True)
            ) for k in [1, 3, 5]
        ])
        self.output_channels = 3 * out_channels
    def forward(self, x):
        features = [conv(x) for conv in self.conv_layers]
        return torch.cat(features, dim=1)

# ========== MSCDAE网络 ==========
class MSCDAEBlock(nn.Module):
    def __init__(self, in_channels=1, base_channels=16):
        super().__init__()
        self.encoder1 = MultiScaleConvBlock(in_channels, base_channels)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.encoder2 = MultiScaleConvBlock(self.encoder1.output_channels, base_channels*2)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.bottleneck_channels = self.encoder2.output_channels
        self.upconv1 = nn.ConvTranspose2d(self.bottleneck_channels, base_channels*2, 3, 2, 1, 1)
        self.decoder1 = MultiScaleConvBlock(base_channels*2, base_channels)
        self.upconv2 = nn.ConvTranspose2d(self.decoder1.output_channels, base_channels, 3, 2, 1, 1)
        self.decoder2 = MultiScaleConvBlock(base_channels, base_channels//2)
        self.final_conv = nn.Conv2d(self.decoder2.output_channels, in_channels, 1)
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        e1 = self.encoder1(x)
        e1p = self.pool1(e1)
        e2 = self.encoder2(e1p)
        e2p = self.pool2(e2)
        d1 = self.upconv1(e2p)
        d1b = self.decoder1(d1)
        d2 = self.upconv2(d1b)
        d2b = self.decoder2(d2)
        out = self.final_conv(d2b)
        return self.sigmoid(out)

# ========== MSCDAE多层金字塔系统 ==========
class MSCDAE:
    def __init__(self, levels=3, patch_size=64, stride=32, batch_size=16, epochs=30, lr=0.001, noise_type='sp', noise_param=0.05, gamma=3.0, device=None):
        self.levels = levels
        self.patch_size = patch_size
        self.stride = stride
        self.batch_size = batch_size
        self.epochs = epochs
        self.lr = lr
        self.noise_type = noise_type
        self.noise_param = noise_param
        self.gamma = gamma
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.models = [MSCDAEBlock().to(self.device) for _ in range(levels)]
        self.optimizers = [optim.Adam(m.parameters(), lr=lr) for m in self.models]
        self.criterion = nn.MSELoss()
        self.error_stats = [{'mean': None, 'std': None, 'threshold': None} for _ in range(levels)]
        logger.info(f"MSCDAE v3 initialized: {levels} levels, device={self.device}")

    def get_noise_transform(self):
        if self.noise_type == 'sp':
            return SaltPepperNoise(prob=self.noise_param)
        elif self.noise_type == 'gauss':
            return GaussianNoise(std=self.noise_param)
        else:
            return None

    def train(self, image_dir):
        logger.info(f"Start training: {image_dir}")
        transform = transforms.Compose([
            transforms.Resize((256, 256)),
            NormalizeWeber(),
        ])
        dataset = DefectDataset(image_dir, transform=transform)
        dataloader = DataLoader(dataset, batch_size=1, shuffle=True)
        noise_transform = self.get_noise_transform()
        for level in range(self.levels):
            logger.info(f"Training pyramid level {level+1}/{self.levels}")
            model = self.models[level]
            optimizer = self.optimizers[level]
            model.train()
            all_errors = []
            for epoch in range(self.epochs):
                epoch_loss = 0.0
                batch_count = 0
                for img in dataloader:
                    scale_factor = 0.5 ** level
                    scaled_size = (int(img.shape[2] * scale_factor), int(img.shape[3] * scale_factor))
                    scaled_img = resize(img, scaled_size)
                    patches = []
                    for h in range(0, scaled_img.shape[2] - self.patch_size + 1, self.stride):
                        for w in range(0, scaled_img.shape[3] - self.patch_size + 1, self.stride):
                            patch = scaled_img[:, :, h:h+self.patch_size, w:w+self.patch_size]
                            patches.append(patch)
                    if not patches:
                        continue
                    batch = torch.cat(patches, dim=0).to(self.device)
                    noisy_batch = []
                    for patch in patches:
                        p = patch.squeeze(0) if patch.dim() == 4 else patch
                        noisy_p = noise_transform(p) if noise_transform else p
                        noisy_p = noisy_p.unsqueeze(0) if noisy_p.dim() == 3 else noisy_p
                        noisy_batch.append(noisy_p)
                    noisy_batch = torch.cat(noisy_batch, dim=0).to(self.device)
                    optimizer.zero_grad()
                    outputs = model(noisy_batch)
                    loss = self.criterion(outputs, batch)
                    loss.backward()
                    optimizer.step()
                    with torch.no_grad():
                        for j in range(batch.size(0)):
                            clean_patch = batch[j].unsqueeze(0)
                            noisy_patch = noisy_batch[j].unsqueeze(0)
                            output = model(noisy_patch)
                            error = torch.mean((output - clean_patch) ** 2).item()
                            all_errors.append(error)
                    epoch_loss += loss.item() * batch.size(0)
                    batch_count += batch.size(0)
                avg_loss = epoch_loss / batch_count if batch_count > 0 else 0
                logger.info(f"  Epoch {epoch+1}/{self.epochs}, Loss: {avg_loss:.6f}")
            if all_errors:
                mean_error = np.mean(all_errors)
                std_error = np.std(all_errors)
                threshold = mean_error + self.gamma * std_error
                self.error_stats[level] = {'mean': mean_error, 'std': std_error, 'threshold': threshold}
                logger.info(f"Level {level+1} Stats: Mean={mean_error:.6f}, Std={std_error:.6f}, Threshold={threshold:.6f}")
        logger.info("Training Completed")
        return self

    def save_model(self, save_dir):
        os.makedirs(save_dir, exist_ok=True)
        for level, model in enumerate(self.models):
            torch.save(model.state_dict(), os.path.join(save_dir, f"model_level_{level}.pth"))
        np.save(os.path.join(save_dir, "error_stats.npy"), self.error_stats)
        config = {
            'levels': self.levels,
            'patch_size': self.patch_size,
            'stride': self.stride,
            'gamma': self.gamma,
            'noise_type': self.noise_type,
            'noise_param': self.noise_param
        }
        np.save(os.path.join(save_dir, "config.npy"), config)
        logger.info(f"Model saved to: {save_dir}")

    def load_model(self, save_dir):
        for level in range(self.levels):
            self.models[level].load_state_dict(torch.load(os.path.join(save_dir, f"model_level_{level}.pth"), map_location=self.device))
        self.error_stats = np.load(os.path.join(save_dir, "error_stats.npy"), allow_pickle=True).tolist()
        config = np.load(os.path.join(save_dir, "config.npy"), allow_pickle=True).item()
        self.levels = config['levels']
        self.patch_size = config['patch_size']
        self.stride = config['stride']
        self.gamma = config['gamma']
        self.noise_type = config.get('noise_type', 'sp')
        self.noise_param = config.get('noise_param', 0.05)
        logger.info(f"Model loaded from {save_dir}")
        return self

    def detect(self, image_path, output_dir=None):
        transform = transforms.Compose([
            transforms.Resize((256, 256)),
            NormalizeWeber(),
        ])
        img = Image.open(image_path).convert('L')
        img_tensor = transform(transforms.ToTensor()(img)).unsqueeze(0)
        original_img = img_tensor.clone()
        residual_maps = []
        segmentation_maps = []
        for level in range(self.levels):
            model = self.models[level]
            model.eval()
            scale_factor = 0.5 ** level
            scaled_size = (int(img_tensor.shape[2] * scale_factor), int(img_tensor.shape[3] * scale_factor))
            scaled_img = resize(img_tensor, scaled_size)
            residual_map = torch.zeros((1, 1, scaled_img.shape[2], scaled_img.shape[3])).to(self.device)
            count_map = torch.zeros((1, 1, scaled_img.shape[2], scaled_img.shape[3])).to(self.device)
            with torch.no_grad():
                for h in range(0, scaled_img.shape[2] - self.patch_size + 1, self.stride):
                    for w in range(0, scaled_img.shape[3] - self.patch_size + 1, self.stride):
                        patch = scaled_img[:, :, h:h+self.patch_size, w:w+self.patch_size].to(self.device)
                        output = model(patch)
                        error = (output - patch) ** 2
                        residual_map[:, :, h:h+self.patch_size, w:w+self.patch_size] += error
                        count_map[:, :, h:h+self.patch_size, w:w+self.patch_size] += 1
            count_map[count_map == 0] = 1
            residual_map = residual_map / count_map
            residual_map_full = resize(residual_map, (img_tensor.shape[2], img_tensor.shape[3]))
            residual_maps.append(residual_map_full)
            threshold = self.error_stats[level]['threshold']
            segmentation_map = (residual_map_full > threshold).float()
            segmentation_maps.append(segmentation_map)
        final_segmentation = torch.zeros_like(segmentation_maps[0])
        if self.levels == 1:
            final_segmentation = segmentation_maps[0]
        else:
            and_results = []
            for i in range(self.levels - 1):
                and_result = segmentation_maps[i] * segmentation_maps[i + 1]
                and_results.append(and_result)
            for and_result in and_results:
                final_segmentation = torch.max(final_segmentation, and_result)
        defect_ratio = torch.sum(final_segmentation) / (final_segmentation.shape[2] * final_segmentation.shape[3])
        defect_ratio = defect_ratio.item()
        is_nc = defect_ratio > 0.01
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            self.visualize_results(original_img, residual_maps, segmentation_maps, final_segmentation, is_nc, defect_ratio, output_dir)
        logger.info(f"检测结果: {'不合格' if is_nc else '合格'}, 缺陷比例: {defect_ratio:.6f}")
        return {'is_nc': is_nc, 'defect_ratio': defect_ratio, 'segmentation': final_segmentation.cpu().numpy()}

    @staticmethod
    def apply_defect_mask(original_image, defect_mask, color=(1, 0, 0), alpha=0.5):
        import numpy as np
        if isinstance(original_image, torch.Tensor):
            original_np = original_image.squeeze().cpu().numpy()
        else:
            original_np = original_image
        if len(original_np.shape) == 2:
            original_rgb = np.stack([original_np, original_np, original_np], axis=2)
        else:
            original_rgb = original_np
        mask = defect_mask.squeeze()
        overlay = original_rgb.copy()
        overlay[mask > 0, 0] = color[0]
        overlay[mask > 0, 1] = color[1]
        overlay[mask > 0, 2] = color[2]
        highlighted = alpha * overlay + (1 - alpha) * original_rgb
        highlighted = np.clip(highlighted, 0, 1)
        return highlighted

    def visualize_results(self, original_img, residual_maps, segmentation_maps, final_segmentation, is_nc, defect_ratio, output_dir):
        n_levels = len(residual_maps)
        plt.figure(figsize=(5 * (n_levels + 2), 10))
        plt.subplot(2, n_levels + 2, 1)
        plt.imshow(original_img.squeeze().cpu().numpy(), cmap='gray')
        plt.title("原始图像")
        plt.axis('off')
        plt.subplot(2, n_levels + 2, 2)
        highlighted = self.apply_defect_mask(original_img, final_segmentation)
        plt.imshow(highlighted)
        plt.title(f"缺陷标记（{'不合格' if is_nc else '合格'}，比例{defect_ratio:.4f}）")
        plt.axis('off')
        for i in range(n_levels):
            plt.subplot(2, n_levels + 2, 3 + i)
            plt.imshow(residual_maps[i].squeeze().cpu().numpy(), cmap='jet')
            plt.title(f"残差图 Level {i+1}")
            plt.axis('off')
            plt.subplot(2, n_levels + 2, 3 + n_levels + i)
            plt.imshow(segmentation_maps[i].squeeze().cpu().numpy(), cmap='gray')
            plt.title(f"分割图 Level {i+1}")
            plt.axis('off')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "detection_results.png"))
        plt.close()

    def test_batch_images(self, image_dir, output_dir=None, threshold=0.01):
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        image_files = [f for f in os.listdir(image_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))]
        results = []
        nc_count = 0
        defect_ratios = []
        image_names = []
        for image_file in image_files:
            image_path = os.path.join(image_dir, image_file)
            image_output_dir = os.path.join(output_dir, os.path.splitext(image_file)[0]) if output_dir else None
            result = self.detect(image_path, image_output_dir)
            if result['is_nc']:
                nc_count += 1
            defect_ratios.append(result['defect_ratio'])
            image_names.append(image_file)
            results.append({'image': image_file, 'is_nc': result['is_nc'], 'defect_ratio': result['defect_ratio']})
        nc_ratio = nc_count / len(image_files) if image_files else 0
        summary = {
            'total_images': len(image_files),
            'nc_count': nc_count,
            'nc_ratio': nc_ratio,
            'threshold': threshold,
            'results': results
        }
        if output_dir:
            with open(os.path.join(output_dir, "summary_report.txt"), "w") as f:
                f.write(f"测试时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"总图片数: {len(image_files)}\n")
                f.write(f"不合格数: {nc_count}\n")
                f.write(f"不合格比例: {nc_ratio:.4f}\n")
                f.write(f"缺陷阈值: {threshold}\n\n")
                f.write("详细结果:\n")
                for result in results:
                    f.write(f"图片: {result['image']}, 结果: {'不合格' if result['is_nc'] else '合格'}, 缺陷比例: {result['defect_ratio']:.6f}\n")
            if defect_ratios:
                plt.figure(figsize=(max(12, len(defect_ratios) * 0.5), 6))
                bars = plt.bar(range(len(defect_ratios)), defect_ratios)
                plt.axhline(y=threshold, color='r', linestyle='-', label=f'阈值 ({threshold})')
                for i, bar in enumerate(bars):
                    bar.set_color('r' if defect_ratios[i] > threshold else 'g')
                plt.xlabel("图片序号")
                plt.ylabel("缺陷比例")
                plt.title("每张图片缺陷比例")
                if len(image_names) <= 20:
                    plt.xticks(range(len(image_names)), image_names, rotation=45)
                else:
                    tick_interval = max(1, len(image_names) // 10)
                    tick_indices = range(0, len(image_names), tick_interval)
                    tick_labels = [image_names[i] for i in tick_indices]
                    plt.xticks(tick_indices, tick_labels, rotation=45)
                plt.tight_layout()
                plt.savefig(os.path.join(output_dir, "defect_ratios_summary.png"))
                plt.close()
        logger.info(f"批量检测完成: 总数={len(image_files)}, 不合格数={nc_count}, 不合格比例={nc_ratio:.4f}")
        return summary

# ========== 主程序入口 ==========
def main():
    # 配置参数
    train_dir = "data/train"
    test_dir = "data/test"
    save_dir = "mscdae_v3_models"
    result_dir = "mscdae_v3_results"
    # 初始化模型
    model = MSCDAE(levels=3, patch_size=64, stride=32, batch_size=16, epochs=30, lr=0.001, noise_type='sp', noise_param=0.05, gamma=3.0)
    # 训练
    model.train(train_dir)
    # 保存
    model.save_model(save_dir)
    # 加载（可选）
    # model.load_model(save_dir)
    # 批量检测
    model.test_batch_images(test_dir, output_dir=result_dir, threshold=0.01)

if __name__ == "__main__":
    main()
```

---

## 说明

- **多尺度特征融合**：每层金字塔用`MultiScaleConvBlock`，每个block用1/3/5卷积核，特征拼接。
- **数据增强**：支持椒盐噪声、高斯噪声，韦伯归一化。
- **patch滑窗**：高分辨率层用patch滑窗训练/推理，提升细节检测能力。
- **端到端结构**：每层金字塔为一个端到端MSCDAE网络。
- **批量检测、可视化、统计**：支持批量检测、自动生成可视化和统计报告。
- **日志与异常**：全流程日志，便于追踪和调试。
- **参数化**：所有关键参数均可配置，便于实验和部署。

---

如需进一步定制（如多GPU、混合精度、Web界面等），可随时告知！
