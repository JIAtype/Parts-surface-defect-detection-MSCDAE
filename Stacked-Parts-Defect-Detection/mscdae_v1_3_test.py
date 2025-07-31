import torch
import torchvision.transforms as transforms
from PIL import Image
import os
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import logging
import argparse
import cv2
import matplotlib.pyplot as plt

# 导入模型 (确保路径正确，能够导入之前定义的MSCDAE类)
from mscdae_v1_3 import MSCDAE, load_model

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 中文显示问题
plt.rcParams['font.sans-serif'] = ['SimHei']  # 设置字体为 SimHei
plt.rcParams['axes.unicode_minus'] = False    # 解决负号显示问题

def load_and_preprocess_image(image_path, transform=None):
    """加载并预处理单张图像"""
    if transform is None:
        transform = transforms.Compose([
            transforms.Grayscale(),
            transforms.Resize((256, 256)),
            transforms.ToTensor()
        ])
    
    try:
        image = Image.open(image_path).convert('RGB')
        tensor = transform(image)
        return tensor, image
    except Exception as e:
        logger.error(f"Failed to load image '{image_path}' : {e}")
        return None, None

def visualize_results(results, save_path=None):
    """可视化检测结果"""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # 原始图像
    axes[0, 0].imshow(results['original'].permute(1, 2, 0).cpu().numpy())
    axes[0, 0].set_title('Original Image')
    axes[0, 0].axis('off')
    
    # 重建图像
    axes[0, 1].imshow(results['reconstructed'].permute(1, 2, 0).cpu().numpy())
    axes[0, 1].set_title('Reconstructing the image')
    axes[0, 1].axis('off')
    
    # 误差图
    error_map = results['error_map'].cpu().numpy()
    im = axes[1, 0].imshow(error_map, cmap='jet')
    axes[1, 0].set_title(f'Reconstruction error (mean: {error_map.mean():.4f})')
    axes[1, 0].axis('off')
    fig.colorbar(im, ax=axes[1, 0], fraction=0.046, pad=0.04)
    
    # 缺陷掩码
    axes[1, 1].imshow(results['defect_mask'].cpu().numpy(), cmap='gray')
    axes[1, 1].set_title(f'Defect Mask (Threshold: {results["threshold"]:.4f})')
    axes[1, 1].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
        logger.info(f"Results saved to {save_path}")
    
    plt.show()

def apply_defect_mask(original_image, defect_mask, color=(0, 0, 255), alpha=0.5):
    """在原始图像上应用缺陷掩码，突出显示缺陷区域"""
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
        original_np = cv2.cvtColor(original_np, cv2.COLOR_GRAY2BGR)
    
    # 创建与原图相同大小的掩码图像
    mask_np = cv2.resize(defect_mask.cpu().numpy().astype(np.uint8) * 255, 
                         (original_np.shape[1], original_np.shape[0]))
    
    # 创建一个彩色覆盖图
    overlay = original_np.copy()
    overlay[mask_np > 0] = color
    
    # 将覆盖图与原图混合
    highlighted = cv2.addWeighted(overlay, alpha, original_np, 1 - alpha, 0)
    
    return highlighted

def test_single_image(model, image_path, device, threshold_factor=2.0, save_dir=None):
    """测试单张图像并可视化结果"""
    # 加载并预处理图像
    image_tensor, original_image = load_and_preprocess_image(image_path)
    if image_tensor is None:
        return
    
    # 检测缺陷
    results = detect_defects(model, image_tensor, device, threshold_factor)
    
    # 可视化结果
    if save_dir:
        save_path = os.path.join(save_dir, f"{Path(image_path).stem}_results.png")
    else:
        save_path = None
    
    visualize_results(results, save_path)
    
    # 在原始图像上标记缺陷
    highlighted_image = apply_defect_mask(original_image, results['defect_mask'])
    
    if save_dir:
        highlight_path = os.path.join(save_dir, f"{Path(image_path).stem}_highlighted.png")
        cv2.imwrite(highlight_path, cv2.cvtColor(highlighted_image, cv2.COLOR_RGB2BGR))
        logger.info(f"标记的图像已保存至 {highlight_path}")
    
    plt.figure(figsize=(8, 8))
    plt.imshow(highlighted_image)
    plt.title("Marked defects")
    plt.axis('off')
    plt.show()
    
    return results

def detect_defects(model, image, device, threshold_factor=2.0):
    """检测图像中的缺陷"""
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

def test_batch_images(model, image_dir, device, threshold_factor=2.0, save_dir=None):
    """测试文件夹中的所有图像"""
    # 确保保存目录存在
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
    
    # 获取所有图像文件
    image_files = [f for f in os.listdir(image_dir) 
                  if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))]
    
    if not image_files:
        logger.error(f"目录 '{image_dir}' 中没有找到图像文件")
        return
    
    logger.info(f"找到 {len(image_files)} 个图像文件")
    
    # 对每个图像进行测试
    for image_file in image_files:
        image_path = os.path.join(image_dir, image_file)
        logger.info(f"处理图像: {image_path}")
        try:
            test_single_image(model, image_path, device, threshold_factor, save_dir)
        except Exception as e:
            logger.error(f"处理图像 '{image_path}' 时出错: {e}")
            import traceback
            traceback.print_exc()

def main():
    parser = argparse.ArgumentParser(description='测试缺陷检测模型')
    parser.add_argument('--model_path', type=str, default='final_mscdae_model.pth',
                        help='模型权重文件路径')
    parser.add_argument('--data_dir', type=str, default='data',
                        help='包含测试图像的目录')
    parser.add_argument('--single_image', type=str, default=None,
                        help='单张图像的路径（可选）')
    parser.add_argument('--threshold', type=float, default=2.0,
                        help='缺陷检测阈值因子（默认：标准差的2倍）')
    parser.add_argument('--output_dir', type=str, default='results',
                        help='保存结果的目录')
    args = parser.parse_args()
    
    # 选择设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"使用设备: {device}")
    
    # 初始化模型
    model = MSCDAE().to(device)
    
    # 加载模型权重
    if not os.path.exists(args.model_path):
        logger.error(f"模型文件 '{args.model_path}' 不存在")
        return
    
    checkpoint = torch.load(args.model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    logger.info(f"已加载模型 '{args.model_path}'")
    
    # 创建输出目录
    if args.output_dir:
        os.makedirs(args.output_dir, exist_ok=True)

    args.data_dir = 'data/test_samples'

    # 测试模型
    if args.single_image:
        if os.path.exists(args.single_image):
            logger.info(f"测试单张图像: {args.single_image}")
            test_single_image(model, args.single_image, device, args.threshold, args.output_dir)
        else:
            logger.error(f"图像文件 '{args.single_image}' 不存在")
    else:
        if os.path.exists(args.data_dir):
            logger.info(f"测试目录中的所有图像: {args.data_dir}")
            test_batch_images(model, args.data_dir, device, args.threshold, args.output_dir)
        else:
            logger.error(f"数据目录 '{args.data_dir}' 不存在")

if __name__ == '__main__':
    main()

# 模型名
# final_mscdae_model
# 测试数据保存在
# data
# 结果保存在
# results

# 测试整个数据目录中的图像
# python mscdae_v1_3_test.py --model_path final_mscdae_model.pth --data_dir data --output_dir results

# 测试单张图像
# python mscdae_v1_3_test.py --model_path final_mscdae_model.pth --single_image data/test_image.jpg --output_dir results

# 调整缺陷检测阈值
# python mscdae_v1_3_test.py --data_dir data --threshold 1.5  # 降低阈值，检测更多潜在缺陷