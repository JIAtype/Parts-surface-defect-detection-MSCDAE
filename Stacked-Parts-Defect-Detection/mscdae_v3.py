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

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("mscdae_model.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class SaltPepperNoise(object):
    """Add salt and pepper noise to images"""
    def __init__(self, prob=0.05):
        self.prob = prob
        
    def __call__(self, img_tensor):
        # Convert to numpy array
        img_np = img_tensor.numpy()
        # Add salt and pepper noise
        noise_mask = np.random.random(img_np.shape) < self.prob
        salt_mask = np.random.random(img_np.shape) < 0.5
        # Add white pixels (salt)
        img_np[noise_mask & salt_mask] = 1.0
        # Add black pixels (pepper)
        img_np[noise_mask & ~salt_mask] = 0.0
        # Convert back to tensor
        return torch.from_numpy(img_np)

class GaussianPyramid(object):
    """Generate Gaussian pyramid for an image"""
    def __init__(self, levels=3):
        self.levels = levels
        
    def __call__(self, img_tensor):
        # Convert to numpy array
        img_np = img_tensor.numpy()[0]  # Take first channel, assuming grayscale
        pyramid = [img_np]
        
        for i in range(1, self.levels):
            img_np = cv2.pyrDown(img_np)
            pyramid.append(img_np)
            
        # Convert back to tensor
        pyramid_tensors = [torch.from_numpy(img).unsqueeze(0) for img in pyramid]
        return pyramid_tensors

class PatchExtractor(object):
    """Extract fixed size patches from an image"""
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
    """Dataset for defect detection"""
    def __init__(self, image_dir, transform=None, patch_size=64, stride=32):
        self.image_dir = image_dir
        self.transform = transform
        self.patch_extractor = PatchExtractor(patch_size, stride)
        self.image_files = [f for f in os.listdir(image_dir) if f.endswith(('.png', '.jpg', '.jpeg'))]
        
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        img_path = os.path.join(self.image_dir, self.image_files[idx])
        # Read image and convert to grayscale
        img = Image.open(img_path).convert('L')
        # Convert to tensor
        img_tensor = transforms.ToTensor()(img)
        
        if self.transform:
            img_tensor = self.transform(img_tensor)
            
        return img_tensor

class ConvolutionalDenoisingAutoencoder(nn.Module):
    """Convolutional Denoising Autoencoder model"""
    def __init__(self, input_channels=1):
        super(ConvolutionalDenoisingAutoencoder, self).__init__()
        
        # Encoder
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
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(True),
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(True),
            nn.Conv2d(32, input_channels, kernel_size=3, stride=1, padding=1),
            nn.Sigmoid()  # Output range [0,1]
        )
        
    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

class MSCDAE:
    """Multi-Scale Convolutional Denoising Autoencoder"""
    def __init__(self, levels=3, patch_size=64, stride=32, batch_size=32, epochs=50, learning_rate=0.001,
                 noise_prob=0.05, gamma=3.0, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.levels = levels
        self.patch_size = patch_size
        self.stride = stride
        self.batch_size = batch_size
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.noise_prob = noise_prob
        self.gamma = gamma  # Threshold parameter
        self.device = device
        
        # Create a CDAE model for each pyramid level
        self.models = [ConvolutionalDenoisingAutoencoder().to(self.device) for _ in range(levels)]
        self.optimizers = [optim.Adam(model.parameters(), lr=learning_rate) for model in self.models]
        self.criterion = nn.MSELoss()
        
        # Store reconstruction error statistics for each level
        self.error_stats = [{'mean': None, 'std': None, 'threshold': None} for _ in range(levels)]
        
        logger.info(f"Initialized MSCDAE model: levels={levels}, device={device}")
        
    def train(self, image_dir):
        """Train the model"""
        logger.info(f"Starting training: image directory={image_dir}")
        
        # Data preprocessing and transformations
        transform = transforms.Compose([
            transforms.Resize((256, 256)),  # Resize to fixed size
            # Removed Weber normalization
        ])
        
        noise_transform = transforms.Compose([
            SaltPepperNoise(prob=self.noise_prob)  # Add salt and pepper noise
        ])
        
        # Create dataset
        dataset = DefectDataset(image_dir, transform=transform, patch_size=self.patch_size, stride=self.stride)
        dataloader = DataLoader(dataset, batch_size=1, shuffle=True)  # Load one image at a time
        
        # Train a CDAE model for each pyramid level
        for level in range(self.levels):
            logger.info(f"Training pyramid level {level + 1}/{self.levels}")
            model = self.models[level]
            optimizer = self.optimizers[level]
            model.train()
            
            all_errors = []  # Collect all reconstruction errors
            
            for epoch in range(self.epochs):
                epoch_loss = 0.0
                batch_count = 0
                
                for i, img in enumerate(dataloader):
                    # Scale to current pyramid level
                    scale_factor = 0.5 ** level
                    scaled_size = (int(img.shape[2] * scale_factor), int(img.shape[3] * scale_factor))
                    scaled_img = resize(img, scaled_size)
                    
                    # Extract patches
                    patches = []
                    for h in range(0, scaled_img.shape[2] - self.patch_size + 1, self.stride):
                        for w in range(0, scaled_img.shape[3] - self.patch_size + 1, self.stride):
                            patch = scaled_img[:, :, h:h+self.patch_size, w:w+self.patch_size]
                            patches.append(patch)
                    
                    if not patches:
                        continue
                    
                    # Combine patches into a batch
                    batch = torch.cat(patches, dim=0).to(self.device)
                    
                    # Add noise
                    noisy_batch = []
                    for patch in patches:
                        # Ensure patch dimensions are correct [1, C, H, W]
                        if patch.dim() == 4:
                            p = patch.squeeze(0)  # Change to [C, H, W]
                        else:
                            p = patch
                        
                        # Add noise
                        noisy_p = noise_transform(p)
                        
                        # Ensure correct dimensions
                        if noisy_p.dim() == 3:
                            noisy_p = noisy_p.unsqueeze(0)  # Change back to [1, C, H, W]
                        
                        noisy_batch.append(noisy_p)
                    
                    noisy_batch = torch.cat(noisy_batch, dim=0).to(self.device)

                    # Training step
                    optimizer.zero_grad()
                    outputs = model(noisy_batch)
                    loss = self.criterion(outputs, batch)
                    loss.backward()
                    optimizer.step()
                    
                    # Collect reconstruction errors for statistics
                    with torch.no_grad():
                        for j in range(batch.size(0)):
                            clean_patch = batch[j].unsqueeze(0)
                            noisy_patch = noisy_batch[j].unsqueeze(0)
                            output = model(noisy_patch)
                            error = torch.mean((output - clean_patch) ** 2).item()
                            all_errors.append(error)
                    
                    epoch_loss += loss.item() * batch.size(0)
                    batch_count += batch.size(0)
                
                # Output training loss for each epoch
                avg_loss = epoch_loss / batch_count if batch_count > 0 else 0
                logger.info(f"  Epoch {epoch+1}/{self.epochs}, Loss: {avg_loss:.6f}")
            
            # Calculate reconstruction error statistics
            if all_errors:
                mean_error = np.mean(all_errors)
                std_error = np.std(all_errors)
                threshold = mean_error + self.gamma * std_error
                
                self.error_stats[level] = {
                    'mean': mean_error,
                    'std': std_error,
                    'threshold': threshold
                }
                
                logger.info(f"Level {level + 1} statistics: Mean={mean_error:.6f}, Std={std_error:.6f}, Threshold={threshold:.6f}")
        
        logger.info("Training completed")
        return self
    
    def save_model(self, save_dir):
        """Save model and statistics"""
        os.makedirs(save_dir, exist_ok=True)
        
        # Save model parameters
        for level, model in enumerate(self.models):
            torch.save(model.state_dict(), os.path.join(save_dir, f"model_level_{level}.pth"))
        
        # Save reconstruction error statistics
        np.save(os.path.join(save_dir, "error_stats.npy"), self.error_stats)
        
        # Save configuration
        config = {
            'levels': self.levels,
            'patch_size': self.patch_size,
            'stride': self.stride,
            'gamma': self.gamma
        }
        np.save(os.path.join(save_dir, "config.npy"), config)
        
        logger.info(f"Model saved to: {save_dir}")
    
    def load_model(self, save_dir):
        """Load saved model and statistics"""
        # Load model parameters
        for level in range(self.levels):
            self.models[level].load_state_dict(torch.load(os.path.join(save_dir, f"model_level_{level}.pth"), map_location=self.device))
        
        # Load reconstruction error statistics
        self.error_stats = np.load(os.path.join(save_dir, "error_stats.npy"), allow_pickle=True).tolist()
        
        # Load configuration
        config = np.load(os.path.join(save_dir, "config.npy"), allow_pickle=True).item()
        self.levels = config['levels']
        self.patch_size = config['patch_size']
        self.stride = config['stride']
        self.gamma = config['gamma']
        
        logger.info(f"Model loaded from {save_dir}")
        return self
    
    def detect(self, image_path, output_dir=None):
        """Detect defects in an image"""
        logger.info(f"Detecting defects in image: {image_path}")
        
        # Data preprocessing
        transform = transforms.Compose([
            transforms.Resize((256, 256)),  # Resize to fixed size
            # Removed Weber normalization
        ])
        
        # Read image
        img = Image.open(image_path).convert('L')
        img_tensor = transform(transforms.ToTensor()(img)).unsqueeze(0)  # [1, 1, H, W]
        
        # Save original image for visualization
        original_img = img_tensor.clone()
        
        # Defect detection for each pyramid level
        residual_maps = []
        segmentation_maps = []
        
        for level in range(self.levels):
            logger.info(f"Processing pyramid level {level + 1}/{self.levels}")
            model = self.models[level]
            model.eval()
            
            # Scale to current pyramid level
            scale_factor = 0.5 ** level
            scaled_size = (int(img_tensor.shape[2] * scale_factor), int(img_tensor.shape[3] * scale_factor))
            scaled_img = resize(img_tensor, scaled_size)
            
            # Create empty residual map
            residual_map = torch.zeros((1, 1, scaled_img.shape[2], scaled_img.shape[3])).to(self.device)
            count_map = torch.zeros((1, 1, scaled_img.shape[2], scaled_img.shape[3])).to(self.device)
            
            # Sliding window to extract patches and calculate reconstruction errors
            with torch.no_grad():
                for h in range(0, scaled_img.shape[2] - self.patch_size + 1, self.stride):
                    for w in range(0, scaled_img.shape[3] - self.patch_size + 1, self.stride):
                        patch = scaled_img[:, :, h:h+self.patch_size, w:w+self.patch_size].to(self.device)
                        output = model(patch)
                        
                        # Calculate reconstruction residual
                        error = (output - patch) ** 2
                        
                        # Update residual map and count map
                        residual_map[:, :, h:h+self.patch_size, w:w+self.patch_size] += error
                        count_map[:, :, h:h+self.patch_size, w:w+self.patch_size] += 1
            
            # Process positions with count 0
            count_map[count_map == 0] = 1
            # Calculate average residual
            residual_map = residual_map / count_map
            
            # Resize residual map to original size
            residual_map_full = resize(residual_map, (img_tensor.shape[2], img_tensor.shape[3]))
            residual_maps.append(residual_map_full)
            
            # Apply threshold for defect segmentation
            threshold = self.error_stats[level]['threshold']
            segmentation_map = (residual_map_full > threshold).float()
            segmentation_maps.append(segmentation_map)
        
        # Fusion of segmentation results from different levels
        # Using logical operations: AND for adjacent levels, OR for final result
        final_segmentation = torch.zeros_like(segmentation_maps[0])
        
        # If only one level, directly use that level's segmentation map
        if self.levels == 1:
            final_segmentation = segmentation_maps[0]
        else:
            # First perform AND operation on adjacent levels
            and_results = []
            for i in range(self.levels - 1):
                and_result = segmentation_maps[i] * segmentation_maps[i + 1]  # Logical AND
                and_results.append(and_result)
            
            # Then perform OR operation on AND results
            for and_result in and_results:
                final_segmentation = torch.max(final_segmentation, and_result)  # Logical OR
        
        # Calculate statistics
        defect_ratio = torch.sum(final_segmentation) / (final_segmentation.shape[2] * final_segmentation.shape[3])
        defect_ratio = defect_ratio.item()
        
        # Determine if NC product
        is_nc = defect_ratio > 0.01  # If defect pixel ratio > 1%, classify as NC product
        
        result = {
            'is_nc': is_nc,
            'defect_ratio': defect_ratio,
            'segmentation': final_segmentation.cpu().numpy()
        }
        
        # If output directory specified, save visualization results
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            
            # Save original image
            original_np = original_img.squeeze().cpu().numpy()
            plt.figure(figsize=(10, 10))
            plt.imshow(original_np, cmap='gray')
            plt.title("Original Image")
            plt.savefig(os.path.join(output_dir, "original.png"))
            plt.close()
            
            # Save residual maps
            for i, residual_map in enumerate(residual_maps):
                residual_np = residual_map.squeeze().cpu().numpy()
                plt.figure(figsize=(10, 10))
                plt.imshow(residual_np, cmap='jet')
                plt.colorbar()
                plt.title(f"Residual Map - Level {i+1}")
                plt.savefig(os.path.join(output_dir, f"residual_level_{i+1}.png"))
                plt.close()
            
            # Save segmentation maps
            for i, segmentation_map in enumerate(segmentation_maps):
                segmentation_np = segmentation_map.squeeze().cpu().numpy()
                plt.figure(figsize=(10, 10))
                plt.imshow(segmentation_np, cmap='gray')
                plt.title(f"Segmentation - Level {i+1}")
                plt.savefig(os.path.join(output_dir, f"segmentation_level_{i+1}.png"))
                plt.close()
            
            # Save final segmentation result
            final_np = final_segmentation.squeeze().cpu().numpy()
            plt.figure(figsize=(10, 10))
            plt.imshow(final_np, cmap='gray')
            plt.title(f"Final Segmentation (NC: {is_nc}, Defect Ratio: {defect_ratio:.6f})")
            plt.savefig(os.path.join(output_dir, "final_segmentation.png"))
            plt.close()
            
            # Mark defect areas on original image
            original_rgb = np.stack([original_np, original_np, original_np], axis=2)
            overlay = original_rgb.copy()
            overlay[final_np > 0, 0] = 1.0  # Red marking for defects
            overlay[final_np > 0, 1:] = 0.0
            
            plt.figure(figsize=(10, 10))
            plt.imshow(overlay)
            plt.title(f"Defect Overlay (NC: {is_nc}, Defect Ratio: {defect_ratio:.6f})")
            plt.savefig(os.path.join(output_dir, "defect_overlay.png"))
            plt.close()
        
        logger.info(f"Detection result: {'NC' if is_nc else 'AC'}, Defect ratio: {defect_ratio:.6f}")
        return result
    
    def test_batch_images(self, image_dir, output_dir=None, threshold=0.01):
        """Batch test images and summarize results"""
        logger.info(f"Batch testing images: {image_dir}")
        
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        
        # Get all images in directory
        image_files = [f for f in os.listdir(image_dir) if f.endswith(('.png', '.jpg', '.jpeg'))]
        
        results = []
        nc_count = 0
        
        for image_file in image_files:
            image_path = os.path.join(image_dir, image_file)
            
            # Create output subdirectory for each image
            if output_dir:
                image_output_dir = os.path.join(output_dir, os.path.splitext(image_file)[0])
                os.makedirs(image_output_dir, exist_ok=True)
            else:
                image_output_dir = None
            
            # Detect defects
            result = self.detect(image_path, image_output_dir)
            
            # Summarize results
            if result['is_nc']:
                nc_count += 1
            
            results.append({
                'image': image_file,
                'is_nc': result['is_nc'],
                'defect_ratio': result['defect_ratio']
            })
        
        # Calculate NC product ratio
        nc_ratio = nc_count / len(image_files) if image_files else 0
        
        # Save summary results
        summary = {
            'total_images': len(image_files),
            'nc_count': nc_count,
            'nc_ratio': nc_ratio,
            'threshold': threshold,
            'results': results
        }
        
        if output_dir:
            # Save summary report
            with open(os.path.join(output_dir, "summary_report.txt"), "w") as f:
                f.write(f"Test time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"Total images: {len(image_files)}\n")
                f.write(f"NC product count: {nc_count}\n")
                f.write(f"NC product ratio: {nc_ratio:.4f}\n")
                f.write(f"Defect threshold: {threshold}\n\n")
                f.write("Detailed results:\n")
                for result in results:
                    f.write(f"Image: {result['image']}, Result: {'NC' if result['is_nc'] else 'AC'}, Defect ratio: {result['defect_ratio']:.6f}\n")
            
            # Save summary chart
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
        
        logger.info(f"Batch testing completed: Total={len(image_files)}, NC count={nc_count}, NC ratio={nc_ratio:.4f}")
        return summary

# Usage example
def run_demo():
    # Use absolute path
    base_dir = "D:/aiml/cv"

    # Create directory structure
    os.makedirs(f"{base_dir}/data/ac_samples", exist_ok=True)
    os.makedirs(f"{base_dir}/data/test_samples", exist_ok=True)
    os.makedirs(f"{base_dir}/models", exist_ok=True)
    os.makedirs(f"{base_dir}/results", exist_ok=True)
    
    # Initialize model
    model = MSCDAE(levels=3, patch_size=64, stride=32, batch_size=32, epochs=30, learning_rate=0.001, noise_prob=0.05, gamma=3.0)
    
    # Train model
    model.train(f"{base_dir}/data/ac_samples")
    
    # Save model
    model.save_model(f"{base_dir}/models/mscdae_model")
    
    # Detect on test samples
    model.test_batch_images(f"{base_dir}/data/test_samples", f"{base_dir}/results/test_results")

if __name__ == "__main__":
    # Run demo example
    run_demo()


# Complete MSCDAE (Multi-Scale Convolutional Denoising Autoencoder) unsupervised learning model for metal surface defect detection.

# Unsupervised learning method: Model is trained only on defect-free (AC) samples, no labeled defect samples needed
# Multi-scale analysis: Uses Gaussian pyramid to extract image features at different resolutions
# Convolutional Denoising Autoencoder: Uses powerful neural network structure to learn normal patterns from defect-free samples

# Core features:

# Image preprocessing:
# - Salt and pepper noise addition, enhancing model robustness
# - Multi-scale analysis with Gaussian pyramid

# Model training:
# - Independent training of CDAE model for each pyramid level
# - Automatic calculation of reconstruction error statistics and optimal thresholds for each level

# Defect detection:
# - Defect localization based on reconstruction residuals
# - Multi-scale segmentation result fusion (AND and OR logical operations)
# - Defect ratio calculation and NC/AC determination

# Complete functionality:
# - Model saving and loading
# - Detailed analysis of single images
# - Batch testing and summary reporting
# - Visualization output