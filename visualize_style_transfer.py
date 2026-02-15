#!/usr/bin/env python3
"""
Visualize style transfer augmentation on CIFAR images.

This script:
1. Loads CIFAR-10/100 dataset
2. Applies style transfer transformation
3. Displays input vs output side-by-side
4. Saves comparison images

Usage:
    python visualize_style_transfer.py --dataset cifar10 --num-images 4
    python visualize_style_transfer.py --dataset cifar100 --num-images 6 --save
"""

import os
import sys
import argparse
import numpy as np
from pathlib import Path
from typing import Tuple, List, Optional

import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torchvision.datasets import CIFAR10, CIFAR100
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

# Add project path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root / "learning" / "solo-learn"))

# Import style transfer
try:
    from solo.data.style_transfer import load_models, load_feat_files, NSTTransform
except ImportError as e:
    print(f"Error importing style transfer: {e}")
    print("Make sure solo-learn is in the path")
    sys.exit(1)


class StyleTransferVisualizer:
    """Visualize style transfer on CIFAR images."""
    
    def __init__(
        self,
        dataset: str = "cifar10",
        num_images: int = 4,
        alpha: float = 1.0,
        probability: float = 1.0,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        output_dir: Optional[str] = None,
    ):
        """
        Initialize visualizer.
        
        Args:
            dataset: "cifar10" or "cifar100"
            num_images: Number of images to visualize
            alpha: Style transfer strength (0=content only, 1=full style)
            probability: Probability of applying augmentation (0=none, 1=always)
            device: torch device
            output_dir: Directory to save visualizations
        """
        self.dataset_name = dataset
        self.num_images = num_images
        self.alpha = alpha
        self.probability = probability
        self.device = torch.device(device)
        self.output_dir = Path(output_dir) if output_dir else Path("style_transfer_outputs")
        self.output_dir.mkdir(exist_ok=True)
        
        print(f"Device: {self.device}")
        print(f"Output directory: {self.output_dir}")
        
        # Load dataset
        self.dataset = self._load_dataset()
        
        # Create style transfer transform
        self.transform_aug, self.transform_no_aug = self._create_transforms()
        self.style_transform = None  # Will be initialized if needed
    
    def _load_dataset(self) -> torch.utils.data.Dataset:
        """Load CIFAR dataset."""
        print(f"\nLoading {self.dataset_name.upper()} dataset...")
        
        # Standard CIFAR transforms (no augmentation, just normalization)
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(
                (0.4914, 0.4822, 0.4465),
                (0.2470, 0.2435, 0.2616)
            )
        ])
        
        if self.dataset_name.lower() == "cifar10":
            dataset = CIFAR10(
                root="./datasets",
                train=True,
                download=True,
                transform=transform
            )
        elif self.dataset_name.lower() == "cifar100":
            dataset = CIFAR100(
                root="./datasets",
                train=True,
                download=True,
                transform=transform
            )
        else:
            raise ValueError(f"Unknown dataset: {self.dataset_name}")
        
        print(f"Loaded {len(dataset)} images from {self.dataset_name.upper()}")
        return dataset
    
    def _create_transforms(self) -> Tuple[transforms.Compose, transforms.Compose]:
        """Create augmentation transforms."""
        # With augmentation (crop, flip, etc.)
        transform_aug = transforms.Compose([
            transforms.RandomResizedCrop(32, scale=(0.08, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(
                (0.4914, 0.4822, 0.4465),
                (0.2470, 0.2435, 0.2616)
            )
        ])
        
        # Without augmentation (just normalize)
        transform_no_aug = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(
                (0.4914, 0.4822, 0.4465),
                (0.2470, 0.2435, 0.2616)
            )
        ])
        
        return transform_aug, transform_no_aug
    
    def _setup_style_transfer(self, use_real_models: bool = False) -> Optional[NSTTransform]:
        """
        Setup style transfer with synthetic or real models.
        
        Args:
            use_real_models: If True, use real VGG+decoder. If False, use mock models.
        
        Returns:
            NSTTransform or None if setup fails
        """
        try:
            if use_real_models:
                print("\nSetting up style transfer with real models...")
                # Try to load real models
                encoder_path = os.path.expandvars(
                    "${HOME}/my_work/code/ssl-aug-benchmark/augmentation/mbda/experiments/adaIN/vgg_normalised.pth"
                )
                decoder_path = os.path.expandvars(
                    "${HOME}/my_work/code/ssl-aug-benchmark/augmentation/mbda/experiments/adaIN/decoder.pth"
                )
                features_path = os.path.expandvars(
                    "${HOME}/my_work/code/ssl-aug-benchmark/augmentation/mbda/features/style_feats_adain_1000.npy"
                )
                
                if not all(os.path.exists(p) for p in [encoder_path, decoder_path, features_path]):
                    print("Real models not found, falling back to synthetic models")
                    use_real_models = False
                else:
                    vgg, decoder = load_models(encoder_path, decoder_path, self.device)
                    style_feats = load_feat_files(features_path, self.device)
            
            if not use_real_models:
                print("\nSetting up style transfer with synthetic models...")
                vgg, decoder = self._create_mock_models()
                style_feats = torch.randn(10, 512, device=self.device)
            
            # Create NSTTransform
            transform = NSTTransform(
                style_feats=style_feats,
                vgg=vgg,
                decoder=decoder,
                alpha_min=self.alpha,
                alpha_max=self.alpha,
                probability=self.probability,
                device=self.device
            )
            
            print(f"Style transfer initialized with {len(style_feats)} styles")
            return transform
        except Exception as e:
            print(f"Error setting up style transfer: {e}")
            return None
    
    def _create_mock_models(self) -> Tuple[nn.Module, nn.Module]:
        """Create mock VGG encoder and decoder for testing."""
        class MockVGG(nn.Module):
            def __init__(self):
                super().__init__()
                self.layers = nn.Sequential(
                    nn.Conv2d(3, 64, 3, padding=1),
                    nn.ReLU(),
                    nn.Conv2d(64, 512, 3, padding=1),
                )
            
            def forward(self, x):
                return self.layers(x)
        
        class MockDecoder(nn.Module):
            def __init__(self):
                super().__init__()
                self.layers = nn.Sequential(
                    nn.Conv2d(512, 256, 3, padding=1),
                    nn.ReLU(),
                    nn.Conv2d(256, 3, 3, padding=1),
                )
            
            def forward(self, x):
                return torch.clamp(self.layers(x), 0, 1)
        
        vgg = MockVGG().to(self.device).eval()
        decoder = MockDecoder().to(self.device).eval()
        
        for param in vgg.parameters():
            param.requires_grad = False
        for param in decoder.parameters():
            param.requires_grad = False
        
        return vgg, decoder
    
    def _denormalize(self, img: torch.Tensor) -> torch.Tensor:
        """Denormalize CIFAR image from normalized to [0, 1] range."""
        denorm = transforms.Compose([
            transforms.Normalize(
                mean=(-0.4914 / 0.2470, -0.4822 / 0.2435, -0.4465 / 0.2616),
                std=(1 / 0.2470, 1 / 0.2435, 1 / 0.2616)
            )
        ])
        return torch.clamp(denorm(img), 0, 1)
    
    def get_sample_images(self) -> List[Tuple[torch.Tensor, int, int]]:
        """
        Get sample images from dataset.
        
        Returns:
            List of (image, label, index) tuples
        """
        # Get random indices
        indices = torch.randperm(len(self.dataset))[:self.num_images]
        
        samples = []
        for idx in indices:
            img, label = self.dataset[int(idx)]
            samples.append((img, label, int(idx)))
        
        return samples
    
    def visualize_comparison(self, use_real_models: bool = False):
        """
        Create side-by-side visualization of input vs output.
        
        Args:
            use_real_models: Use real VGG models if available
        """
        # Setup style transfer
        self.style_transform = self._setup_style_transfer(use_real_models)
        if self.style_transform is None:
            print("Failed to initialize style transfer")
            return
        
        # Get sample images
        samples = self.get_sample_images()
        
        # Create figure with subplots
        fig = plt.figure(figsize=(16, 4 * self.num_images))
        gs = GridSpec(self.num_images, 3, figure=fig, hspace=0.3, wspace=0.2)
        
        class_names = (
            ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]
            if self.dataset_name.lower() == "cifar10"
            else [f"class_{i}" for i in range(100)]
        )
        
        print(f"\nProcessing {self.num_images} images...")
        
        for row, (img_normalized, label, idx) in enumerate(samples):
            # Denormalize image for visualization
            img_original = self._denormalize(img_normalized.clone())
            
            # Apply style transfer
            with torch.no_grad():
                img_stylized = self.style_transform(img_original.to(self.device))
            
            # Detach and move to CPU for plotting
            img_original_np = img_original.cpu().numpy().transpose(1, 2, 0)
            img_stylized_np = img_stylized.cpu().detach().numpy().transpose(1, 2, 0)
            
            # Plot original
            ax1 = fig.add_subplot(gs[row, 0])
            ax1.imshow(img_original_np)
            ax1.set_title(f"Original\n({class_names[label]}, idx={idx})", fontsize=10)
            ax1.axis("off")
            
            # Plot difference map
            diff = np.abs(img_original_np - img_stylized_np)
            ax2 = fig.add_subplot(gs[row, 1])
            ax2.imshow(diff)
            ax2.set_title(f"Difference\n(mean: {diff.mean():.4f})", fontsize=10)
            ax2.axis("off")
            
            # Plot stylized
            ax3 = fig.add_subplot(gs[row, 2])
            ax3.imshow(np.clip(img_stylized_np, 0, 1))
            ax3.set_title(f"Stylized\n(alpha={self.alpha}, prob={self.probability})", fontsize=10)
            ax3.axis("off")
        
        # Save figure
        output_path = self.output_dir / f"style_transfer_{self.dataset_name}_{self.num_images}img.png"
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Saved comparison to: {output_path}")
        
        # Display
        plt.show()
    
    def save_individual_images(self, use_real_models: bool = False):
        """
        Save individual input/output images.
        
        Args:
            use_real_models: Use real VGG models if available
        """
        # Setup style transfer
        self.style_transform = self._setup_style_transfer(use_real_models)
        if self.style_transform is None:
            print("Failed to initialize style transfer")
            return
        
        # Get sample images
        samples = self.get_sample_images()
        
        # Create subdirectories
        input_dir = self.output_dir / "input"
        output_dir_st = self.output_dir / "output"
        input_dir.mkdir(exist_ok=True)
        output_dir_st.mkdir(exist_ok=True)
        
        print(f"\nSaving {self.num_images} image pairs...")
        
        for idx, (img_normalized, label, sample_idx) in enumerate(samples):
            # Denormalize image
            img_original = self._denormalize(img_normalized.clone())
            
            # Apply style transfer
            with torch.no_grad():
                img_stylized = self.style_transform(img_original.to(self.device))
            
            # Convert to PIL and save
            from torchvision.transforms import ToPILImage
            to_pil = ToPILImage()
            
            # Save original
            img_pil_original = to_pil(img_original)
            input_path = input_dir / f"{idx:03d}_original_{label}.png"
            img_pil_original.save(input_path)
            
            # Save stylized
            img_stylized_clipped = torch.clamp(img_stylized, 0, 1)
            img_pil_stylized = to_pil(img_stylized_clipped.cpu().detach())
            output_path = output_dir_st / f"{idx:03d}_stylized_{label}.png"
            img_pil_stylized.save(output_path)
            
            print(f"  Saved pair {idx+1}/{self.num_images}: {input_path.name} → {output_path.name}")
        
        print(f"All images saved to: {self.output_dir}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Visualize style transfer augmentation on CIFAR images"
    )
    
    parser.add_argument(
        "--dataset",
        type=str,
        default="cifar10",
        choices=["cifar10", "cifar100"],
        help="Dataset to use"
    )
    
    parser.add_argument(
        "--num-images",
        type=int,
        default=4,
        help="Number of images to visualize"
    )
    
    parser.add_argument(
        "--alpha",
        type=float,
        default=1.0,
        help="Style transfer strength (0=content, 1=full style)"
    )
    
    parser.add_argument(
        "--probability",
        type=float,
        default=1.0,
        help="Probability of applying augmentation (0=never, 1=always)"
    )
    
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to use (cuda or cpu)"
    )
    
    parser.add_argument(
        "--output-dir",
        type=str,
        default="style_transfer_outputs",
        help="Directory to save visualizations"
    )
    
    parser.add_argument(
        "--save-individual",
        action="store_true",
        help="Save individual input/output images"
    )
    
    parser.add_argument(
        "--use-real-models",
        action="store_true",
        help="Use real VGG models if available"
    )
    
    parser.add_argument(
        "--compare-only",
        action="store_true",
        help="Only create comparison visualization (default)"
    )
    
    args = parser.parse_args()
    
    # Print info
    print("""
╔════════════════════════════════════════════════════════════════╗
║        Style Transfer Visualization on CIFAR Images           ║
╚════════════════════════════════════════════════════════════════╝
    """)
    
    print(f"Configuration:")
    print(f"  Dataset: {args.dataset.upper()}")
    print(f"  Num images: {args.num_images}")
    print(f"  Alpha (strength): {args.alpha}")
    print(f"  Probability: {args.probability}")
    print(f"  Device: {args.device}")
    print(f"  Output dir: {args.output_dir}")
    print(f"  Use real models: {args.use_real_models}")
    
    # Create visualizer
    visualizer = StyleTransferVisualizer(
        dataset=args.dataset,
        num_images=args.num_images,
        alpha=args.alpha,
        probability=args.probability,
        device=args.device,
        output_dir=args.output_dir,
    )
    
    # Run visualization
    if args.save_individual:
        print("\n[1/2] Creating comparison visualization...")
        visualizer.visualize_comparison(use_real_models=args.use_real_models)
        
        print("\n[2/2] Saving individual images...")
        visualizer.save_individual_images(use_real_models=args.use_real_models)
    else:
        print("\nCreating comparison visualization...")
        visualizer.visualize_comparison(use_real_models=args.use_real_models)
    
    print("\n✓ Done!")


if __name__ == "__main__":
    main()
