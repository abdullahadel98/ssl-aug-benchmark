# Copyright 2023 solo-learn development team.

# Permission is hereby granted, free of charge, to any person obtaining a copy of
# this software and associated documentation files (the "Software"), to deal in
# the Software without restriction, including without limitation the rights to use,
# copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the
# Software, and to permit persons to whom the Software is furnished to do so,
# subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all copies
# or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR
# PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE
# FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR
# OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.

"""
MVTec-AD dataset loader.

MVTec-AD has a specific hierarchical structure:
    mvtec/
    ├── bottle/
    │   ├── train/
    │   │   └── good/           <- Normal training images
    │   ├── test/
    │   │   ├── good/           <- Normal test images
    │   │   ├── [anomaly_type]/ <- Anomalous test images
    │   │   └── ...
    │   └── ground_truth/
    ├── cable/
    ├── ... (15 categories total)

For SSL pretraining, we use only the 'train/good/' images from each category,
treating each category as a separate class.
"""

import os
from pathlib import Path
from typing import Optional, Tuple, Union

from torch.utils.data import Dataset
from torchvision.datasets import ImageFolder


def create_mvtec_train_dataset(
    root_path: Union[str, Path],
    transform=None,
    split: str = "train",
) -> ImageFolder:
    """
    Creates an MVTec-AD dataset loader that aggregates all categories.
    
    MVTec-AD structure uses category/train/good/ for normal images.
    This function creates a flat ImageFolder structure by treating each category
    as a class label and aggregating all 'good' images from train or test splits.
    
    Args:
        root_path: Path to the mvtec root directory (containing bottle/, cable/, etc.)
        transform: Optional image transformations
        split: Either "train" (uses train/good/) or "test" (uses test/good/)
        
    Returns:
        ImageFolder dataset with aggregated MVTec-AD images
    """
    
    root_path = Path(root_path)
    
    # Create a temporary structured directory for ImageFolder
    # This is the cleanest way without modifying torchvision's ImageFolder
    # Alternative: we could subclass ImageFolder, but this is simpler
    
    # MVTec categories (15 object types)
    MVTEC_CATEGORIES = [
        "bottle", "cable", "capsule", "carpet", "grid", "hazelnut",
        "leather", "metal_nut", "pill", "screw", "tile", "toothbrush",
        "transistor", "wood", "zipper"
    ]
    
    # Verify all categories exist
    for category in MVTEC_CATEGORIES:
        cat_path = root_path / category / split / "good"
        if not cat_path.exists():
            raise FileNotFoundError(
                f"MVTec-AD category not found: {cat_path}\n"
                f"Expected structure: mvtec/{{category}}/{split}/good/\n"
                f"Verify you're pointing to the correct MVTec root directory."
            )
    
    # Create ImageFolder with a custom structure
    # We'll use a wrapper that handles the MVTec structure internally
    return MVTecImageFolder(root_path, transform=transform, split=split)


class MVTecImageFolder(ImageFolder):
    """
    Custom ImageFolder for MVTec-AD that handles the hierarchical structure.
    
    Maps MVTec categories to class indices:
    - Class 0: bottle
    - Class 1: cable
    - ...
    - Class 14: zipper
    """
    
    MVTEC_CATEGORIES = [
        "bottle", "cable", "capsule", "carpet", "grid", "hazelnut",
        "leather", "metal_nut", "pill", "screw", "tile", "toothbrush",
        "transistor", "wood", "zipper"
    ]
    
    def __init__(
        self,
        root: Union[str, Path],
        transform=None,
        split: str = "train",
    ):
        """
        Initialize MVTec-AD dataset.
        
        Args:
            root: Path to mvtec root (containing category subdirectories)
            transform: Image transformations
            split: "train" or "test" - which split to use
        """
        self.root = Path(root)
        self.split = split
        self.transform = transform
        
        # Build image list and class mapping
        self.samples = []
        self.class_to_idx = {cat: idx for idx, cat in enumerate(self.MVTEC_CATEGORIES)}
        self.classes = self.MVTEC_CATEGORIES
        self.imgs = []  # Required by ImageFolder
        
        # Aggregate all images from all categories
        for category_idx, category in enumerate(self.MVTEC_CATEGORIES):
            category_path = self.root / category / split / "good"
            
            if not category_path.exists():
                raise FileNotFoundError(
                    f"Category path not found: {category_path}\n"
                    f"Ensure MVTec-AD is properly extracted with structure:\n"
                    f"  mvtec/{{category}}/{split}/good/*.png"
                )
            
            # Get all images in this category's good folder
            image_files = sorted([
                f for f in category_path.iterdir()
                if f.suffix.lower() in {'.png', '.jpg', '.jpeg'}
            ])
            
            for img_path in image_files:
                self.samples.append((str(img_path), category_idx))
                self.imgs.append((str(img_path), category_idx))
        
        if not self.samples:
            raise RuntimeError(
                f"No images found in MVTec-AD dataset at {self.root}\n"
                f"Looking for structure: mvtec/{{category}}/{split}/good/"
            )
        
        print(
            f"MVTec-AD Dataset ({split} split): "
            f"Found {len(self.samples)} images from {len(self.classes)} categories"
        )
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, index: int):
        """Get image and label at given index."""
        from PIL import Image
        
        path, target = self.samples[index]
        img = Image.open(path).convert("RGB")
        
        if self.transform is not None:
            img = self.transform(img)
        
        return img, target


def prepare_mvtec_dataset(
    train_data_path: Union[str, Path],
    val_data_path: Optional[Union[str, Path]] = None,
    transform_train=None,
    transform_val=None,
) -> Tuple[ImageFolder, Optional[ImageFolder]]:
    """
    Prepare MVTec-AD train and validation datasets.
    
    Args:
        train_data_path: Path to MVTec root directory
        val_data_path: Optional path to MVTec root for validation (usually same as train)
        transform_train: Transformations for training split
        transform_val: Transformations for validation split
        
    Returns:
        Tuple of (train_dataset, val_dataset)
    """
    
    train_dataset = create_mvtec_train_dataset(
        train_data_path,
        transform=transform_train,
        split="train",
    )
    
    # For validation, use test/good/ images if val_data_path is provided
    val_dataset = None
    if val_data_path is not None:
        val_dataset = create_mvtec_train_dataset(
            val_data_path,
            transform=transform_val,
            split="test",
        )
    
    return train_dataset, val_dataset
