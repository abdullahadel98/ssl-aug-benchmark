#!/usr/bin/env python3
"""
Evaluate linear evaluation models on CIFAR-100-C corrupted test sets.

This script evaluates models trained with main_linear.py on CIFAR-100-C corruption robustness.
It loads pretrained linear models and tests them on 19 corruption types with 5 severity levels each.

Usage:
    python eval_cifar100c.py --checkpoint <path_to_checkpoint.ckpt> --config <path_to_config.yaml>
    
    Or use the linear checkpoint directly:
    python eval_cifar100c.py --checkpoint ./trained_models/linear/checkpoint.ckpt \\
                             --config scripts/linear/cifar-100/simclr.yaml \\
                             --cifar100c-path ./datasets/CIFAR-100-C \\
                             --batch-size 256
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
from omegaconf import OmegaConf
from torch.utils.data import DataLoader, TensorDataset
from torchmetrics.classification import MulticlassCalibrationError
from torchvision import transforms
from tqdm import tqdm

from solo.args.linear import parse_cfg
from solo.methods.base import BaseMethod
from solo.methods.linear import LinearModel


# CIFAR-100-C corruption types (in order)
CIFAR100C_CORRUPTIONS = [
    'gaussian_noise', 'shot_noise', 'impulse_noise',  # Noise (indices 0-2)
    'defocus_blur', 'glass_blur', 'motion_blur', 'zoom_blur',  # Blur (3-6)
    'snow', 'frost', 'fog', 'brightness',  # Weather (7-10)
    'contrast', 'elastic_transform', 'pixelate', 'jpeg_compression',  # Digital (11-14)
    'speckle_noise', 'gaussian_blur', 'saturate', 'spatter',  # Additional (15-18)
]


class LogitsOnlyWrapper(nn.Module):
    """Wrapper to extract only logits from LinearModel output."""
    
    def __init__(self, model: LinearModel):
        super().__init__()
        self.model = model
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass returning only logits tensor."""
        out = self.model(x)
        if isinstance(out, dict):
            return out["logits"]
        return out


def load_linear_model(checkpoint_path: str, config_path: str = None, args_json_path: str = None) -> nn.Module:
    """Load a trained LinearModel from checkpoint.
    
    Args:
        checkpoint_path: Path to the .ckpt checkpoint file
        config_path: Optional path to the config yaml file
        args_json_path: Optional path to args.json file containing config
        
    Returns:
        Loaded model wrapped to return only logits
    """
    checkpoint_path = Path(checkpoint_path)
    
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    
    print(f"Loading checkpoint: {checkpoint_path}")
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    
    # Try to get config from checkpoint or load from file
    if "hyper_parameters" in checkpoint:
        cfg_dict = checkpoint["hyper_parameters"]
        if "cfg" in cfg_dict:
            cfg = cfg_dict["cfg"]
        else:
            cfg = OmegaConf.create(cfg_dict)
    elif args_json_path:
        # Load config from args.json file
        args_json_path = Path(args_json_path)
        if not args_json_path.exists():
            raise FileNotFoundError(f"args.json not found: {args_json_path}")
        print(f"Loading config from: {args_json_path}")
        cfg = OmegaConf.load(args_json_path)
        cfg = parse_cfg(cfg)
    elif config_path:
        cfg = OmegaConf.load(config_path)
        cfg = parse_cfg(cfg)
    else:
        raise ValueError("No config found in checkpoint, args.json, or config_path provided")
    
    # Build backbone
    backbone_model = BaseMethod._BACKBONES[cfg.backbone.name]
    backbone = backbone_model(method=cfg.pretrain_method, **cfg.backbone.kwargs)
    
    if cfg.backbone.name.startswith("resnet"):
        backbone.fc = nn.Identity()
        # CIFAR adaptation
        backbone.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=2, bias=False)
        backbone.maxpool = nn.Identity()
    
    # Create LinearModel
    model = LinearModel(backbone, cfg=cfg)
    
    # Load state dict
    if "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    else:
        state_dict = checkpoint
    
    model.load_state_dict(state_dict, strict=True)
    print(f"✓ Model loaded successfully")
    
    # Wrap to return only logits
    wrapped_model = LogitsOnlyWrapper(model)
    return wrapped_model


def load_cifar100c_corruption(
    cifar100c_path: Path, 
    corruption: str
) -> Tuple[np.ndarray, np.ndarray]:
    """Load a specific CIFAR-100-C corruption dataset.
    
    Args:
        cifar100c_path: Path to CIFAR-100-C directory
        corruption: Name of corruption type
        
    Returns:
        Tuple of (images, labels) as numpy arrays
    """
    corruption_file = cifar100c_path / f"{corruption}.npy"
    labels_file = cifar100c_path / "labels.npy"
    
    if not corruption_file.exists():
        raise FileNotFoundError(f"Corruption file not found: {corruption_file}")
    if not labels_file.exists():
        raise FileNotFoundError(f"Labels file not found: {labels_file}")
    
    # Load data
    images = np.load(corruption_file)
    labels = np.load(labels_file)
    
    return images, labels


def create_dataloader(
    images: np.ndarray,
    labels: np.ndarray,
    batch_size: int = 256,
    num_workers: int = 4
) -> DataLoader:
    """Create DataLoader from numpy arrays.
    
    Args:
        images: Images array of shape (N, H, W, C) with values in [0, 255]
        labels: Labels array of shape (N,)
        batch_size: Batch size for DataLoader
        num_workers: Number of workers for data loading
        
    Returns:
        DataLoader instance
    """
    # Convert to torch tensors and normalize
    # CIFAR-100-C images are uint8 in [0, 255] with shape (N, 32, 32, 3)
    images_tensor = torch.from_numpy(images).float()
    
    # Transpose from (N, H, W, C) to (N, C, H, W)
    images_tensor = images_tensor.permute(0, 3, 1, 2)
    
    # Normalize to [0, 1]
    images_tensor = images_tensor / 255.0
    
    # Apply CIFAR normalization (ImageNet stats are commonly used)
    normalize = transforms.Normalize(
        mean=[0.5071, 0.4867, 0.4408],
        std=[0.2675, 0.2565, 0.2761]
    )
    images_tensor = normalize(images_tensor)
    
    labels_tensor = torch.from_numpy(labels).long()
    
    dataset = TensorDataset(images_tensor, labels_tensor)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return loader


@torch.no_grad()
def evaluate_corruption(
    model: nn.Module,
    dataloader: DataLoader,
    device: str = "cuda",
    num_classes: int = 100
) -> Tuple[float, float]:
    """Evaluate model on a single corruption dataset.
    
    Args:
        model: Model to evaluate
        dataloader: DataLoader for corrupted images
        device: Device to run evaluation on
        num_classes: Number of classes
        
    Returns:
        Tuple of (accuracy, calibration_error)
    """
    model.eval()
    model = model.to(device)
    
    correct = 0
    total = 0
    
    all_targets = []
    all_predictions = []
    
    for images, targets in tqdm(dataloader, desc="Evaluating", leave=False):
        images = images.to(device)
        targets = targets.to(device)
        
        # Forward pass
        logits = model(images)
        
        # Get predictions
        _, predicted = logits.max(1)
        
        # Accumulate accuracy
        correct += predicted.eq(targets).sum().item()
        total += targets.size(0)
        
        # Store for calibration error
        all_targets.append(targets)
        all_predictions.append(logits)
    
    # Compute accuracy
    accuracy = 100.0 * correct / total
    
    # Compute calibration error
    all_targets = torch.cat(all_targets)
    all_predictions = torch.cat(all_predictions)
    
    calibration_metric = MulticlassCalibrationError(
        num_classes=num_classes,
        n_bins=15,
        norm='l1'
    )
    calibration_error = float(
        calibration_metric(all_predictions.cpu(), all_targets.cpu())
    )
    
    return accuracy, calibration_error


def evaluate_all_corruptions(
    model: nn.Module,
    cifar100c_path: Path,
    batch_size: int = 256,
    device: str = "cuda",
    num_workers: int = 4
) -> Dict[str, Dict[str, float]]:
    """Evaluate model on all CIFAR-100-C corruptions.
    
    Args:
        model: Model to evaluate
        cifar100c_path: Path to CIFAR-100-C directory
        batch_size: Batch size for evaluation
        device: Device to run evaluation on
        num_workers: Number of workers for data loading
        
    Returns:
        Dictionary mapping corruption names to metrics
    """
    results = {}
    
    print(f"\nEvaluating on CIFAR-100-C corruptions")
    print(f"Dataset path: {cifar100c_path}")
    print(f"Batch size: {batch_size}")
    print(f"Device: {device}\n")
    
    for corruption in tqdm(CIFAR100C_CORRUPTIONS, desc="Corruptions"):
        try:
            # Load corruption dataset
            images, labels = load_cifar100c_corruption(cifar100c_path, corruption)
            
            # Create dataloader
            dataloader = create_dataloader(
                images, labels, batch_size, num_workers
            )
            
            # Evaluate
            accuracy, calibration_error = evaluate_corruption(
                model, dataloader, device
            )
            
            results[corruption] = {
                "accuracy": accuracy,
                "calibration_error": calibration_error
            }
            
            print(f"{corruption:20s}: {accuracy:6.2f}% accuracy, "
                  f"ECE: {calibration_error:.4f}")
            
        except Exception as e:
            print(f"Error evaluating {corruption}: {e}")
            results[corruption] = {
                "accuracy": 0.0,
                "calibration_error": 1.0,
                "error": str(e)
            }
    
    return results


def compute_aggregate_metrics(results: Dict[str, Dict[str, float]]) -> Dict[str, float]:
    """Compute aggregate metrics from per-corruption results.
    
    Args:
        results: Per-corruption results dictionary
        
    Returns:
        Dictionary of aggregate metrics
    """
    accuracies = [r["accuracy"] for r in results.values() if "error" not in r]
    calibration_errors = [r["calibration_error"] for r in results.values() if "error" not in r]
    
    if not accuracies:
        return {}
    
    # Compute averages
    aggregates = {
        "mean_accuracy": np.mean(accuracies),
        "mean_calibration_error": np.mean(calibration_errors),
        "min_accuracy": np.min(accuracies),
        "max_accuracy": np.max(accuracies),
        "std_accuracy": np.std(accuracies),
    }
    
    # Compute by corruption category (if we have all 19 corruptions)
    if len(accuracies) >= 19:
        accs = [results[c]["accuracy"] for c in CIFAR100C_CORRUPTIONS[:19]]
        
        aggregates.update({
            "accuracy_noise": np.mean(accs[0:3]),  # gaussian, shot, impulse
            "accuracy_blur": np.mean(accs[3:7]),   # defocus, glass, motion, zoom
            "accuracy_weather": np.mean(accs[7:11]),  # snow, frost, fog, brightness
            "accuracy_digital": np.mean(accs[11:15]),  # contrast, elastic, pixelate, jpeg
            "accuracy_c15": np.mean(accs[0:15]),  # Original 15 corruptions
            "accuracy_c19": np.mean(accs[0:19]),  # All 19 corruptions
        })
    
    return aggregates


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate linear models on CIFAR-100-C corruptions"
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to trained linear model checkpoint (.ckpt)"
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to config file (if not in checkpoint)"
    )
    parser.add_argument(
        "--cifar100c-path",
        type=str,
        default="./datasets/CIFAR-100-C",
        help="Path to CIFAR-100-C dataset directory"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=256,
        help="Batch size for evaluation"
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=4,
        help="Number of data loading workers"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to use for evaluation"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Path to save results JSON file"
    )
    parser.add_argument(
        "--args-json",
        type=str,
        default=None,
        help="Path to args.json file containing config"
    )
    
    args = parser.parse_args()
    
    # Validate paths
    cifar100c_path = Path(args.cifar100c_path)
    if not cifar100c_path.exists():
        raise FileNotFoundError(f"CIFAR-100-C path not found: {cifar100c_path}")
    
    # Load model
    model = load_linear_model(args.checkpoint, args.config, args.args_json)
    model = model.to(args.device)
    model.eval()
    
    # Evaluate on all corruptions
    results = evaluate_all_corruptions(
        model,
        cifar100c_path,
        batch_size=args.batch_size,
        device=args.device,
        num_workers=args.num_workers
    )
    
    # Compute aggregate metrics
    aggregates = compute_aggregate_metrics(results)
    
    # Print summary
    print("\n" + "="*60)
    print("CIFAR-100-C Evaluation Summary")
    print("="*60)
    print(f"Mean Accuracy:        {aggregates.get('mean_accuracy', 0):.2f}%")
    print(f"Std Accuracy:         {aggregates.get('std_accuracy', 0):.2f}%")
    print(f"Min Accuracy:         {aggregates.get('min_accuracy', 0):.2f}%")
    print(f"Max Accuracy:         {aggregates.get('max_accuracy', 0):.2f}%")
    print(f"Mean Calibration Err: {aggregates.get('mean_calibration_error', 0):.4f}")
    
    if "accuracy_c19" in aggregates:
        print(f"\nBy Category:")
        print(f"  Noise (3):          {aggregates['accuracy_noise']:.2f}%")
        print(f"  Blur (4):           {aggregates['accuracy_blur']:.2f}%")
        print(f"  Weather (4):        {aggregates['accuracy_weather']:.2f}%")
        print(f"  Digital (4):        {aggregates['accuracy_digital']:.2f}%")
        print(f"\nAggregates:")
        print(f"  C-15 (original):    {aggregates['accuracy_c15']:.2f}%")
        print(f"  C-19 (all):         {aggregates['accuracy_c19']:.2f}%")
    
    print("="*60)
    
    # Save results
    if args.output:
        output_path = Path(args.output)
    else:
        checkpoint_name = Path(args.checkpoint).stem
        output_path = Path(f"cifar100c_results_{checkpoint_name}.json")
    
    output_data = {
        "checkpoint": str(args.checkpoint),
        "cifar100c_path": str(args.cifar100c_path),
        "results": results,
        "aggregates": aggregates
    }
    
    with open(output_path, 'w') as f:
        json.dump(output_data, f, indent=2)
    
    print(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    main()
