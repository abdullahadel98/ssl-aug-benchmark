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

import os
import numpy as np
import torch
import torch.nn as nn
from typing import Optional, List, Dict, Any
from omegaconf import DictConfig, OmegaConf

from solo.utils.misc import omegaconf_select


class BatchAugmentationMixin:
    """
    Mixin class to add batch-level augmentations to SSL methods.
    
    This mixin provides:
    - Setup of batch-level augmentation modules
    - Application of augmentations in training loop
    - Configuration validation
    - Device management
    
    Usage:
        class SimCLR(BatchAugmentationMixin, BaseMethod):
            def __init__(self, cfg):
                super().__init__(cfg)
                # After BaseMethod.__init__
                self.setup_batch_augmentations(cfg)
            
            def training_step(self, batch, batch_idx):
                indexes, X, targets = batch
                X = [X] if isinstance(X, torch.Tensor) else X
                
                # Apply batch augmentations
                X = self.apply_batch_augmentations(X)
                
                # Continue with original training_step
                ...
    """

    def setup_batch_augmentations(self, cfg: DictConfig) -> None:
        """
        Initialize batch-level augmentation modules.
        
        Args:
            cfg (DictConfig): Configuration object with batch_augmentations section
        """
        batch_aug_cfg = omegaconf_select(cfg, "batch_augmentations")
        
        self.batch_augmentations = {}
        
        # Optimization: Cache augmentation references and flags for hot path performance
        self._style_transfer_aug = None
        self._has_style_transfer = False
        self._has_augmentations = False
        self._aug_device = self.device
        
        if batch_aug_cfg is None:
            return
        
        # Style Transfer Setup
        st_cfg = omegaconf_select(batch_aug_cfg, "style_transfer")
        if st_cfg and st_cfg.get("enabled", False):
            self._setup_style_transfer(st_cfg)

    def _setup_style_transfer(self, cfg: Dict[str, Any]) -> None:
        """
        Initialize AdaIN style transfer augmentation using NSTTransform.
        
        Args:
            cfg (Dict): Style transfer configuration with keys:
                - features_path (str): Path to pre-extracted style features
                - encoder_path (str): Path to VGG encoder state dict
                - decoder_path (str): Path to decoder state dict
                - alpha_min (float): Minimum alpha for blending
                - alpha_max (float): Maximum alpha for blending
                - probability (float): Probability of applying augmentation
        """
        try:
            from solo.data.style_transfer import load_models, load_feat_files, NSTTransform
            
            # Expand environment variables in paths
            features_path = os.path.expandvars(os.path.expanduser(cfg.features_path))
            encoder_path = os.path.expandvars(os.path.expanduser(cfg.encoder_path))
            decoder_path = os.path.expandvars(os.path.expanduser(cfg.decoder_path))
            
            # Load models - ensure device is GPU
            device = self.device if torch.cuda.is_available() else torch.device("cuda:0")
            vgg, decoder = load_models(
                encoder_path=encoder_path,
                decoder_path=decoder_path,
                device=device,
            )
            
            # Load pre-extracted style features
            if not os.path.exists(features_path):
                raise FileNotFoundError(f"Style features not found: {features_path}")
            
            style_feats = load_feat_files(
                path=features_path,
                device=device,
            )
            
            # Create NSTTransform augmentation
            self.batch_augmentations["style_transfer"] = NSTTransform(
                style_feats=style_feats,
                vgg=vgg,
                decoder=decoder,
                alpha_min=cfg.get("alpha_min", 1.0),
                alpha_max=cfg.get("alpha_max", 1.0),
                probability=cfg.get("probability", 0.5),
                device=device,
            )
            
            # Optimization: Cache augmentation reference for hot path
            self._style_transfer_aug = self.batch_augmentations["style_transfer"]
            self._has_style_transfer = True
            self._has_augmentations = True
            
            self.log_text = f"✓ Style Transfer initialized with {len(style_feats)} styles"
            
        except ImportError as e:
            raise ImportError(
                f"Failed to import style transfer modules. Ensure solo/data/style_transfer.py is available. Error: {e}"
            )
        except Exception as e:
            raise RuntimeError(f"Failed to setup style transfer: {e}")

    def apply_batch_augmentations(self, X: List[torch.Tensor]) -> List[torch.Tensor]:
        """
        Apply batch-level augmentations to crop batches.
        
        Args:
            X (List[torch.Tensor]): List of batches, typically:
                - X[0]: Large crops [batch_size*num_large_crops, C, H, W]
                - X[1]: Small crops [batch_size*num_small_crops, C, H, W] (if multicrop)
                - ...
        
        Returns:
            List[torch.Tensor]: Augmented crop batches
        """
        # Optimization: Early exit with single flag check instead of dictionary lookup
        if not self._has_augmentations:
            return X
        
        # Optimization: Use cached augmentation reference (no dict lookup)
        if self._has_style_transfer:
            X[0] = self._style_transfer_aug(X[0])
        
        return X

    def apply_batch_augmentations_to_crops(
        self,
        X: List[torch.Tensor],
        crop_indices: Optional[List[int]] = None,
    ) -> List[torch.Tensor]:
        """
        Apply batch augmentations to specific crop types.
        
        Args:
            X (List[torch.Tensor]): List of batches
            crop_indices (List[int], optional): Indices of X to augment.
                Defaults to [0] (large crops only).
        
        Returns:
            List[torch.Tensor]: Augmented crop batches
        """
        if crop_indices is None:
            crop_indices = [0]  # Apply to large crops by default
        
        # Optimization: Early exit with single flag check
        if not self._has_augmentations:
            return X
        
        # Optimization: Use cached augmentation reference
        if self._has_style_transfer:
            for idx in crop_indices:
                if idx < len(X):
                    X[idx] = self._style_transfer_aug(X[idx])
        
        return X

    def get_batch_augmentation_stats(self) -> Dict[str, Any]:
        """Get statistics about configured batch augmentations."""
        return {
            "batch_augmentations_enabled": len(self.batch_augmentations) > 0,
            "augmentations": list(self.batch_augmentations.keys()),
        }


# Configuration schema validation helper
def validate_batch_augmentation_config(cfg: DictConfig) -> None:
    """
    Validate batch augmentation configuration.
    
    Args:
        cfg (DictConfig): Configuration to validate
        
    Raises:
        ValueError: If configuration is invalid
    """
    batch_aug_cfg = omegaconf_select(cfg, "batch_augmentations")
    
    if batch_aug_cfg is None:
        return
    
    st_cfg = omegaconf_select(batch_aug_cfg, "style_transfer")
    if st_cfg and st_cfg.get("enabled", False):
        if not st_cfg.get("features_path"):
            raise ValueError("style_transfer.features_path must be specified when enabled")
        
        alpha_min = st_cfg.get("alpha_min", 1.0)
        alpha_max = st_cfg.get("alpha_max", 1.0)
        prob = st_cfg.get("probability", 0.5)
        
        if not (0 <= alpha_min <= 1 and 0 <= alpha_max <= 1):
            raise ValueError("alpha_min and alpha_max must be in [0, 1]")
        
        if not (0 <= prob <= 1):
            raise ValueError("probability must be in [0, 1]")
        
        if alpha_min > alpha_max:
            raise ValueError("alpha_min must be <= alpha_max")
