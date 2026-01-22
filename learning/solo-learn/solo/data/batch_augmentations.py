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
import math
import random
import numpy as np
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torchvision.transforms.v2 as transforms
import torchvision.transforms.functional as F


class BatchStyleTransfer:
    """
    Batch-level Neural Style Transfer using AdaIN.
    Applies style transfer to a batch of augmented crops.
    
    This module loads pre-extracted style features and applies AdaIN
    (Adaptive Instance Normalization) style transfer to batched tensors.
    
    Parameters:
        vgg (nn.Module): Pre-trained VGG encoder (cut at layer 31)
        decoder (nn.Module): AdaIN decoder
        style_features (torch.Tensor): Pre-extracted style features [num_styles, feat_dim]
        alpha_min (float): Minimum alpha for style transfer blending [0, 1]
        alpha_max (float): Maximum alpha for style transfer blending [0, 1]
        probability (float): Probability of applying style transfer to samples in batch [0, 1]
        device (str or torch.device): Device to run computations on
    """

    def __init__(
        self,
        vgg: nn.Module,
        decoder: nn.Module,
        style_features: torch.Tensor,
        alpha_min: float = 1.0,
        alpha_max: float = 1.0,
        probability: float = 0.5,
        device: str = "cuda",
    ):
        self.vgg = vgg
        self.decoder = decoder
        self.style_features = style_features
        self.alpha_min = alpha_min
        self.alpha_max = alpha_max
        self.probability = probability
        self.device = device if isinstance(device, torch.device) else torch.device(device)
        
        self.num_styles = len(style_features)
        self.upsample = nn.Upsample(size=(224, 224), mode='bilinear', align_corners=False)
        self.to_pil_img = transforms.ToPILImage()

    @torch.no_grad()
    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply style transfer to a batch of images.
        
        Args:
            x (torch.Tensor): Batch of images [batch_size, C, H, W]
            
        Returns:
            torch.Tensor: Style-transferred batch [batch_size, C, H, W]
        """
        batchsize = x.size(0)
        ratio = int(math.floor(batchsize * self.probability + random.random()))
        
        # If no samples to stylize, return as-is
        if ratio == 0:
            return x

        # Detect grayscale
        was_grayscale = x.shape[1] == 1
        if was_grayscale:
            # Repeat channel → RGB
            x = x.repeat(1, 3, 1, 1)

        _, _, H, W = x.shape
        if (H, W) != (224, 224):
            x = self.upsample(x)

        # Choose random subset to stylize
        idy = torch.randperm(self.num_styles)[:ratio]
        idx = torch.randperm(batchsize)[:ratio]

        x = x.to(self.device)
        x[idx] = self._style_transfer(x[idx], self.style_features[idy])
        stl_imgs = x.cpu()

        # Restore original resolution
        if (H, W) != (224, 224):
            stl_imgs = nn.Upsample(size=(H, W), mode='bilinear', align_corners=False)(stl_imgs)

        # Convert back to grayscale if needed
        if was_grayscale:
            stl_imgs = F.rgb_to_grayscale(stl_imgs)

        return stl_imgs

    @torch.no_grad()
    def _style_transfer(self, content: torch.Tensor, style: torch.Tensor) -> torch.Tensor:
        """
        Apply AdaIN style transfer to content using given style features.
        
        Args:
            content (torch.Tensor): Content images [batch_size, 3, 224, 224]
            style (torch.Tensor): Style features [num_styles, feat_dim]
            
        Returns:
            torch.Tensor: Style-transferred images [batch_size, 3, 224, 224]
        """
        alpha = np.random.uniform(low=self.alpha_min, high=self.alpha_max)
        
        # Extract content features
        content_f = self.vgg(content)
        
        # Apply AdaIN
        feat = self._adaptive_instance_normalization(content_f, style)
        feat = feat * alpha + content_f * (1 - alpha)
        
        # Decode back to image space
        return self.decoder(feat)

    @staticmethod
    @torch.no_grad()
    def _adaptive_instance_normalization(
        content_feat: torch.Tensor, 
        style_feat: torch.Tensor
    ) -> torch.Tensor:
        """
        Adaptive Instance Normalization (AdaIN).
        
        Args:
            content_feat (torch.Tensor): Content features [batch_size, C, H, W]
            style_feat (torch.Tensor): Style features [num_styles, C, H, W] or [C]
            
        Returns:
            torch.Tensor: AdaIN-normalized features
        """
        # Handle both pre-extracted features (vectors) and spatial features
        if style_feat.ndim == 2:
            # Pre-extracted features [num_styles, C]
            # Use random style for each sample in batch
            num_samples = content_feat.size(0)
            style_indices = torch.randint(0, style_feat.size(0), (num_samples,))
            style_feat = style_feat[style_indices]  # [num_samples, C]
            
            # Compute statistics
            c_mean = content_feat.mean(dim=(2, 3), keepdim=True)  # [B, C, 1, 1]
            c_std = content_feat.std(dim=(2, 3), keepdim=True)    # [B, C, 1, 1]
            s_mean = style_feat.mean(dim=1, keepdim=True).unsqueeze(-1)  # [B, C, 1, 1]
            s_std = style_feat.std(dim=1, keepdim=True).unsqueeze(-1)    # [B, C, 1, 1]
        else:
            # Spatial features [num_styles, C, H, W] or [C, H, W]
            c_mean = content_feat.mean(dim=(2, 3), keepdim=True)
            c_std = content_feat.std(dim=(2, 3), keepdim=True)
            s_mean = style_feat.mean(dim=(2, 3), keepdim=True)
            s_std = style_feat.std(dim=(2, 3), keepdim=True)

        # Normalize content
        normalized = (content_feat - c_mean) / (c_std + 1e-5)
        
        # Apply style statistics
        return normalized * s_std + s_mean


class BatchGaussianBlur:
    """
    Batch-level Gaussian blur augmentation.
    Applies different blur radii to random samples in the batch.
    """

    def __init__(self, sigma_min: float = 0.1, sigma_max: float = 2.0, probability: float = 0.5):
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.probability = probability

    @torch.no_grad()
    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply Gaussian blur to batch.
        
        Args:
            x (torch.Tensor): Batch of images [batch_size, C, H, W]
            
        Returns:
            torch.Tensor: Blurred batch
        """
        batchsize = x.size(0)
        ratio = int(math.floor(batchsize * self.probability + random.random()))
        
        if ratio == 0:
            return x

        idx = torch.randperm(batchsize)[:ratio]
        
        for i in idx:
            sigma = random.uniform(self.sigma_min, self.sigma_max)
            x[i] = transforms.GaussianBlur(kernel_size=5, sigma=(sigma, sigma))(x[i:i+1])[0]
        
        return x


def adaptive_instance_normalization(
    content_feat: torch.Tensor,
    style_feat: torch.Tensor,
    eps: float = 1e-5
) -> torch.Tensor:
    """
    Adaptive Instance Normalization (AdaIN) - standalone function.
    
    Args:
        content_feat (torch.Tensor): Content features
        style_feat (torch.Tensor): Style features
        eps (float): Small value for numerical stability
        
    Returns:
        torch.Tensor: AdaIN-normalized features
    """
    c_mean, c_std = _batch_statistics(content_feat)
    s_mean, s_std = _batch_statistics(style_feat)

    normalized = (content_feat - c_mean) / (c_std + eps)
    return normalized * s_std + s_mean


def _batch_statistics(x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute mean and std for spatial dimensions.
    
    Args:
        x (torch.Tensor): Input tensor [B, C, H, W] or [C, H, W]
        
    Returns:
        Tuple[torch.Tensor, torch.Tensor]: (mean, std)
    """
    if x.ndim == 4:
        # [B, C, H, W]
        mean = x.mean(dim=(2, 3), keepdim=True)
        std = x.std(dim=(2, 3), keepdim=True)
    elif x.ndim == 3:
        # [C, H, W]
        mean = x.mean(dim=(1, 2), keepdim=True)
        std = x.std(dim=(1, 2), keepdim=True)
    else:
        raise ValueError(f"Expected 3D or 4D tensor, got {x.ndim}D")
    
    return mean, std
