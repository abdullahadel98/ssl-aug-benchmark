import os
import math
import random
import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms.v2 as transforms
import torchvision.transforms.functional as F
from solo.backbones.adaIN import model as adaINmodel
from solo.backbones.adaIN import utils

def load_models(encoder_path: str, decoder_path: str, device):
    """
    Load pre-trained VGG encoder and decoder.
    
    Args:
        encoder_path: Path to VGG encoder state dict
        decoder_path: Path to decoder state dict
        device: torch device to load models on
    
    Returns:
        vgg, decoder: Loaded and eval-mode models on device
    """
    # Expand environment variables in paths
    encoder_path = os.path.expandvars(os.path.expanduser(encoder_path))
    decoder_path = os.path.expandvars(os.path.expanduser(decoder_path))
    
    vgg = adaINmodel.vgg
    decoder = adaINmodel.decoder
    vgg.load_state_dict(torch.load(encoder_path, weights_only=True))
    vgg = nn.Sequential(*list(vgg.children())[:31])
    decoder.load_state_dict(torch.load(decoder_path, weights_only=True))

    # Move to device and set to float32 for numerical stability
    vgg = vgg.to(device).float()
    decoder = decoder.to(device).float()

    vgg.eval()
    decoder.eval()
    
    # Disable gradient computation for style transfer networks (frozen models)
    for param in vgg.parameters():
        param.requires_grad = False
    for param in decoder.parameters():
        param.requires_grad = False
    
    return vgg, decoder

def load_feat_files(path: str, device):
    """
    Load pre-extracted style features.
    
    Args:
        path: Path to numpy file containing style features
        device: torch device to load features on
    
    Returns:
        style_feats_tensor: Tensor of style features on device
    """
    # Expand environment variables in path
    path = os.path.expandvars(os.path.expanduser(path))
    style_feats_np = np.load(path)
    style_feats_tensor = torch.from_numpy(style_feats_np).to(device)
    return style_feats_tensor


class NSTTransform(transforms.Transform):
    """
    A class to apply neural style transfer with AdaIN to datasets in the training pipeline.
    Now supports both RGB (3-channel) and grayscale (1-channel) images.
    
    Parameters:
        style_feats: Style features extracted from the style images using adaIN Encoder
        vgg: AdaIN Encoder
        decoder: AdaIN Decoder
        alpha_min: Minimum strength of style transfer [between 0 and 1]
        alpha_max: Maximum strength of style transfer [between 0 and 1]
        probability: Probability of applying style transfer [between 0 and 1]
        device: torch device for computation
    """

    def __init__(self, style_feats, vgg, decoder,
                 alpha_min=1.0, alpha_max=1.0,
                 probability=0.5, device=None):
        super().__init__()
        self.vgg = vgg
        self.decoder = decoder
        self.alpha_min = alpha_min
        self.alpha_max = alpha_max
        self.device = device or next(vgg.parameters()).device
        self.upsample = nn.Upsample(size=(224, 224), mode='bilinear', align_corners=False)
        self.style_features = style_feats
        self.num_styles = len(style_feats)
        self.probability = probability

    @torch.no_grad()
    def __call__(self, x):
        # Ensure input is on correct device (critical for mixed precision with DDP)
        # Update self.device if VGG has been moved to different device (happens during DDP setup)
        if hasattr(self, 'vgg'):
            vgg_device = next(self.vgg.parameters()).device
            if self.device != vgg_device:
                self.device = vgg_device
                # Also move style features to new device if needed
                if self.style_features.device != self.device:
                    self.style_features = self.style_features.to(device=self.device, non_blocking=True)
        
        x = x.to(device=self.device, non_blocking=True)
        
        single_image = x.ndimension() == 3
        if single_image:
            x = x.unsqueeze(0)  # [C,H,W] → [1,C,H,W]

        # Preserve original dtype and device for restoration (mixed precision support)
        original_dtype = x.dtype
        original_device = x.device

        batchsize = x.size(0)
        ratio = int(math.floor(batchsize * self.probability + random.random()))
        if ratio == 0:
            return x.squeeze(0) if single_image else x

        # Detect grayscale
        was_grayscale = x.shape[1] == 1
        if was_grayscale:
            # Repeat channel → RGB
            x = x.repeat(1, 3, 1, 1)

        _, _, H, W = x.shape
        if (H, W) != (224, 224):
            x = self.upsample(x)

        # Choose random subset to stylize (GPU-accelerated, on correct device)
        idy = torch.randperm(self.num_styles, device=self.device)[:ratio]
        idx = torch.randperm(batchsize, device=self.device)[:ratio]

        # Convert to float32 on correct device (needed for VGG in mixed precision)
        x_work = x.to(device=self.device, dtype=torch.float32, non_blocking=True)
        
        # Extract subset and apply style transfer
        x_subset = x_work[idx]  # Select subset to augment

        x_subset = self.style_transfer(self.vgg, self.decoder, x_subset, self.style_features[idy])
        x_work[idx] = x_subset  # Assign back
        
        # Optimization 1: Post-process on GPU before final transfer
        if (H, W) != (224, 224):
            x_work = nn.Upsample(size=(H, W), mode='bilinear', align_corners=False)(x_work)

        # Convert back to grayscale on GPU if needed
        if was_grayscale:
            x_work = F.rgb_to_grayscale(x_work)

        # Convert back to original dtype and device (mixed precision support)
        stl_imgs = x_work.to(dtype=original_dtype, device=original_device)

        if single_image:
            stl_imgs = stl_imgs.squeeze(0)  # Back to [C,H,W]

        return stl_imgs

    @staticmethod
    def _adaptive_instance_normalization(content_feat, style_feat):
        """
        Adaptive Instance Normalization (AdaIN) using adaIN.utils.
        
        Args:
            content_feat: Content features [N, C, H, W]
            style_feat: Style features [N, C] or [N, C, H, W]
        
        Returns:
            Normalized content features with style statistics
        """
        return utils.adaptive_instance_normalization(content_feat, style_feat)

    @torch.no_grad()
    def style_transfer(self, vgg, decoder, content, style):
        """
        Apply style transfer to content using VGG encoder and decoder.
        Designed for mixed precision training - keeps VGG in float32.
        
        Args:
            vgg: VGG encoder (stays in float32)
            decoder: AdaIN decoder (stays in float32)
            content: Content images [N, C, H, W] (float32)
            style: Style features [N, C] (float32)
        
        Returns:
            Stylized images [N, C, H, W] (float32)
        """
        # Force VGG back to float32 if it was converted by DDP
        vgg.float()
        decoder.float()
        
        # Disable automatic mixed precision for VGG operations
        # This prevents DDP wrapper from converting models to lower precision
        with torch.autocast(device_type=self.device.type, enabled=False):
            # Ensure content is on correct device and in float32
            # Check and move device first, then dtype
            if content.device != self.device:
                content = content.to(device=self.device, non_blocking=True)
            if content.dtype != torch.float32:
                content = content.to(dtype=torch.float32, non_blocking=True)
            
            # Forward through VGG (stays in float32)
            alpha = np.random.uniform(low=self.alpha_min, high=self.alpha_max)
            content_f = vgg(content)
            
            # Adaptive instance normalization
            feat = utils.adaptive_instance_normalization(content_f, style)
            
            # Optimization 5: In-place blending for memory efficiency
            feat.mul_(alpha).add_(content_f, alpha=1-alpha)
            
            # Decode back to image space (stays in float32)
            return decoder(feat)
            
            # Adaptive instance normalization
            feat = utils.adaptive_instance_normalization(content_f, style)
            
            # Optimization 5: In-place blending for memory efficiency
            feat.mul_(alpha).add_(content_f, alpha=1-alpha)
            
            # Decode back to image space (stays in float32)
            return decoder(feat)

