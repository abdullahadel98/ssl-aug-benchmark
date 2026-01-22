"""
Example: How to integrate BatchAugmentationMixin into SimCLR

This example shows the minimal changes needed to add batch-level
style transfer augmentations to any SSL method in solo-learn.
"""

from typing import Any, Dict, List, Sequence

import omegaconf
import torch

from solo.losses.simclr import simclr_loss_func
from solo.methods.base import BaseMethod
from solo.methods.batch_augmentation_mixin import BatchAugmentationMixin


class SimCLRWithStyleTransfer(BatchAugmentationMixin, BaseMethod):
    """
    SimCLR with batch-level style transfer augmentation.
    
    This is a drop-in replacement for the standard SimCLR that adds
    neural style transfer at the batch level.
    
    Changes from standard SimCLR:
    1. Inherits from BatchAugmentationMixin
    2. Calls setup_batch_augmentations() in __init__
    3. Calls apply_batch_augmentations() in training_step()
    """

    def __init__(self, cfg: omegaconf.DictConfig):
        """Implements SimCLR with batch augmentations.

        Extra cfg settings:
            method_kwargs:
                proj_output_dim (int): number of dimensions of the projected features.
                proj_hidden_dim (int): number of neurons in the hidden layers of the projector.
                temperature (float): temperature for the softmax in the contrastive loss.
            
            batch_augmentations:  # NEW
                style_transfer:
                    enabled (bool): whether to apply style transfer
                    features_path (str): path to style features
                    alpha_min (float): minimum blend strength
                    alpha_max (float): maximum blend strength
                    probability (float): probability of augmentation
        """

        # Initialize base method first
        super().__init__(cfg)

        self.temperature: float = cfg.method_kwargs.temperature

        proj_hidden_dim: int = cfg.method_kwargs.proj_hidden_dim
        proj_output_dim: int = cfg.method_kwargs.proj_output_dim

        # projector
        self.projector = torch.nn.Sequential(
            torch.nn.Linear(self.features_dim, proj_hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(proj_hidden_dim, proj_output_dim),
        )

        # ✓ NEW: Setup batch augmentations
        self.setup_batch_augmentations(cfg)

    @staticmethod
    def add_and_assert_specific_cfg(cfg: omegaconf.DictConfig) -> omegaconf.DictConfig:
        """Adds method specific default values/checks for config."""

        cfg = super(SimCLRWithStyleTransfer, SimCLRWithStyleTransfer).add_and_assert_specific_cfg(cfg)

        assert not omegaconf.OmegaConf.is_missing(cfg, "method_kwargs.proj_output_dim")
        assert not omegaconf.OmegaConf.is_missing(cfg, "method_kwargs.proj_hidden_dim")
        assert not omegaconf.OmegaConf.is_missing(cfg, "method_kwargs.temperature")

        return cfg

    @property
    def learnable_params(self) -> List[dict]:
        """Adds projector parameters to the parent's learnable parameters."""

        extra_learnable_params = [{"name": "projector", "params": self.projector.parameters()}]
        return super().learnable_params + extra_learnable_params

    def forward(self, X: torch.tensor) -> Dict[str, Any]:
        """Performs the forward pass of the backbone and the projector."""

        out = super().forward(X)
        z = self.projector(out["feats"])
        out.update({"z": z})
        return out

    def multicrop_forward(self, X: torch.tensor) -> Dict[str, Any]:
        """Performs the forward pass for the multicrop views."""

        out = super().multicrop_forward(X)
        z = self.projector(out["feats"])
        out.update({"z": z})
        return out

    def training_step(self, batch: Sequence[Any], batch_idx: int) -> torch.Tensor:
        """Training step for SimCLR with batch augmentations.

        Args:
            batch (Sequence[Any]): a batch of data in the format of [img_indexes, [X], Y], where
                [X] is a list of size num_crops containing batches of images.
            batch_idx (int): index of the batch.

        Returns:
            torch.Tensor: total loss composed of SimCLR loss and classification loss.
        """

        indexes = batch[0]
        _, X, targets = batch

        # Prepare crops
        X = [X] if isinstance(X, torch.Tensor) else X
        assert len(X) == self.num_crops

        # ✓ NEW: Apply batch-level augmentations (style transfer)
        X = self.apply_batch_augmentations(X)

        # Original SimCLR training logic
        outs = [self.base_training_step(x, targets) for x in X[: self.num_large_crops]]
        outs = {k: [out[k] for out in outs] for k in outs[0].keys()}

        if self.multicrop:
            multicrop_outs = [self.multicrop_forward(x) for x in X[self.num_large_crops :]]
            for k in multicrop_outs[0].keys():
                outs[k] = outs.get(k, []) + [out[k] for out in multicrop_outs]

        # loss and stats
        outs["loss"] = sum(outs["loss"]) / self.num_large_crops
        outs["acc1"] = sum(outs["acc1"]) / self.num_large_crops
        outs["acc5"] = sum(outs["acc5"]) / self.num_large_crops

        metrics = {
            "train_class_loss": outs["loss"],
            "train_acc1": outs["acc1"],
            "train_acc5": outs["acc5"],
        }

        self.log_dict(metrics, on_epoch=True, sync_dist=True)

        if self.knn_eval:
            targets = targets.repeat(self.num_large_crops)
            mask = targets != -1
            self.knn(
                train_features=torch.cat(outs["feats"][: self.num_large_crops])[mask].detach(),
                train_targets=targets[mask],
            )

        return outs

    # If you want method-specific behavior, override training_step like above
    # The mixin provides apply_batch_augmentations() which by default:
    # - Applies to X[0] (large crops)
    # - Can be customized via apply_batch_augmentations_to_crops()


# ============================================================================
# USAGE EXAMPLE
# ============================================================================

"""
To use this integrated method:

1. Update your config file to include batch_augmentations:

    # configs/pretrain/simclr_with_st.yaml
    batch_augmentations:
      style_transfer:
        enabled: true
        features_path: "augmentation/mbda/features/style_feats_adain_1000.npy"
        alpha_min: 0.7
        alpha_max: 1.0
        probability: 0.5

2. Update METHODS registry in solo/methods/__init__.py:

    from solo.methods.simclr_with_style_transfer import SimCLRWithStyleTransfer
    
    METHODS = {
        "simclr": SimCLR,
        "simclr_st": SimCLRWithStyleTransfer,  # NEW
        ...
    }

3. Run training:

    python main_pretrain.py \
        --config-path configs/pretrain \
        --config-name simclr_with_st \
        method=simclr_st

OR just use standard SimCLR with batch_augmentations config:

    python main_pretrain.py \
        --config-path configs/pretrain \
        --config-name simclr \
        +batch_augmentations.style_transfer.enabled=true \
        +batch_augmentations.style_transfer.features_path="augmentation/mbda/features/style_feats_adain_1000.npy"

============================================================================
"""
