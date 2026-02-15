# AI Agent Instructions for SSL Augmentation Benchmark

## Project Overview

This is an SSL (Self-Supervised Learning) augmentation benchmark integrating **batch-level Neural Style Transfer (AdaIN)** into the **solo-learn** framework. The project evaluates how augmentation methods affect SSL model training and downstream performance.

### Architecture

```
ssl-aug-benchmark/
├── learning/solo-learn/          # Main SSL framework (PyTorch Lightning)
│   ├── solo/methods/             # SSL method implementations (SimCLR, MoCo, BYOL, DINO, etc.)
│   │   ├── batch_augmentation_mixin.py     # ⭐ Mixin to enable batch-level augmentations
│   │   ├── {simclr,mocov3,byol,dino}.py    # Individual SSL methods (inherit mixin)
│   │   └── base.py               # BaseMethod parent class
│   ├── solo/data/
│   │   ├── batch_augmentations.py # ⭐ BatchStyleTransfer class (AdaIN implementation)
│   │   ├── style_transfer.py     # Style transfer module
│   │   └── pretrain_dataloader.py
│   └── configs/pretrain/         # YAML configs for each SSL method + dataset
│
├── augmentation/mbda/            # Data augmentation experiments
│   ├── experiments/
│   │   ├── style_transfer.py     # Original NSTTransform (reference)
│   │   └── adaIN/                # Pre-trained AdaIN encoder/decoder weights
│   ├── features/
│   │   └── style_feats_adain_1000.npy  # Pre-extracted style features
│   └── data/
│
└── scripts/                      # Experiment runners
    ├── run_linear_evaluations.sh # Evaluate all pretrained models
    └── read_display.py           # Visualization utilities
```

### Key Data Flow

**Augmentation Pipeline** (batch-level):
```
Dataset → Per-Image Augments (ColorJitter, Rotation, Crop)
       → Batch Collation (6-8 crops per image)
       → BATCH STYLE TRANSFER (AdaIN) ⭐ ← Applied here
       → Model Forward Pass → Loss Computation
```

## Critical Workflows

### 1. Pretraining SSL Models

```bash
source ~/miniconda3/etc/profile.d/conda.sh
conda activate sololearn
cd learning/solo-learn/

python main_pretrain.py --config-path scripts/pretrain/cifar/ ...
```

**Command Pattern** (from learning/solo-learn/):
```bash
python main_pretrain.py \
    --config-path scripts/pretrain/[dataset]/ \
    --config-name [method_config].yaml \
    ++name="[experiment_name]" \
    ++data.dataset=[cifar10|cifar100|imagenet|mvtec-ad] \
    ++checkpoint.dir="/path/to/experiments/[exp_name]"
```

**Real Examples:**
```bash
# SimCLR baseline (no style transfer)
python main_pretrain.py --config-path scripts/pretrain/cifar/ --config-name simclr_original.yaml ++name="simclr-og-cifar100" ++data.dataset=cifar100

# DINO with TrivialAug
python main_pretrain.py --config-path scripts/pretrain/cifar/ --config-name dino_trivAug.yaml ++name="dino-trivaug-cifar100" ++data.dataset=cifar100

# BYOL with MVTec-AD
python main_pretrain.py --config-path scripts/pretrain/mvtec-ad/ --config-name byol.yaml ++name="byol-mvtec-ad" ++data.dataset=mvtec-ad
```

**Key Config Additions** for style transfer (YAML):
```yaml
batch_augmentations:
  style_transfer:
    enabled: true
    features_path: "augmentation/mbda/features/style_feats_adain_1000.npy"
    probability: 0.5      # Probability of applying style transfer
    alpha_min: 1.0        # Strength of style transfer blending
    alpha_max: 1.0
```

### 2. Linear Evaluation (Downstream Task)

```bash
source ~/miniconda3/etc/profile.d/conda.sh
conda activate sololearn
cd learning/solo-learn/

python main_pretrain.py --config-path scripts/pretrain/cifar/ ...
```

**Automated script:**
```bash
./run_linear_evaluations.sh
```

**Manual evaluation:**
```bash
cd learning/solo-learn/
python main_linear.py \
    --config-path scripts/linear/cifar/ \
    --config-name simclr.yaml \
    ++backbone.pretrained_weights_path="/path/to/checkpoint.ckpt" \
    ++data.dataset=cifar100
```


## Code Patterns & Conventions

### Adding Style Transfer to SSL Methods

**Pattern**: Inherit `BatchAugmentationMixin` + call setup methods

```python
# In learning/solo-learn/solo/methods/[method].py
from solo.methods.batch_augmentation_mixin import BatchAugmentationMixin
from solo.methods.base import BaseMethod

class SimCLR(BatchAugmentationMixin, BaseMethod):
    def __init__(self, cfg):
        super().__init__(cfg)
        self.setup_batch_augmentations(cfg)  # ← Required
    
    def training_step(self, batch, batch_idx):
        indexes, X, targets = batch
        X = [X] if isinstance(X, torch.Tensor) else X
        X = self.apply_batch_augmentations(X)  # ← Apply here
        # ... rest of training logic ...
```

**Mixin Implementation Details** (`batch_augmentation_mixin.py`):
- `setup_batch_augmentations(cfg)` - Initializes AdaIN models & loads style features
- `apply_batch_augmentations(X)` - Applies batch-level augmentations to list of crops
- Handles device placement, model loading, and configuration validation

### Batch Augmentation Module

**Class**: `BatchStyleTransfer` (in `solo/data/batch_augmentations.py`)

```python
class BatchStyleTransfer:
    def __init__(self, vgg, decoder, style_features, alpha_min=1.0, probability=0.5):
        # vgg: VGG encoder (layers 0-31)
        # decoder: AdaIN decoder
        # style_features: Pre-extracted tensor [1000, feat_dim]
        # alpha: Blending strength [0,1] where 1=full style, 0=original
        # probability: Chance to apply style per batch
```

Key behaviors:
- Upsamples crops to 224×224 (AdaIN requirement)
- Handles both RGB and grayscale (auto-converts grayscale to RGB)
- Randomly selects subset of batch to stylize (controlled by probability)
- Returns batch with same shape as input

### Configuration System (OmegaConf)

- **Configs**: `learning/solo-learn/configs/pretrain/[dataset]/`
- **Override Pattern**: `++key=value` from CLI (e.g., `++devices=[0,1]`)
- **Access Pattern**: `cfg.method_kwargs.temperature` or `omegaconf_select(cfg, "path.to.key")`

## Developer Conventions

### Experiment Tracking

- **WandB Integration**: Logs training metrics automatically
- **Checkpoint Directory**: `~/my_work/code/experiments/[exp_name]/` (contains `.ckpt` files)
- **Environment Setup**: Use `sololearn` conda environment; load WandB API key from `.env`

### File Organization

- **Logs**: Root directory (`.log`, `.err.log` files from SLURM)
- **Pretrained Weights**: `learning/solo-learn/trained_models/` or passed via CLI
- **Features**: `augmentation/mbda/features/` (1000 pre-extracted style features)
- **Configs**: Dataset-specific subdirectories in `configs/pretrain/`

### Testing & Validation

- Run `test_style_transfer.py` to validate AdaIN models load correctly
- Linear evaluation logs show downstream task performance (top-1 accuracy)
- Check `linear_eval.log` for evaluation completion status

## External Dependencies & Integration

- **PyTorch Lightning**: Training framework (version in `requirements.txt`)
- **Solo-Learn**: SSL method implementations & base classes
- **AdaIN Models**: Pre-trained VGG + decoder weights in `augmentation/mbda/experiments/adaIN/`
- **Datasets**: CIFAR-10/100, ImageNet, MVTec-AD (auto-downloaded or pre-staged)
- **WandB**: Experiment tracking (optional; disable with `wandb.enabled=False`)

## Common Troubleshooting

| Issue | Check |
|-------|-------|
| CUDA OOM | Reduce batch size via `++data.batch_size=64` |
| Style features not found | Verify path exists: `augmentation/mbda/features/style_feats_adain_1000.npy` |
| Mixin not working | Ensure inheritance order: `class Method(Mixin, BaseMethod)` |
| SLURM job fails | Check conda environment: `conda activate sololearn` |
| WandB auth error | Load `.env`: `source .env` before running |

## Key Files to Read First

1. [MASTER_INTEGRATION_GUIDE.md](MASTER_INTEGRATION_GUIDE.md) - Complete integration reference (read for detailed understanding)
2. [README.md](README.md) - Project overview & quick navigation
3. [SETUP.md](SETUP.md) - Environment setup
4. `learning/solo-learn/solo/methods/batch_augmentation_mixin.py` - How augmentations integrate
5. `learning/solo-learn/solo/data/batch_augmentations.py` - How style transfer is applied

## Important Paths Reference

```
learning/solo-learn/
├── main_pretrain.py              # Entry point for SSL pretraining
├── main_linear.py                # Entry point for downstream evaluation
├── solo/methods/[method].py       # Modify here to add mixin
└── configs/pretrain/cifar/        # Modify YAML configs here

augmentation/mbda/
├── features/style_feats_adain_1000.npy  # Load in config: features_path
└── experiments/adaIN/             # Pre-trained weights (VGG, decoder)

scripts/
├── run_linear_evaluations.sh      # Run all downstream evals
└── read_display.py                # Visualization
```
