# Style Transfer Module Test Suite

This guide explains how to run and understand the comprehensive tests for the `style_transfer.py` module in your SSL-Aug-Benchmark project.

## Overview

The test suite (`test_style_transfer.py`) validates the neural style transfer augmentation module used in your SimCLR experiments with style transfer. It tests:

1. **Module Loading**: VGG encoder, decoder, and style features
2. **Input/Output Validation**: Shape, dtype, and device consistency
3. **Functionality**: Probability-based application, alpha blending, grayscale handling
4. **Robustness**: Memory efficiency, device transfers, gradient mode compatibility
5. **Integration**: Batch augmentation mixin integration

## Running the Tests

### Basic Usage

```bash
cd /home/RUS_CIP/st190519/my_work/code/ssl-aug-benchmark

# Run with conda environment
conda run -n sololearn python test_style_transfer.py

# Or if conda is not needed
python test_style_transfer.py
```

### Output Example

```
================================================================================
StyleTransfer Module Test Suite
Device: cuda
================================================================================

Test 1: Synthetic Style Features Creation
─────────────────────────────────────────────────────────────────────────────
[✓ PASS] Create synthetic style features
      Shape: (10, 512), dtype: torch.float32

Test 2: Synthetic Model Creation (Mock VGG + Decoder)
─────────────────────────────────────────────────────────────────────────────
[✓ PASS] Create synthetic VGG encoder
      Total params: 297216
[✓ PASS] Create synthetic Decoder
      Total params: 1186819

...
```

## Test Breakdown

### Test 1: Synthetic Style Features Creation
- **Purpose**: Verify style features tensor creation
- **Input**: Random tensor with shape (10, 512)
- **Expected Output**: Proper tensor on correct device
- **Why Synthetic**: Uses pre-extracted features from AdaIN encoder

### Test 2: Synthetic Model Creation
- **Purpose**: Initialize mock VGG encoder and decoder
- **Input**: None (creates models internally)
- **Expected Output**: Two trainable models with correct parameter counts
- **Note**: Tests use mock models instead of real VGG to avoid file dependencies

### Test 3: NSTTransform Initialization
- **Purpose**: Instantiate the style transfer transform
- **Configuration**:
  - `alpha_min`: 0.7 (minimum blending strength)
  - `alpha_max`: 1.0 (maximum blending strength)
  - `probability`: 0.5 (50% chance to apply)
- **Expected Output**: Successfully initialized transform object

### Test 4: Single Image Input [C, H, W]
- **Purpose**: Test with individual images
- **Test Cases**:
  - RGB image: (3, 32, 32)
  - RGB image: (3, 64, 64)
  - Grayscale: (1, 32, 32)
- **Validation Checks**:
  - ✓ Shape preserved
  - ✓ Dtype preserved
  - ✓ Device consistency
  - ✓ No NaN/Inf values

### Test 5: Batch Input [B, C, H, W]
- **Purpose**: Test with batch processing (main training scenario)
- **Test Cases**:
  - Small batch: (2, 3, 32, 32)
  - Medium batch: (8, 3, 64, 64)
  - Large batch: (16, 3, 224, 224) - matches training config
  - Grayscale batch: (4, 1, 32, 32)
- **Validation Checks**:
  - ✓ Shape consistency
  - ✓ Dtype preservation
  - ✓ Value range [0, 1]
  - ✓ Finite values (no NaN/Inf)

### Test 6: Different Data Types
- **Purpose**: Verify mixed precision compatibility
- **Data Types Tested**:
  - float32 (standard)
  - float64 (double precision)
  - bfloat16 (mixed precision, if CUDA available)
- **Validation**: Output dtype is compatible with input (float32 is acceptable)

### Test 7: Probability-based Application
- **Purpose**: Verify that style transfer respects probability parameter
- **Scenarios**:
  - probability = 1.0 (always apply) → output ≠ input
  - probability = 0.0 (never apply) → output = input
- **Importance**: Critical for controlling augmentation strength during training

### Test 8: Alpha Blending Strength
- **Purpose**: Test that alpha parameter controls style transfer strength
- **Alpha Values Tested**:
  - alpha = 1.0 (full style transfer)
  - alpha = 0.5 (50% style transfer)
  - alpha = 0.0 (no style transfer)
- **Validation**: Different alpha values produce different outputs
- **Formula**: `output = alpha * stylized + (1-alpha) * content`

### Test 9: Grayscale Image Handling
- **Purpose**: Ensure grayscale images work correctly
- **Mechanism**: 
  - Convert grayscale (1, H, W) → RGB (3, H, W)
  - Apply style transfer
  - Convert back to grayscale
- **Test Cases**:
  - Single grayscale image: (1, 32, 32)
  - Batch of grayscale images: (4, 1, 32, 32)

### Test 10: Memory Efficiency
- **Purpose**: Verify no memory leaks with large batches
- **Test Batch**: (16, 3, 64, 64) on GPU
- **Validation**: Memory usage < 1000 MB
- **Importance**: Prevents OOM errors during training

### Test 11: Device Transfer
- **Purpose**: Ensure augmentation works on CPU and GPU
- **Scenarios**:
  - CPU processing
  - GPU processing (if CUDA available)
- **Validation**: Output device matches input device

### Test 12: No-Gradient Mode (Inference)
- **Purpose**: Verify augmentation works in `torch.no_grad()` context
- **Importance**: Standard for inference and validation
- **Validation**:
  - ✓ No gradients in output
  - ✓ Shape preservation

### Test 13: Reproducibility
- **Purpose**: Test that fixed random seed produces identical results
- **Procedure**:
  1. Set seed=42, apply transform
  2. Set seed=42 again, apply transform
  3. Compare outputs
- **Validation**: Outputs are identical (tolerance: 1e-5)

## Configuration Details (from simclr_styletrans.yaml)

```yaml
batch_augmentations:
  style_transfer:
    enabled: true
    features_path: "$HOME/my_work/code/ssl-aug-benchmark/augmentation/mbda/features/style_feats_adain_1000.npy"
    encoder_path: "$HOME/my_work/code/ssl-aug-benchmark/augmentation/mbda/experiments/adaIN/vgg_normalised.pth"
    decoder_path: "$HOME/my_work/code/ssl-aug-benchmark/augmentation/mbda/experiments/adaIN/decoder.pth"
    
    # Blending strength [0, 1]
    alpha_min: 0.7
    alpha_max: 1.0
    
    # Probability of applying [0, 1]
    probability: 0.1
```

### Parameter Meanings

| Parameter | Range | Meaning | Default |
|-----------|-------|---------|---------|
| `alpha_min` | [0, 1] | Minimum blend of style | 0.7 |
| `alpha_max` | [0, 1] | Maximum blend of style | 1.0 |
| `probability` | [0, 1] | Chance to apply augmentation | 0.1 |

## Input/Output Flow

### Input to Style Transfer

```
Batch from DataLoader
    ↓
[B, C, H, W] format
    ↓
NSTTransform.__call__()
    ↓
With probability p:
  - Resize to (224, 224)
  - Select random style features
  - Extract content features via VGG
  - Apply AdaIN (Adaptive Instance Normalization)
  - Blend: feat = alpha*stylized + (1-alpha)*content
  - Decode back to image space
  - Resize to original dimensions
    ↓
[B, C, H, W] output
```

### Output Properties

- **Shape**: Same as input (e.g., [16, 3, 224, 224])
- **Dtype**: float32 (internally) → may return float32 or original dtype
- **Values**: [0, 1] normalized range
- **Gradient**: None (no_grad context)
- **Device**: Same as input (CPU/CUDA)

## Integration with Training Pipeline

### In BatchAugmentationMixin

```python
class SimCLR(BatchAugmentationMixin, BaseMethod):
    def __init__(self, cfg):
        super().__init__(cfg)
        self.setup_batch_augmentations(cfg)
    
    def training_step(self, batch, batch_idx):
        _, X, targets = batch
        X = [X] if isinstance(X, torch.Tensor) else X
        
        # Apply batch-level augmentations (style transfer)
        X = self.apply_batch_augmentations(X)
        
        # Continue with training
        outs = [self.base_training_step(x, targets) for x in X[:self.num_large_crops]]
        ...
```

### Augmentation Pipeline

1. **Image-level augmentations** (in dataloader):
   - Random resized crop
   - Color jitter
   - Random grayscale
   - Gaussian blur
   - Random horizontal flip

2. **Batch-level augmentations** (after dataloader):
   - Style transfer (NSTTransform) ← **Tested here**

## Troubleshooting

### Test Failures

#### "Device match failed"
- **Issue**: `cuda:0` vs `cuda` device mismatch
- **Fix**: Already handled in updated test script

#### "Transform not initialized"
- **Issue**: Import or model loading failed
- **Check**: Ensure all dependencies are installed
  ```bash
  pip install torch torchvision pytorch-lightning omegaconf
  ```

#### "FileNotFoundError for style features"
- **Issue**: Path to style features doesn't exist
- **Fix**: Update paths in config to match your system
  ```yaml
  features_path: "/absolute/path/to/style_feats_adain_1000.npy"
  ```

#### "CUDA out of memory"
- **Issue**: Batch size too large for GPU
- **Fix**: Reduce batch size in test or model config

### Common Issues

1. **Slow tests**: VGG inference is compute-intensive. Tests take ~30-60 seconds on RTX 4090.

2. **Alpha blending not visible**: With small images (32x32), style transfer may be subtle. Use larger images (224x224) for clearer effects.

3. **Grayscale conversion issues**: Ensure input has correct shape [B, 1, H, W] for grayscale.

## Test Coverage Summary

```
Style Transfer Module Tests: 13 major tests
├── Initialization (2 tests)
│   ├── Synthetic features creation
│   └── Synthetic VGG + Decoder creation
├── Input/Output Validation (4 tests)
│   ├── Single image input
│   ├── Batch input
│   ├── Different data types
│   └── Grayscale handling
├── Functionality (3 tests)
│   ├── Probability-based application
│   ├── Alpha blending strength
│   └── Reproducibility
└── Robustness (4 tests)
    ├── Memory efficiency
    ├── Device transfers
    ├── No-gradient mode
    └── Device transfers

Batch Augmentation Mixin Integration: 3 tests
├── Setup
├── Apply augmentations
└── Get statistics

Total: 16+ individual test cases
```

## Next Steps

### For Validation
1. Run test script before training experiments
2. Verify all input/output tests pass
3. Check memory usage on your hardware

### For Debugging
1. Add `print()` statements in `style_transfer.py`
2. Visualize augmented images:
   ```python
   import torchvision.transforms.v2 as transforms
   img = transform(batch[0]).cpu()
   transforms.ToPILImage()(img).show()
   ```

### For Integration
1. Ensure config paths are absolute or relative to working directory
2. Use `$HOME` or `~` in YAML for cross-machine compatibility
3. Test with actual training command:
   ```bash
   python main_pretrain.py --config-path scripts/pretrain/cifar/ \
     --config-name simclr_styletrans.yaml ++data.dataset=cifar100
   ```

## References

- **AdaIN Style Transfer**: [Arbitrary Style Transfer in Real-time with Adaptive Instance Normalization](https://arxiv.org/abs/1703.06868)
- **Your Config**: `learning/solo-learn/scripts/pretrain/cifar/simclr_styletrans.yaml`
- **Implementation**: `learning/solo-learn/solo/data/style_transfer.py`
- **Integration**: `learning/solo-learn/solo/methods/batch_augmentation_mixin.py`
